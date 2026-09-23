"""Remote-only lifetime and corpus guards; never modify numerical training state."""
from __future__ import annotations

import datetime
import json
import re
import tempfile
import time
from pathlib import Path

from dataset.incremental_frontier import read_frontier, read_run_contract
from dataset.src.joint_checkpoint import _verify_published_checkpoint_manifest


def now():
    return datetime.datetime.now(datetime.timezone.utc).isoformat()


class CorpusHold(RuntimeError):
    pass


class ClaimBlocked(RuntimeError):
    pass


def validate_segment(segment_id):
    if not re.fullmatch(r"[a-zA-Z0-9-]{8,80}", segment_id):
        raise ValueError("durable dispatch requires an explicit unique segment ID")


def reserve(frontier, contract, step):
    if (frontier.get('contract_sha256') != contract.get('contract_sha256')
            or frontier.get('run_id') != contract.get('run_id')
            or frontier.get('validation_ready') is not True
            or type(frontier.get('producer_complete')) is not bool):
        raise RuntimeError('Invalid frontier identity or readiness')
    expected = 0
    for row in frontier.get('ready_train_shards', []):
        if (not isinstance(row, dict) or type(row.get('first_block_id')) is not int
                or row['first_block_id'] != expected or type(row.get('last_block_id')) is not int
                or row['last_block_id'] < expected):
            raise RuntimeError('Noncontiguous READY coverage')
        expected = row['last_block_id'] + 1
    planned = frontier.get('planned_train_blocks')
    if (type(planned) is not int or planned != contract.get('planned_train_blocks')
            or not step <= expected <= planned
            or frontier.get('last_ready_train_block_id') != expected - 1):
        raise RuntimeError('Invalid frontier bounds')
    if frontier['producer_complete'] and (expected != planned or not re.fullmatch(
            r'[0-9a-f]{64}', frontier.get('final_manifest_sha256') or '')):
        raise RuntimeError('Producer completion without complete coverage')
    return expected - step


class RemoteControl:
    def __init__(self, request, registry, checkpoints, dataset, *, margin=5000):
        if not request.streaming or not request.checkpoint_bucket:
            raise ValueError('Durable mode requires dataset and checkpoint buckets')
        if margin < 2 * request.checkpoint_every_steps:
            raise ValueError('Frontier margin must cover at least two checkpoint intervals')
        self.request = request
        self.registry = registry
        self.checkpoints = checkpoints
        self.dataset = dataset
        self.margin = margin
        self.run = request.run_id
        self.hold_key = f'run/{self.run}/control/data-hold.json'
        self.contract = read_run_contract(dataset, run_id=request.dataset_shard_run_id)
        self.last_step = 0
        self.last_report = 0.0
        self.has_update = False

    def frontier(self):
        return read_frontier(self.dataset, run_id=self.request.dataset_shard_run_id,
                             contract=self.contract)

    def check_launch(self, step):
        frontier = self.frontier()
        remaining = reserve(frontier, self.contract, step)
        hold = self.checkpoints.read_json(self.hold_key)
        if hold and not frontier['producer_complete']:
            raise CorpusHold('Existing remote hold requires complete producer coverage')
        if not frontier['producer_complete'] and remaining <= self.margin:
            self.persist_hold(frontier, f'step-{step:08d}', 'preflight')
            raise CorpusHold('Insufficient READY reserve for an autonomous segment')
        return frontier

    def prepare_claim(self, segment_id):
        validate_segment(segment_id)
        # Reserve a unique segment before claiming the run; neither key is silently recycled.
        if not self.registry.put(('segment', self.run, segment_id), {
                'phase': 'preparing', 'at': now(), 'segment_id': segment_id,
                'source_commit': self.request.source_commit,
                'run_source_commit': self.request.resume_source_commit or self.request.source_commit,
        }, skip_if_exists=True):
            raise ClaimBlocked('Segment ID already used; inspect its existing record')
        if not self.registry.put(('active', self.run), segment_id, skip_if_exists=True):
            self.status(segment_id, 'blocked_claim')
            raise ClaimBlocked('Remote run already claimed; reconcile before dispatch')

    def claim_training(self, segment_id, call_id):
        validate_segment(segment_id)
        if self.registry.get(('active', self.run)) != segment_id:
            raise ClaimBlocked('Durable segment does not own run claim')
        if not self.registry.put(('attempt', self.run, segment_id), {
                'call_id': call_id, 'at': now()}, skip_if_exists=True):
            raise ClaimBlocked('GPU attempt already started; automatic replay refused')
        self.status(segment_id, 'restoring', call_id=call_id)

    def status(self, segment_id, phase, **fields):
        key = ('segment', self.run, segment_id)
        record = dict(self.registry.get(key) or {})
        record.update(at=now(), phase=phase, last_observed_step=self.last_step, **fields)
        self.registry.put(key, record)

    def finish(self, segment_id, result):
        self.status(segment_id, result['status'], result=result)
        # Keep the lease on failure/ambiguity. Only normal terminal state releases it.
        if result['status'] in ('complete', 'completed', 'drained', 'data_hold'):
            if self.registry.get(('active', self.run)) != segment_id:
                raise RuntimeError('Run ownership changed before release')
            self.registry.pop(('active', self.run))

    def verify_checkpoint(self, checkpoint_id):
        pointer = self.checkpoints.read_json(f'run/{self.run}/latest.json')
        prefix = f'run/{self.run}/checkpoints/{checkpoint_id}/last'
        if (not pointer or pointer.get('checkpoint_id') != checkpoint_id
                or pointer.get('last_prefix') != prefix):
            raise RuntimeError('Published latest does not match checkpoint event')
        with tempfile.TemporaryDirectory(prefix='remote-checkpoint-verify-') as tmp:
            root = Path(tmp)
            self.checkpoints.download_tree(prefix, root)
            _verify_published_checkpoint_manifest(root, pointer['checkpoint_manifest'])
            metadata = json.loads((root / 'checkpoint.json').read_text())
        step = int(checkpoint_id[5:])
        pipeline = metadata.get('pipeline_state', {})
        if (metadata.get('checkpoint_id') != checkpoint_id
                or metadata.get('optimizer_step_complete') is not True
                or pipeline.get('last_consumed_block_id') != step - 1
                or pipeline.get('gradient_accumulation_position') != 0):
            raise RuntimeError('Incomplete optimizer or data state in checkpoint')
        return pointer

    def observe(self, line, segment_id):
        try:
            event = json.loads(line)
        except ValueError:
            return
        if not isinstance(event, dict):
            return
        if 'step' in event and 'loss' in event:
            self.last_step = int(event['step'])
            self.has_update = True
            if time.monotonic() - self.last_report >= 60:
                self.status(segment_id, 'running', loss=event['loss'],
                            tokens_per_second=event.get('tokens_per_second'))
                self.last_report = time.monotonic()
        published = event.get('remote_publication')
        if not isinstance(published, dict):
            return
        checkpoint_id = published.get('checkpoint_id')
        if not isinstance(checkpoint_id, str) or not re.fullmatch(r'step-\d{8}', checkpoint_id):
            raise RuntimeError('Invalid remote checkpoint event')
        step = int(checkpoint_id[5:])
        frontier = self.frontier()
        remaining = reserve(frontier, self.contract, step)
        self.status(segment_id, 'running' if self.has_update else 'restoring', checkpoint=checkpoint_id,
                    ready_blocks_remaining=remaining)
        if frontier['producer_complete'] or remaining > self.margin:
            return
        self.persist_hold(frontier, checkpoint_id, segment_id)
        # The provider wrapper kills its child on this exception, after durable hold.
        raise CorpusHold('Verified checkpoint and hold committed; stop before data frontier')

    def persist_hold(self, frontier, checkpoint_id, segment_id):
        step = int(checkpoint_id[5:])
        pointer = self.verify_checkpoint(checkpoint_id)
        hold = {'at': now(), 'reason': 'waiting_for_complete_corpus', 'run_id': self.run,
                'segment_id': segment_id, 'checkpoint': checkpoint_id,
                'step': step, 'hf_latest': pointer, 'frontier': frontier,
                'full_checkpoint_hashes_verified': True,
                'resume_requires': 'producer_complete and complete coverage plus preflight'}
        self.checkpoints.write_json(self.hold_key, hold)
        if self.checkpoints.read_json(self.hold_key) != hold:
            raise RuntimeError('Remote hold read-back mismatch')
