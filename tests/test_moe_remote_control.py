import json
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from moe_remote_control import RemoteControl, CorpusHold, reserve


class Registry(dict):
    def put(self, key, value, *, skip_if_exists=False):
        if skip_if_exists and key in self:
            return False
        self[key] = value
        return True


def frontier(ready=20000, complete=False):
    return dict(run_id='data', contract_sha256='hash', validation_ready=True,
                producer_complete=complete, planned_train_blocks=20000,
                last_ready_train_block_id=ready-1,
                ready_train_shards=[dict(first_block_id=0, last_block_id=ready-1)],
                final_manifest_sha256='a'*64 if complete else None)


class ControlTest(unittest.TestCase):
    def setUp(self):
        self.registry = Registry()
        self.checkpoints = Mock()
        self.checkpoints.read_json.return_value = None
        self.request = SimpleNamespace(streaming=True, checkpoint_bucket='checkpoint-bucket',
            checkpoint_every_steps=2500, run_id='training', dataset_shard_run_id='data',
            source_commit='a'*40, resume_source_commit='b'*40)
        self.contract = dict(run_id='data', contract_sha256='hash', planned_train_blocks=20000)
        with patch('moe_remote_control.read_run_contract', return_value=self.contract):
            self.control = RemoteControl(self.request, self.registry, self.checkpoints, Mock())
        self.control.frontier = Mock(return_value=frontier())

    def test_claim_refuses_second_writer_and_replayed_input(self):
        self.control.prepare_claim('segment-1')
        with self.assertRaises(RuntimeError):
            self.control.prepare_claim('segment-2')
        self.control.claim_training('segment-1', 'fc-1')
        with self.assertRaises(RuntimeError):
            self.control.claim_training('segment-1', 'fc-1')
        self.assertEqual(self.registry[('attempt', 'training', 'segment-1')]['call_id'], 'fc-1')

    def test_failed_segment_keeps_lease_after_client_exit(self):
        self.control.prepare_claim('segment-1')
        self.control.claim_training('segment-1', 'fc-1')
        self.control.finish('segment-1', {'status': 'failed'})
        self.assertEqual(self.registry[('active', 'training')], 'segment-1')
        with self.assertRaises(RuntimeError):
            self.control.prepare_claim('segment-2')

    def test_clean_drain_releases_run_but_never_old_attempt(self):
        self.control.prepare_claim('segment-1')
        self.control.claim_training('segment-1', 'fc-1')
        self.control.finish('segment-1', {'status': 'drained'})
        self.assertNotIn(('active', 'training'), self.registry)
        self.control.prepare_claim('segment-2')
        with self.assertRaises(RuntimeError):
            self.control.claim_training('segment-1', 'fc-1')

    def test_reusing_completed_segment_cannot_reacquire_run(self):
        self.control.prepare_claim('segment-1')
        self.control.claim_training('segment-1', 'fc-1')
        self.control.finish('segment-1', {'status': 'complete'})
        with self.assertRaisesRegex(RuntimeError, 'already used'):
            self.control.prepare_claim('segment-1')
        self.assertNotIn(('active', 'training'), self.registry)

    def test_lost_local_dispatch_receipt_still_exposes_remote_call(self):
        self.control.prepare_claim('segment-1')
        self.control.claim_training('segment-1', 'fc-survives-client')
        self.assertNotIn(('dispatch', 'training', 'segment-1'), self.registry)
        self.assertEqual(self.registry[('segment', 'training', 'segment-1')]['call_id'],
                         'fc-survives-client')

    def test_persisted_hold_requires_complete_producer(self):
        self.checkpoints.read_json.return_value = {'reason': 'waiting_for_complete_corpus'}
        with self.assertRaises(CorpusHold):
            self.control.check_launch(100)
        self.control.frontier.return_value = frontier(complete=True)
        self.control.check_launch(100)

    def test_no_gpu_launch_with_small_ready_reserve(self):
        self.control.verify_checkpoint = Mock(return_value={})
        saved = {}
        self.checkpoints.write_json.side_effect = lambda key, value: saved.update(value)
        self.checkpoints.read_json.side_effect = lambda key: dict(saved)
        with self.assertRaises(CorpusHold):
            self.control.check_launch(15000)

    def test_gap_and_false_completion_rejected(self):
        invalid = frontier()
        invalid['ready_train_shards'][0]['first_block_id'] = 1
        with self.assertRaises(RuntimeError):
            reserve(invalid, self.contract, 0)
        with self.assertRaises(RuntimeError):
            reserve(frontier(19000, complete=True), self.contract, 0)

    def test_verified_hold_persisted_before_stop(self):
        self.control.prepare_claim('segment-1')
        self.control.verify_checkpoint = Mock(return_value={'checkpoint_id': 'step-00015000'})
        saved = {}
        self.checkpoints.write_json.side_effect = lambda key, value: saved.update(value)
        self.checkpoints.read_json.side_effect = lambda key: dict(saved)
        with self.assertRaises(CorpusHold):
            self.control.observe(json.dumps({'remote_publication': {'checkpoint_id': 'step-00015000'}}), 'segment-1')
        self.assertTrue(saved['full_checkpoint_hashes_verified'])
        self.assertEqual(saved['step'], 15000)
        self.control.verify_checkpoint.assert_called_once_with('step-00015000')

    def test_unverifiable_snapshot_cannot_claim_safe_hold(self):
        self.control.verify_checkpoint = Mock(side_effect=RuntimeError('hash mismatch'))
        with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
            self.control.observe(json.dumps({'remote_publication': {'checkpoint_id': 'step-00015000'}}), 'segment-1')
        self.checkpoints.write_json.assert_not_called()

    def test_running_updates_only_add_control_status(self):
        event = {'step': 100, 'loss': 3.0}
        self.control.observe(json.dumps(event), 'segment-1')
        self.assertEqual(self.control.last_step, 100)
        self.assertEqual(self.registry[('segment', 'training', 'segment-1')]['loss'], 3.0)
        self.checkpoints.write_json.assert_not_called()


if __name__ == '__main__':
    unittest.main()
