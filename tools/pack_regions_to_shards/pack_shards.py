#!/usr/bin/env python3
"""Pack the merged 100B corpus into the trainer's schema-v2 shards, reusing the trainer's own code.

The trainer (Small-LLM, Rocco) does not read region files. It reads "context_plus_one" shards:
2049-token uint16 records (documents concatenated with EOD, cut every 2048 tokens with the target
token carried over as the next record's first input), 64 records per block, ~1 GiB immutable shards
named train-NNNNNN.bin / validation-NNNNNN.bin, plus a manifest that its verify() checks field by
field. Nothing here reimplements that format: SequencePacker, PreparedBlockBuilder,
ImmutableShardWriter, TokenDeficitScheduler, is_validation and verify are imported from the
trainer's repository and driven with our documents.

Two deliberate deviations from the trainer's own producer, both recorded in the manifest:
  * documents are emitted in the merged corpus order — the deficit scheduler's re-stratification is
    bypassed (its emit() is still called so the scheduler block in the manifest is truthful);
  * the tokenizer contract is written from the file the corpus was actually built with (the worker
    contracts pin its sha256), not from install_superbpe_retokenization(), which would record the
    trainer repo's copy of superbpe_8000.json — a different tokenizer under the same name.

    python3 pack_shards.py --source-repo roccoangelella/small-llm-corpus-100b-v2 --regions 2100 \
        --run-id moe-100b-superbpe-b64-dataset-002 --out corpus-13b-v2/shards \
        [--bucket OWNER/NAME] [--no-upload] [--local-in DIR]
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import os
import shutil
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
TRAINER = Path(os.environ.get('SMALL_LLM_TRAINER_ROOT', str(ROOT.parents[1])))   # tools/pack_regions_to_shards/ -> repo root
sys.path.insert(0, str(TRAINER))
from dataset import config                                                          # noqa: E402
config.EOD_TOKEN_ID = 7992   # the trainer's classes insert config.EOD_TOKEN_ID; its default is GPT-2's 50256
from dataset.src.split import is_validation                                         # noqa: E402
from dataset.src.storage import write_json_atomic                                   # noqa: E402
from dataset.src.streaming import (SEQUENCE_FORMAT, STREAM_CACHE_SCHEMA_VERSION,   # noqa: E402
                                   ImmutableShardWriter, PreparedBlockBuilder,
                                   SequencePacker, SourceDocument, TokenDeficitScheduler)
from dataset.src.verify import verify                                               # noqa: E402
from dataset.incremental_frontier import (build_run_contract, publish_frontier,      # noqa: E402
                                          publish_run_contract)
from dataset.production.policy import stable_hash                                   # noqa: E402
from fast_packer import FastBlockPacker, FastSequencePacker                       # noqa: E402
BUILDER = Path(os.environ.get('SMALL_LLM_BUILDER_ROOT', '/home/edo/Documents/2_Code/study/small-lm-english-next'))
sys.path.insert(0, str(BUILDER))
from merge_workers import load_workers                                            # noqa: E402  (builder's receipt reader)

EOD = 7992
VOCAB = 8000
CONTEXT = 2048
SEQ_PER_BLOCK = 64
SHARD_BYTES = 1024 ** 3
STATE_FILE = 'pack-state.json'


_TOK = None


def _tok_init(path):
    global _TOK
    from tokenizers import Tokenizer
    _TOK = Tokenizer.from_file(path)


def _tok_encode(text):
    ids = _TOK.encode(text, add_special_tokens=False).ids
    if any(t >= 7992 for t in ids):
        raise RuntimeError('token id outside the 7992 learned entries')
    return ids


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()


def iter_documents(bin_path, jsonl_path: Path, region_index: int, as_tuple: bool = True, seen=None, retok=None):
    """Yield (is_validation, SourceDocument) for one merged region, in file order.

    Tokens are sliced out of the .bin by the per-document counters the jsonl carries; the trailing
    EOD the builder wrote after every document is dropped here because the packer inserts its own —
    the token stream is identical and source_token_count then means what the trainer means by it.
    """
    raw = bin_path.read_bytes() if bin_path is not None else None
    docs = []
    with gzip.open(jsonl_path, 'rt', encoding='utf-8') as fp:
        for line in fp:
            d = json.loads(line)
            if seen is not None:                 # cross-region exact dedup, same rule as merge_workers
                h = bytes.fromhex(d['id'])
                if h in seen:
                    continue
                seen.add(h)
            docs.append(d)
    if retok is not None:                        # new tokenizer: re-encode the stored text, ignore .bin
        ids_list = retok(d['text'] for d in docs)
    for j, d in enumerate(docs):
            if retok is not None:
                toks = np.asarray(ids_list[j], dtype=np.int64)
            else:
                off, n = d['token_offset_in_shard'], d['target_token_count']
                chunk = raw[2 * off: 2 * (off + n + 1)]
                if len(chunk) != 2 * (n + 1) or chunk[-2:] != EOD.to_bytes(2, 'little'):
                    raise RuntimeError(f'{jsonl_path.name}: token stream inconsistent at offset {off}')
                toks = np.frombuffer(chunk[:-2], dtype='<u2')
            if toks.size == 0:
                continue
            tokens = tuple(int(t) for t in toks) if as_tuple else toks.astype(np.int64)
            val = is_validation(seed=config.SELECTION_SEED, revision=d['source_revision'],
                                filename=d['source_file'], record_start=int(d['record_start']),
                                probability=config.VALIDATION_PROBABILITY)
            yield val, SourceDocument(source_id=f"{d['source_revision']}:{d['source_file']}:{d['record_start']}",
                                      cluster_id=int(d['cluster_id']), tokens=tokens,
                                      work_item_index=region_index, record_start=int(d['record_start']))


class Packer:
    """Train + validation pipelines with checkpointable state, so a 4-hour run can resume."""

    def __init__(self, out: Path, packer_cls=FastSequencePacker):
        self.out = out
        self.packer_cls = packer_cls
        w = {c: 1 for c in config.ACCEPTED_CLUSTER_IDS}
        self.scheduler = TokenDeficitScheduler(w, 'merged-corpus-arrival-order')
        self.fast = packer_cls is FastSequencePacker
        if self.fast:
            self.train_fb = FastBlockPacker(CONTEXT, SEQ_PER_BLOCK, 'train')
            self.val_fb = FastBlockPacker(CONTEXT, SEQ_PER_BLOCK, 'validation')
        else:
            self.train_packer = packer_cls(CONTEXT, final_partial_sequence_policy='pad_eod')
            self.val_packer = packer_cls(CONTEXT, final_partial_sequence_policy='pad_eod')
            self.train_blocks = PreparedBlockBuilder(SEQ_PER_BLOCK)
            self.val_blocks = PreparedBlockBuilder(SEQ_PER_BLOCK)
        self.train_writer = ImmutableShardWriter(out, split='train', target_bytes=SHARD_BYTES, context_length=CONTEXT)
        self.val_writer = ImmutableShardWriter(out, split='validation', target_bytes=SHARD_BYTES, context_length=CONTEXT)
        self.validation_source_tokens = 0
        self.documents = 0
        self.next_region = 0
        self.finalized = []          # ShardMetadata dicts in finalization order (+ remote info)
        self.seen = set()            # sha256 ids of every accepted document (dedup across regions)
        self.retok = None            # callable(texts) -> list[list[int]] when re-tokenizing
        self.dropped_duplicates = 0

    def _emit(self, packer, blocks, writer, split, doc, cumulative):
        seqs = packer.push(doc)
        out = []
        for s in seqs:
            b = blocks.push(s, split=split, cumulative_source_tokens=cumulative())
            if b is not None:
                out.extend(writer.write_block(b))
        return out

    def consume_region(self, idx, bin_path, jsonl_path):
        done = []
        cum_t = lambda: self.scheduler.total_emitted_source_tokens
        cum_v = lambda: self.validation_source_tokens
        before = len(self.seen)
        n_in = 0
        for val, doc in iter_documents(bin_path, jsonl_path, idx, as_tuple=not self.fast, seen=self.seen, retok=self.retok):
            self.documents += 1
            if val:
                self.validation_source_tokens += doc.source_token_count
                if self.fast:
                    for b in self.val_fb.push(doc, cum_v):
                        done += self.val_writer.write_block(b)
                else:
                    done += self._emit(self.val_packer, self.val_blocks, self.val_writer, 'validation', doc, cum_v)
            else:
                self.scheduler.emit(doc)
                if self.fast:
                    for b in self.train_fb.push(doc, cum_t):
                        done += self.train_writer.write_block(b)
                else:
                    done += self._emit(self.train_packer, self.train_blocks, self.train_writer, 'train', doc, cum_t)
        self.next_region = idx + 1
        return done

    def finish(self):
        done = []
        if self.fast:
            for b in self.train_fb.finish(lambda: self.scheduler.total_emitted_source_tokens):
                done += self.train_writer.write_block(b)
            done += self.train_writer.finish()
            for b in self.val_fb.finish(lambda: self.validation_source_tokens):
                done += self.val_writer.write_block(b)
            done += self.val_writer.finish()
            return done
        for packer, blocks, writer, split, cum in ((self.train_packer, self.train_blocks, self.train_writer, 'train', lambda: self.scheduler.total_emitted_source_tokens),
                                                   (self.val_packer, self.val_blocks, self.val_writer, 'validation', lambda: self.validation_source_tokens)):
            for s in packer.finish():
                b = blocks.push(s, split=split, cumulative_source_tokens=cum())
                if b is not None:
                    done.extend(writer.write_block(b))
            b = blocks.finish(split=split, cumulative_source_tokens=cum())
            if b is not None:
                done.extend(writer.write_block(b))
            done.extend(writer.finish())
        return done

    def manifest(self, run_id, source, tokenizer_sha256, tokenizer_path, complete):
        return {
            'schema_version': STREAM_CACHE_SCHEMA_VERSION, 'sequence_format': SEQUENCE_FORMAT,
            'context_length': CONTEXT, 'stored_tokens_per_sequence': CONTEXT + 1,
            'final_partial_sequence_policy': 'pad_eod', 'sequences_per_block': SEQ_PER_BLOCK,
            'target_shard_bytes': SHARD_BYTES,
            'weights': {'supplied': {str(k): v for k, v in self.scheduler.supplied_weights.items()},
                        'normalized_integer_units': {str(k): v for k, v in self.scheduler.weight_units.items()}},
            'shards': [dict(s) for s in self.finalized],
            'scheduler': self.scheduler.state_dict(),
            'accepted_source_tokens': self.scheduler.total_emitted_source_tokens + self.validation_source_tokens,
            'validation_source_tokens': self.validation_source_tokens,
            'accepted_document_count': self.documents,
            'mixture': {'emitted_source_tokens_per_cluster': {str(k): v for k, v in self.scheduler.emitted_source_tokens.items()}},
            'last_durable_block_id': self.train_writer.shards[-1].last_block_id if self.train_writer.shards else -1,
            'last_durable_train_block_id': self.train_writer.shards[-1].last_block_id if self.train_writer.shards else -1,
            'last_durable_validation_block_id': self.val_writer.shards[-1].last_block_id if self.val_writer.shards else -1,
            'tokenizer': {'source_tokenizer_id': 'gpt2', 'output_tokenizer_id': 'superbpe_8000',
                          'tokenizer_artifact': str(tokenizer_path), 'tokenizer_sha256': tokenizer_sha256,
                          'semantic_vocab_size': VOCAB, 'eod_token_id': EOD},
            'semantic_vocab_size': VOCAB, 'eod_token_id': EOD, 'source_token_unit': 'superbpe_8000',
            'production': {'run_id': run_id, 'target_reached': complete, 'producer': 'pack_shards.py',
                           'ordering': 'merged corpus order (work-plan region order, original record order); deficit scheduler bypassed',
                           'source_corpus': source},
        }

    def state(self):
        return {'next_region': self.next_region, 'documents': self.documents,
                'validation_source_tokens': self.validation_source_tokens,
                'scheduler': self.scheduler.state_dict(),
                'fast': self.fast,
                **({'train_fb': self.train_fb.state_dict(), 'val_fb': self.val_fb.state_dict()} if self.fast else
                   {'train_packer': self.train_packer.state_dict(), 'val_packer': self.val_packer.state_dict(),
                    'train_blocks': self.train_blocks.state_dict(), 'val_blocks': self.val_blocks.state_dict()}),
                'train_writer_index': self.train_writer._index, 'val_writer_index': self.val_writer._index,
                'train_cumulative': {str(k): v for k, v in self.train_writer._cumulative_counts.items()},
                'val_cumulative': {str(k): v for k, v in self.val_writer._cumulative_counts.items()},
                'finalized': self.finalized}

    def load_state(self, st):
        self.next_region = st['next_region']; self.documents = st['documents']
        self.validation_source_tokens = st['validation_source_tokens']
        w = {c: 1 for c in config.ACCEPTED_CLUSTER_IDS}
        self.scheduler = TokenDeficitScheduler.from_state(w, st['scheduler'])
        if st.get('fast'):
            self.fast = True
            self.train_fb = FastBlockPacker.from_state(st['train_fb']); self.val_fb = FastBlockPacker.from_state(st['val_fb'])
        else:
            self.fast = False
            self.train_packer = self.packer_cls.from_state(st['train_packer']); self.val_packer = self.packer_cls.from_state(st['val_packer'])
            self.train_blocks = PreparedBlockBuilder.from_state(st['train_blocks']); self.val_blocks = PreparedBlockBuilder.from_state(st['val_blocks'])
        self.train_writer = ImmutableShardWriter(self.out, split='train', target_bytes=SHARD_BYTES, context_length=CONTEXT, start_index=st['train_writer_index'])
        self.val_writer = ImmutableShardWriter(self.out, split='validation', target_bytes=SHARD_BYTES, context_length=CONTEXT, start_index=st['val_writer_index'])
        self.train_writer._cumulative_counts = {int(k): v for k, v in st['train_cumulative'].items()}
        self.val_writer._cumulative_counts = {int(k): v for k, v in st['val_cumulative'].items()}
        self.finalized = st['finalized']
        for split, w in (('train', self.train_writer), ('validation', self.val_writer)):
            d = self.out / split
            if d.exists():
                for p in d.glob('.*.bin.tmp'):
                    p.unlink()
                for p in d.glob(f'{split}-*.bin'):
                    if int(p.stem.split('-')[-1]) >= w._index:
                        p.unlink()
        # writers rebuild their own shard lists only for files still on disk; the manifest uses self.finalized
        for s in self.finalized:
            (self.train_writer if s['split'] == 'train' else self.val_writer).shards.append(_Row(s))


class _Row:
    def __init__(self, d): self.__dict__.update(d)


def fetch_region(repo, token, idx, dst, local_in, worker_dir, receipt, need_bin=True):
    """Materialise region idx into dst and hash-check it against its receipt.

    Sources, in order: --local-in (flat region-N.* files), the worker dir on this machine (the big
    files are usually gone: streamed to the hub and deleted), the workers repo (worker-X/region-N.*).
    """
    stem = f'region-{idx:06d}'
    want = ['jsonl.gz'] + (['bin'] if need_bin else [])
    for e in want:
        dst_p = dst / f'{stem}.{e}'
        if dst_p.exists():
            continue
        if local_in and (local_in / f'{stem}.{e}').exists():
            shutil.copy2(local_in / f'{stem}.{e}', dst_p)
        elif worker_dir is not None and (worker_dir / f'{stem}.{e}').exists():
            shutil.copy2(worker_dir / f'{stem}.{e}', dst_p)
        else:
            from huggingface_hub import hf_hub_download
            p = hf_hub_download(repo, f'{worker_dir.name}/{stem}.{e}', repo_type='dataset', token=token,
                                local_dir=dst / '.dl')
            Path(p).replace(dst_p)
    for e, k in (('bin', 'tokens_sha256'), ('jsonl.gz', 'texts_sha256')):
        if e in want and sha256_file(dst / f'{stem}.{e}') != receipt[k]:
            raise RuntimeError(f'{stem}.{e}: sha256 != receipt')
    return receipt


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--source-repo', default='roccoangelella/small-llm-corpus-100b-v2-workers')
    ap.add_argument('--root', type=Path, default=BUILDER / 'corpus-13b-v2', help='worker dirs with receipts + index.sqlite')
    ap.add_argument('--retokenize', type=Path, default=None, help='HF tokenizer.json: re-encode the text instead of using the stored tokens')
    ap.add_argument('--tok-workers', type=int, default=8)
    ap.add_argument('--regions', type=int, required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--bucket', default=None, help='HF dataset bucket for the trainer (run/<run_id>/...)')
    ap.add_argument('--no-upload', action='store_true')
    ap.add_argument('--local-in', type=Path, default=None, help='dir with region files already downloaded')
    ap.add_argument('--tokenizer', type=Path, default=TRAINER / 'tokenizer/superbpe_8000_v2.json',
                    help='the tokenizer the regions were built with (sha256 pinned in every worker contract)')
    ap.add_argument('--prefetch', type=int, default=4)
    ap.add_argument('--nominal-training-tokens', type=int, default=100_000_000_000,
                    help='trainer horizon: planned_train_blocks = ceil(N / (2048*64)) = 762,940 for 100e9')
    ap.add_argument('--stop-margin-blocks', type=int, default=64,
                    help='stop packing once train blocks >= planned + margin (surplus is never read)')
    ap.add_argument('--public', action='store_true', help='create the bucket public (the paused run\'s was)')
    ap.add_argument('--crash-after', type=int, default=None, help='TEST ONLY: exit right after this region is checkpointed')
    ap.add_argument('--checkpoint-regions', type=int, default=10,
                    help='every K regions: finalize active shards (validation becomes READY early), save state')
    ap.add_argument('--packer', choices=('fast', 'slow'), default='fast', help='slow = the trainer\'s own SequencePacker (reference)')
    a = ap.parse_args()
    token = os.environ.get('HF_TOKEN')
    import glob as _glob
    dirs = sorted(Path(p) for p in _glob.glob(str(a.root / 'worker-*')) if Path(p).is_dir())
    _, plan_hash, regions = load_workers(dirs)
    missing = [i for i in range(a.regions) if i not in regions]
    if missing:
        sys.exit(f'{len(missing)} regions have no receipt: {missing[:10]}')
    a.out.mkdir(parents=True, exist_ok=True)
    window = a.out / '.window'; window.mkdir(exist_ok=True)
    tok_path = a.retokenize or a.tokenizer
    tok_sha = sha256_file(tok_path)
    store = None
    if not a.no_upload:
        from dataset.src.hf_bucket_shards import HuggingFaceBucketShardStore
        store = HuggingFaceBucketShardStore(a.bucket, token=token, private=not a.public, create_bucket=True)

    packer = Packer(a.out, FastSequencePacker if a.packer == 'fast' else SequencePacker)
    tokenizer_block = {'source_tokenizer_id': 'gpt2', 'output_tokenizer_id': 'superbpe_8000',
                       'tokenizer_artifact': str(tok_path), 'tokenizer_sha256': tok_sha,
                       'semantic_vocab_size': VOCAB, 'eod_token_id': EOD}
    # Same hash recipes as the trainer's producer (dataset/production/policy.py), fed with this packer's
    # configuration: the contract is identity + horizon for the consumer, not a resume key for us.
    schema_h = stable_hash({'stream_cache_schema_version': STREAM_CACHE_SCHEMA_VERSION, 'sequence_format': 'context_plus_one',
                            'context_length': CONTEXT, 'stored_sequence_tokens': CONTEXT + 1, 'sequences_per_block': SEQ_PER_BLOCK,
                            'int_type': config.INT_TYPE, 'byte_order': config.BYTE_ORDER, 'tokenizer': tokenizer_block})
    config_h = stable_hash({'producer': 'pack_shards.py', 'source_repo': a.source_repo, 'regions': a.regions,
                            'work_plan_hash': plan_hash, 'context_length': CONTEXT, 'sequences_per_block': SEQ_PER_BLOCK,
                            'target_shard_bytes': SHARD_BYTES, 'retokenized': a.retokenize is not None, 'tokenizer_sha256': tok_sha,
                            'ordering': 'merged corpus order; deficit scheduler bypassed'})
    contract = build_run_contract(run_id=a.run_id, nominal_training_tokens=a.nominal_training_tokens,
                                  target_source_tokens=a.nominal_training_tokens, minimum_source_tokens=int(a.nominal_training_tokens * 0.9),
                                  maximum_source_tokens=int(a.nominal_training_tokens * 1.1), checkpoint_source_tokens=500_000_000,
                                  context_length=CONTEXT, sequences_per_block=SEQ_PER_BLOCK, target_shard_bytes=SHARD_BYTES,
                                  configuration_hash=config_h, schema_hash=schema_h, work_plan_hash=plan_hash)
    planned_blocks = int(contract['planned_train_blocks'])
    write_json_atomic(a.out / 'run_contract.json', contract)
    if store is not None:
        publish_run_contract(store, run_id=a.run_id, contract=contract)
        print(json.dumps({'event': 'contract', 'planned_train_blocks': planned_blocks, 'sha256': contract['contract_sha256'][:16]}), flush=True)
    if a.retokenize:
        from multiprocessing import Pool
        pool_tok = Pool(a.tok_workers, initializer=_tok_init, initargs=(str(a.retokenize),))
        packer.retok = lambda texts: pool_tok.map(_tok_encode, list(texts), chunksize=256)
    st_path = a.out / STATE_FILE
    if st_path.exists():
        packer.load_state(json.loads(st_path.read_text()))
        sb = (a.out / 'seen.bin').read_bytes() if (a.out / 'seen.bin').exists() else b''
        packer.seen = {sb[i:i + 32] for i in range(0, len(sb), 32)}
        print(json.dumps({'event': 'resume', 'next_region': packer.next_region, 'shards': len(packer.finalized)}), flush=True)

    # Uploads run on their own thread so a 1 GiB shard (about 60 s on this link) never stalls the
    # packing loop; measured: 11 s/region with synchronous uploads, ~5 s/region without. One worker
    # keeps the bucket writes sequential; the frontier is published inside the task, after the
    # store has verified the upload. checkpoint()/finish() drain the queue before persisting state,
    # so the saved state never claims a shard that is not yet durable.
    up_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix='upload')
    up_futs = []

    def _upload_task(d):
        local = a.out / d['filename']
        if store is not None:
            info = store.upload_finalized_shard(run_id=a.run_id, logical_name=d['filename'], local_path=local)
            d.update({'remote_durable': True, 'file_id': info['file_id'], 'remote_sha256': info['sha256']})
            local.unlink()
        packer.finalized.append(d)
        print(json.dumps({'shard': d['filename'], 'blocks': [d['first_block_id'], d['last_block_id']],
                          'bytes': d['byte_size'], 'uploaded': store is not None}), flush=True)
        if store is not None:
            fr = publish_frontier(store, run_id=a.run_id, contract=contract,
                                  durability_manifest={'shards': [x for x in packer.finalized if x.get('remote_durable')]})
            print(json.dumps({'event': 'frontier', 'last_ready_train_block_id': fr['last_ready_train_block_id'],
                              'ready_train_shards': len(fr['ready_train_shards']), 'validation_ready': fr['validation_ready']}), flush=True)
        return d['filename']

    def durable(rows):
        for m in rows:
            d = m.as_dict() if hasattr(m, 'as_dict') else dict(m)
            up_futs.append(up_pool.submit(_upload_task, d))
        for f in [f for f in up_futs if f.done()]:
            f.result()                      # surface an upload failure immediately

    def drain_uploads():
        for f in up_futs:
            f.result()
        up_futs.clear()

    pool = ThreadPoolExecutor(max_workers=a.prefetch)
    futs = {}
    def sched(i):
        if i < a.regions and i not in futs:
            d, r = regions[i]
            futs[i] = pool.submit(fetch_region, a.source_repo, token, i, window, a.local_in, d, r, a.retokenize is None)
    for j in range(packer.next_region, min(a.regions, packer.next_region + a.prefetch)):
        sched(j)
    def checkpoint(idx):
        """Finalize whatever is in the active shards, make it durable, then persist a complete state.

        Mirrors the trainer's producer (finalize_active_shards_for_checkpoint at each durable
        checkpoint): after this call no .tmp shard holds blocks the saved state does not know about,
        so a resume can discard every .tmp and regenerate from next_region deterministically.
        """
        rows = []
        for w in (packer.train_writer, packer.val_writer):
            if w._blocks:
                rows.append(w.finalize_active())
        durable(rows)
        drain_uploads()
        write_json_atomic(st_path, packer.state())
        with open(a.out / 'seen.bin.part', 'wb') as f:
            f.write(b''.join(packer.seen))
        os.replace(a.out / 'seen.bin.part', a.out / 'seen.bin')
        print(json.dumps({'event': 'checkpoint', 'after_region': idx, 'shards': len(packer.finalized),
                          'train_blocks': packer.train_fb.block_id_counter[0] if packer.fast else packer.train_blocks._block_id_counter[0]}), flush=True)

    t0 = time.time()
    n_done = 0
    for idx in range(packer.next_region, a.regions):
        sched(idx + a.prefetch)
        r = futs.pop(idx).result()
        stem = f'region-{idx:06d}'
        rows = packer.consume_region(idx, None if a.retokenize else window / f'{stem}.bin', window / f'{stem}.jsonl.gz')
        durable(rows)
        for e in ('bin', 'jsonl.gz'):
            (window / f'{stem}.{e}').unlink(missing_ok=True)
        n_done += 1
        train_blocks_done = (packer.train_fb.block_id_counter[0] if packer.fast else packer.train_blocks._block_id_counter[0])
        if idx % 10 == 0 or idx == a.regions - 1:
            print(json.dumps({'region': idx, 'docs': packer.documents, 'train_tokens': packer.scheduler.total_emitted_source_tokens,
                              'val_tokens': packer.validation_source_tokens, 'train_blocks': train_blocks_done,
                              'shards': len(packer.finalized), 's_per_region': round((time.time() - t0) / n_done, 1)}), flush=True)
        if train_blocks_done >= planned_blocks + a.stop_margin_blocks:
            print(json.dumps({'event': 'horizon_reached', 'train_blocks': train_blocks_done, 'planned': planned_blocks, 'at_region': idx}), flush=True)
            checkpoint(idx)
            break
        if (idx + 1) % a.checkpoint_regions == 0:
            checkpoint(idx)
            if a.crash_after is not None and idx == a.crash_after:
                print(json.dumps({'event': 'simulated_crash', 'after_region': idx}), flush=True); sys.exit(3)
    durable(packer.finish())
    drain_uploads()
    up_pool.shutdown(wait=True)
    pool.shutdown(wait=True)
    train_blocks_total = (packer.train_fb.block_id_counter[0] if packer.fast else packer.train_blocks._block_id_counter[0])
    horizon_ok = train_blocks_total >= planned_blocks
    manifest = packer.manifest(a.run_id, {'repo': a.source_repo, 'regions': a.regions, 'plan_hash': plan_hash,
                                          'cross_region_duplicates_dropped': packer.dropped_duplicates,
                                          'retokenized': a.retokenize is not None}, tok_sha, tok_path, horizon_ok)
    manifest['production']['completion_reason'] = 'train_horizon_reached' if horizon_ok else 'source_exhausted_before_horizon'
    manifest['production']['planned_train_blocks'] = planned_blocks
    manifest['production']['train_blocks'] = train_blocks_total
    manifest['work_plan_hash'] = plan_hash
    manifest['remote_transport'] = {'backend': 'hf_bucket' if store is not None else 'local_only', 'evict_local_finalized_shards': store is not None,
                                    'hf_bucket_id': a.bucket, 'hf_bucket_private': (not a.public) if store is not None else None, 'incremental_frontier': True}
    write_json_atomic(a.out / config.MANIFEST_FILENAME, manifest)
    if store is None:
        rep = verify(a.out, full_scan=False)
        print(json.dumps({'verify_passed': rep.passed, 'problems': rep.problems,
                          'train_tokens': rep.train_token_count, 'validation_tokens': rep.validation_token_count}), flush=True)
        if not rep.passed:
            sys.exit(2)
    else:
        ready = store.publish_dataset_manifest(run_id=a.run_id, manifest_path=a.out / config.MANIFEST_FILENAME)
        fr = publish_frontier(store, run_id=a.run_id, contract=contract,
                              durability_manifest={'shards': [x for x in packer.finalized if x.get('remote_durable')]},
                              producer_complete=horizon_ok, final_manifest_sha256=sha256_file(a.out / config.MANIFEST_FILENAME))
        print(json.dumps({'published': ready, 'frontier_complete': fr['producer_complete']}), flush=True)
    print(json.dumps({'event': 'done', 'documents': packer.documents,
                      'train_source_tokens': packer.scheduler.total_emitted_source_tokens,
                      'validation_source_tokens': packer.validation_source_tokens,
                      'shards': len(packer.finalized)}), flush=True)


if __name__ == '__main__':
    main()
