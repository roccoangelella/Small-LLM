"""Vectorised drop-in for the trainer's SequencePacker: same records, same attribution, 10x faster.

The trainer's packer keeps one Python object per token, which costs ~50 s per 50M-token region.
This one keeps numpy arrays for token values, cluster ids and kinds, plus one source-id per document,
and cuts records with array slicing. The contract it must honour, from SequencePacker._drain:

  record k  = buffer[0:2049]; then the buffer advances by 2048 (the target token is carried over as
              the next record's first input, flagged overlap_*);
  counts    = source tokens per cluster in the record, excluding index 0 when it is an overlap;
  first/last source id = of the buffer's first/last token at the moment the record is cut;
  finish()  = pad with EOD (kind "padding", source "<padding>") to 2049 and cut once.

The known-answer test in pack_shards.py --self-test drives both packers over the same regions and
requires byte-identical shards and identical manifests before this one is allowed to run for real.
"""
from __future__ import annotations

import numpy as np

from dataset import config
from dataset.src.streaming import PackedSequence, SourceDocument

KIND = {'source': 0, 'inserted_eod': 1, 'padding': 2}
KIND_NAME = {0: 'source', 1: 'inserted_eod', 2: 'padding'}
OVERLAP_NAME = {0: 'overlap_source', 1: 'overlap_eod', 2: 'overlap_eod'}


class FastSequencePacker:
    def __init__(self, context_length: int, *, final_partial_sequence_policy: str = 'pad_eod') -> None:
        if final_partial_sequence_policy not in {'pad_eod', 'error'}:
            raise ValueError("final_partial_sequence_policy must be 'pad_eod' or 'error'")
        self.context_length = context_length
        self.final_partial_sequence_policy = final_partial_sequence_policy
        self.sequence_tokens = context_length + 1
        self._val = np.zeros(0, dtype=np.int64)
        self._cl = np.zeros(0, dtype=np.int64)      # -1 == None
        self._kind = np.zeros(0, dtype=np.int8)
        self._doc = np.zeros(0, dtype=np.int64)     # index into self._src
        self._src: list[str] = []
        self._has_overlap = False
        self._cluster_counts: dict[int, int] = {}

    # ---- input -------------------------------------------------------------------------------
    def push(self, document: SourceDocument) -> list[PackedSequence]:
        n = len(document.tokens)
        add_eod = not document.tokens or document.tokens[-1] != config.EOD_TOKEN_ID
        m = n + (1 if add_eod else 0)
        val = np.empty(m, dtype=np.int64); val[:n] = document.tokens
        cl = np.full(m, document.cluster_id, dtype=np.int64)
        kind = np.zeros(m, dtype=np.int8)
        if add_eod:
            val[n] = config.EOD_TOKEN_ID; cl[n] = -1; kind[n] = KIND['inserted_eod']
        self._src.append(document.source_id)
        doc = np.full(m, len(self._src) - 1, dtype=np.int64)
        self._val = np.concatenate([self._val, val]); self._cl = np.concatenate([self._cl, cl])
        self._kind = np.concatenate([self._kind, kind]); self._doc = np.concatenate([self._doc, doc])
        self._cluster_counts[document.cluster_id] = self._cluster_counts.get(document.cluster_id, 0) + document.source_token_count
        return self._drain()

    # ---- cutting -----------------------------------------------------------------------------
    def _drain(self) -> list[PackedSequence]:
        L = self.sequence_tokens; C = self.context_length
        out: list[PackedSequence] = []
        total = len(self._val)
        if total < L:
            return out
        nrec = (total - L) // C + 1
        # windows: record i covers [i*C, i*C + L)
        idx = np.arange(L)[None, :] + (np.arange(nrec) * C)[:, None]
        V = self._val[idx]; CL = self._cl[idx]; K = self._kind[idx]; D = self._doc[idx]
        overlap = np.zeros(nrec, dtype=bool)
        overlap[0] = self._has_overlap; overlap[1:] = True
        for i in range(nrec):
            first_new = 1 if overlap[i] else 0
            cls = CL[i, first_new:]; ks = K[i, first_new:]
            mask = (ks == 0) & (cls >= 0)
            counts: dict[int, int] = {}
            if mask.any():
                u, c = np.unique(cls[mask], return_counts=True)
                counts = {int(a): int(b) for a, b in zip(u, c)}
            kinds = [KIND_NAME[int(k)] for k in K[i]]
            if overlap[i]:
                kinds[0] = OVERLAP_NAME[int(K[i, 0])]
            clusters = tuple(None if c < 0 else int(c) for c in CL[i])
            out.append(PackedSequence(tuple(int(x) for x in V[i]), self._src[int(D[i, 0])], self._src[int(D[i, -1])],
                                      counts, tuple(kinds), clusters))
        cut = nrec * C
        self._val = self._val[cut:]; self._cl = self._cl[cut:]; self._kind = self._kind[cut:]; self._doc = self._doc[cut:]
        self._has_overlap = len(self._val) > 0
        self._cluster_counts = {}
        # drop source ids no longer referenced (keep indices valid by remapping)
        if len(self._doc):
            lo = int(self._doc.min()); self._src = self._src[lo:]; self._doc = self._doc - lo
        else:
            self._src = []
        return out

    def finish(self) -> list[PackedSequence]:
        if len(self._val) == 0:
            return []
        if self.final_partial_sequence_policy == 'error':
            raise RuntimeError('final partial sequence exists; choose pad_eod to preserve it')
        pad = self.sequence_tokens - len(self._val)
        self._src.append('<padding>')
        self._val = np.concatenate([self._val, np.full(pad, config.EOD_TOKEN_ID, dtype=np.int64)])
        self._cl = np.concatenate([self._cl, np.full(pad, -1, dtype=np.int64)])
        self._kind = np.concatenate([self._kind, np.full(pad, KIND['padding'], dtype=np.int8)])
        self._doc = np.concatenate([self._doc, np.full(pad, len(self._src) - 1, dtype=np.int64)])
        out = self._drain()
        self._val = self._val[:0]; self._cl = self._cl[:0]; self._kind = self._kind[:0]; self._doc = self._doc[:0]
        self._src = []; self._has_overlap = False
        return out

    # ---- state, in the trainer's own carry format so either packer can resume it ------------
    def state_dict(self) -> dict[str, object]:
        carry = [{'value': int(v), 'kind': KIND_NAME[int(k)], 'cluster_id': None if c < 0 else int(c), 'source_id': self._src[int(d)]}
                 for v, k, c, d in zip(self._val, self._kind, self._cl, self._doc)]
        return {'context_length': self.context_length, 'final_partial_sequence_policy': self.final_partial_sequence_policy,
                'carry': carry, 'carry_tokens': [int(v) for v in self._val], 'has_overlap': self._has_overlap,
                'first_source_id': self._src[int(self._doc[0])] if len(self._doc) else None,
                'last_source_id': self._src[int(self._doc[-1])] if len(self._doc) else None,
                'cluster_source_tokens': {str(k): v for k, v in self._cluster_counts.items()}}

    @classmethod
    def from_state(cls, state) -> 'FastSequencePacker':
        inst = cls(int(state['context_length']), final_partial_sequence_policy=str(state['final_partial_sequence_policy']))
        carry = state.get('carry') or []
        if carry:
            src: list[str] = []; doc = []
            for item in carry:
                s = str(item['source_id'])
                if not src or src[-1] != s:
                    src.append(s)
                doc.append(len(src) - 1)
            inst._val = np.array([int(i['value']) for i in carry], dtype=np.int64)
            inst._cl = np.array([-1 if i.get('cluster_id') is None else int(i['cluster_id']) for i in carry], dtype=np.int64)
            inst._kind = np.array([KIND.get(str(i['kind']).replace('overlap_', ''), 0) if str(i['kind']) != 'overlap_eod' else KIND['inserted_eod'] for i in carry], dtype=np.int8)
            inst._doc = np.array(doc, dtype=np.int64); inst._src = src
        if len(inst._val) >= inst.sequence_tokens:
            raise ValueError('invalid packer carry: it already contains a full sequence')
        inst._has_overlap = bool(state.get('has_overlap', False))
        inst._cluster_counts = {int(k): int(v) for k, v in dict(state.get('cluster_source_tokens', {})).items()}
        return inst


# ---------------------------------------------------------------------------------------------
# Block-level path: never materialises per-record Python tuples. Profiling the record-level path
# put 30 of 42 seconds per region in building 2049-int tuples and kind lists for ~24k records.
# Here records stay as numpy windows until a block of 64 is complete, and the block payload is one
# astype('<u2').tobytes(). PreparedBlock is the trainer's own class; ImmutableShardWriter consumes it
# unchanged. The pending (<64) records are held as numpy arrays and serialised only at checkpoints.
# ---------------------------------------------------------------------------------------------
from dataset.src.streaming import PreparedBlock  # noqa: E402


class FastBlockPacker:
    """SequencePacker + PreparedBlockBuilder for one split, vectorised. push(doc) -> [PreparedBlock]."""

    def __init__(self, context_length: int, sequences_per_block: int, split: str, block_id_counter: list[int] | None = None):
        self.C = context_length; self.L = context_length + 1; self.B = sequences_per_block; self.split = split
        self.block_id_counter = block_id_counter if block_id_counter is not None else [0]
        self._val = np.zeros(0, dtype=np.int64); self._cl = np.zeros(0, dtype=np.int64)
        self._kind = np.zeros(0, dtype=np.int8); self._doc = np.zeros(0, dtype=np.int64)
        self._src: list[str] = []
        self._has_overlap = False
        # pending complete records not yet grouped into a block
        self._pv: list[np.ndarray] = []; self._pcounts: list[dict[int, int]] = []
        self._pfirst: list[str] = []; self._plast: list[str] = []
        self._ptok = 0

    # -- documents in -----------------------------------------------------------------------
    def push(self, document, cumulative_source_tokens) -> list[PreparedBlock]:
        toks = np.asarray(document.tokens, dtype=np.int64)
        n = len(toks)
        add_eod = n == 0 or int(toks[-1]) != config.EOD_TOKEN_ID
        m = n + (1 if add_eod else 0)
        val = np.empty(m, dtype=np.int64); val[:n] = toks
        cl = np.full(m, document.cluster_id, dtype=np.int64); kind = np.zeros(m, dtype=np.int8)
        if add_eod:
            val[n] = config.EOD_TOKEN_ID; cl[n] = -1; kind[n] = 1
        self._src.append(document.source_id)
        doc = np.full(m, len(self._src) - 1, dtype=np.int64)
        self._val = np.concatenate([self._val, val]); self._cl = np.concatenate([self._cl, cl])
        self._kind = np.concatenate([self._kind, kind]); self._doc = np.concatenate([self._doc, doc])
        return self._drain(cumulative_source_tokens)

    # -- records out, blocks out --------------------------------------------------------------
    def _drain(self, cumulative) -> list[PreparedBlock]:
        L, C = self.L, self.C
        total = len(self._val)
        if total < L:
            return []
        nrec = (total - L) // C + 1
        idx = np.arange(L)[None, :] + (np.arange(nrec) * C)[:, None]
        V = self._val[idx]; CL = self._cl[idx]; K = self._kind[idx]; D = self._doc[idx]
        overlap = np.ones(nrec, dtype=bool); overlap[0] = self._has_overlap
        # per-record source counts per cluster, excluding index 0 on overlapped records
        mask = (K == 0) & (CL >= 0)
        mask[overlap, 0] = False
        for i in range(nrec):
            cls = CL[i][mask[i]]
            if cls.size:
                u, c = np.unique(cls, return_counts=True)
                counts = {int(a): int(b) for a, b in zip(u, c)}
            else:
                counts = {}
            self._pv.append(V[i]); self._pcounts.append(counts)
            self._pfirst.append(self._src[int(D[i, 0])]); self._plast.append(self._src[int(D[i, -1])])
        cut = nrec * C
        self._val = self._val[cut:]; self._cl = self._cl[cut:]; self._kind = self._kind[cut:]; self._doc = self._doc[cut:]
        self._has_overlap = len(self._val) > 0
        if len(self._doc):
            lo = int(self._doc.min()); self._src = self._src[lo:]; self._doc = self._doc - lo
        else:
            self._src = []
        out = []
        while len(self._pv) >= self.B:
            out.append(self._make(self.B, cumulative))
        return out

    def _make(self, k: int, cumulative) -> PreparedBlock:
        recs, self._pv = self._pv[:k], self._pv[k:]
        counts, self._pcounts = self._pcounts[:k], self._pcounts[k:]
        firsts, self._pfirst = self._pfirst[:k], self._pfirst[k:]
        lasts, self._plast = self._plast[:k], self._plast[k:]
        per_cluster: dict[int, int] = {}
        for c in counts:
            for a, b in c.items():
                per_cluster[a] = per_cluster.get(a, 0) + b
        payload = np.stack(recs).astype('<u2').tobytes()
        bid = self.block_id_counter[0]; self.block_id_counter[0] += 1
        return PreparedBlock(block_id=bid, split=self.split, sequence_count=len(recs), token_count=len(recs) * self.L,
                             payload=payload, cumulative_source_tokens=cumulative(),
                             per_cluster_source_tokens=per_cluster, first_source_id=firsts[0], last_source_id=lasts[-1])

    def finish(self, cumulative) -> list[PreparedBlock]:
        out = []
        if len(self._val):
            pad = self.L - len(self._val)
            self._src.append('<padding>')
            self._val = np.concatenate([self._val, np.full(pad, config.EOD_TOKEN_ID, dtype=np.int64)])
            self._cl = np.concatenate([self._cl, np.full(pad, -1, dtype=np.int64)])
            self._kind = np.concatenate([self._kind, np.full(pad, 2, dtype=np.int8)])
            self._doc = np.concatenate([self._doc, np.full(pad, len(self._src) - 1, dtype=np.int64)])
            out += self._drain(cumulative)
            self._val = self._val[:0]; self._cl = self._cl[:0]; self._kind = self._kind[:0]; self._doc = self._doc[:0]
            self._src = []; self._has_overlap = False
        if self._pv:
            out.append(self._make(len(self._pv), cumulative))
        return out

    # -- checkpoint ---------------------------------------------------------------------------
    def state_dict(self) -> dict[str, object]:
        return {'context_length': self.C, 'sequences_per_block': self.B, 'split': self.split,
                'next_block_id': self.block_id_counter[0],
                'carry': {'val': [int(x) for x in self._val], 'cl': [int(x) for x in self._cl],
                          'kind': [int(x) for x in self._kind], 'doc': [int(x) for x in self._doc], 'src': list(self._src),
                          'has_overlap': self._has_overlap},
                'pending': [{'tokens': [int(x) for x in v], 'counts': {str(k): n for k, n in c.items()}, 'first': f, 'last': l}
                            for v, c, f, l in zip(self._pv, self._pcounts, self._pfirst, self._plast)]}

    @classmethod
    def from_state(cls, st) -> 'FastBlockPacker':
        inst = cls(int(st['context_length']), int(st['sequences_per_block']), str(st['split']), [int(st['next_block_id'])])
        c = st['carry']
        inst._val = np.array(c['val'], dtype=np.int64); inst._cl = np.array(c['cl'], dtype=np.int64)
        inst._kind = np.array(c['kind'], dtype=np.int8); inst._doc = np.array(c['doc'], dtype=np.int64)
        inst._src = list(c['src']); inst._has_overlap = bool(c['has_overlap'])
        for p in st['pending']:
            inst._pv.append(np.array(p['tokens'], dtype=np.int64)); inst._pcounts.append({int(k): int(v) for k, v in p['counts'].items()})
            inst._pfirst.append(p['first']); inst._plast.append(p['last'])
        return inst
