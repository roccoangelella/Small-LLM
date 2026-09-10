---
status: accepted
date: 2026-09-10
supersedes: null
---

# 0175 — Freeze the 8,000-token SuperBPE tokenizer for the MoE line

> Branch sequence. `main` carries different ADRs at 0172-0174 (Rocco's tokenizer-corpus decisions, since removed from `main`),
> and this branch already uses 0168-0174 for the pilot, the Triton seed, the three efficiency contracts and the accepted
> geometry/router. 0175 is the next free number **on this branch**; a merge to `main` will need a renumbering pass.
> This ADR **amends 0171 and 0173 of the tokenizer-corpus series**; see "Amendments" below.

## Context

The MoE geometry (8 layers, d_model 256, 64 experts top-2, ~9.9M active parameters) cannot carry the inherited GPT-2
vocabulary: at V=50,304 the tied embedding alone is 12.9M parameters against ~42M stored, and the output head dominates the
forward pass. ADR 0170 set an 8k-class target and left the algorithm, the special-token inventory, the training sample and the
acceptance metrics open. This ADR closes those.

## Decision

**Vocabulary: exactly 8,000 ids** = 7,192 word-level BPE tokens (byte alphabet included) + 800 cross-word "superword" merges
+ 8 special tokens.

**Algorithm: SuperBPE** (arXiv 2503.13423, COLM 2025) — byte-level BPE in two stages, the second lifting the whitespace
restriction so tokens may bridge spaces. Trained with `kensho-technologies/fastboundlessbpe` (Apache-2.0, commit c2dce31,
2026-07-02), the fast re-implementation from arXiv 2604.05192. Pre-tokenization regex: GPT-4o pattern in simple mode
(no script-specific splitting; the corpus is English). Merge eligibility requires a letter, so digits never fuse with letters.

**Special tokens, at the end of the range, reserved now because adding them later would resize a trained embedding:**

```text
7992  <|endoftext|>
7993  <think>        (ADR 0099)
7994  </think>       (ADR 0099)
7995  <answer>       (ADR 0099)
7996  <|reserved_0|>
7997  <|reserved_1|>
7998  <|reserved_2|>
7999  <|reserved_3|>
```

**Training data.** Stage 1 on 10,000,004,822 effective UTF-8 bytes reconstructed from the frozen
`roccoangelella/small-llm-100m-qualification-datasets` bucket (run `modal-10b-b64-dataset-001`, `producer_complete: true`,
final manifest `d23e7e46…`), 5 shards sampled across the ready range. Stage 2 on a 2.0 GB prefix of the same corpus: 800
supermerges saturate far below 10 GB, and the SuperBPE FAQ states that "subwords and superwords can be learned over different
data". Held-out for all measurements: shard `train-000019`, never used for training.

**Distributed artifact: a standard Hugging Face `tokenizer.json`** (`tokenizer/superbpe_8000.json`), so the trainer, the
evaluator and the providers carry no runtime dependency on the tokenizer's build toolchain.

## Amendments to existing decisions

- **ADR 0171** chose to reconstruct from the pinned ClimbMix source rather than invert the packed shards. This work inverts the
  packed shards instead, on the owner's decision of 2026-09-10: the packed stream **is** the production mixture, so the
  tokenizer is learned on exactly the distribution the model will see. The context+1 overlap is undone before decoding
  (verified: the last token of each stored sequence equals the first of the next on 255/255 boundaries of a real shard) and
  documents are cut on GPT-2 EOD 50256; the reconstructed text contains no literal EOD marker, as 0173 requires.
- **ADR 0173** chose four balanced clusters rather than the production mixture. Not applied, same reason and same decision.
  The cluster-balance question stays open for any future tokenizer, not for this one.

## Measurements (20,000 held-out documents, 57.1 MB)

| | bytes/token | tokens/doc | superword share | round-trip failures |
|---|---:|---:|---:|---:|
| GPT-2 50,257 (reference on this corpus) | 4.665 | — | — | — |
| **SuperBPE 8,000** | **4.1725** | **684.3** | **7.19 %** | **0** |
| plain BPE 8,000 (control arm) | 3.887 | 734.5 | 0 % | 0 |

SuperBPE compresses **7.3 % better than plain BPE at the same vocabulary**, i.e. 6.8 % fewer tokens for the same text.
Against GPT-2 it is 10.6 % worse, which is the price of dropping from 50,257 to 8,000 ids and buys back ~10.8M parameters.
Learned superwords are conversational formulas: ` of the`, ` in the`, ` is a`, ` by the way`, ` you know`, ` kind of`, ` I think`.

**Compression is a cost measurement, not a quality one.** The SuperBPE FAQ states plainly that the efficiency-optimal
transition point is not necessarily the best for downstream performance and that predicting tokenizer quality from intrinsic
features is unsolved. The choice between these two arms, and any reopening of 8k versus 16k, requires a proxy pretraining
compared in **bits per byte** — never in loss per token, because the prediction unit changes with the tokenizer.
`eval_core_v1` already reports bits per decoded byte.

## Verification

- Byte completeness: the 13 alphabet entries absent from the vocabulary are `0xC0`, `0xC1` and `0xF5`–`0xFF`, exactly the
  bytes that **cannot occur in valid UTF-8**. Round-trip is exact on 500 random byte strings, on all C0 control characters,
  on tabs/newlines/CRLF, on emoji, and on CJK. No `unk` token is defined and none is reachable.
  (The control arm initially lacked 30 control bytes; it was retrained with the full ByteLevel alphabet.)
- Corpus reconstruction: 15,863 GPT-2 decode/re-encode round-trips verified during detokenization, 0 failures.
- Export parity against the reference implementation on 5,000 held-out documents: token counts differ by 0.09 %
  (3,439,925 versus 3,436,739), 0 round-trip failures, byte-alphabet mismatches 0.
  The residual difference is structural and is the same trade-off the public SuperBPE release carries: the reference applies
  supermerges only between whole pretokens, while Hugging Face BPE without a regex may also apply them across word fragments.
  **The distributed `tokenizer.json` is the authority**: data and inference use the same file, so they cannot disagree.

## Consequences

- The 10B GPT-2 corpus and `eval_core_v1` must be regenerated with this tokenizer before any MoE run; the dense 200M/100B line
  on `main` keeps GPT-2, so two corpora coexist.
- Comparisons against the dense endpoints are valid only in bits per byte.
- Reproduction: `piani/06-TOKENIZER-8K-PIANO-ESECUTIVO.md` in the knowledge hub carries the exact commands, timings
  (stage 1: 1,735 s single-core; stage 2: 527 s) and the known failure modes.
