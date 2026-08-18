---
status: accepted
date: 2026-08-18
---

# ADR 0100 — Promote the completed atomic pilot to the accepted R0 R-SFT artifact

## Context

The 100M / 2B R0 delimiter ablation completed both matched runs. ADR 0099 subsequently selected the atomic special-token protocol for future reasoning stages even though the tiny textual arm had the lower teacher-forced validation loss, because reasoning boundaries are semantic control concepts and must remain unambiguous from ordinary natural-language text.

The atomic run already trained the full frozen 630-example R0 corpus committed at `artifacts/rsft-r0-pilot-630/generation/reasoning.jsonl`, together with the accepted 10% S0 instruction-retention lane, using the frozen atomic protocol:

```text
50257  <think>
50258  </think>
50259  <answer>
```

The completed artifact is `100m-2b-rsft-r0-atomic-pilot-001`. Its training summary reports one complete pass, 29 optimizer steps, 54,571 consumed loss-bearing target tokens, and held-out validation loss 2.4455797088124576 on 1,653 validation target tokens. It used the completed `100m-2b-sft-s0-001` parent and carries the promoted 50,260 semantic vocabulary plus reasoning-token metadata.

Running the same frozen corpus again only to replace the word `pilot` in the run identity would duplicate compute without adding scientific information.

## Decision

Promote the already-completed run `100m-2b-rsft-r0-atomic-pilot-001` to the accepted 100M / 2B R0 R-SFT artifact.

- Do not retrain the same 630-example corpus under a new production run ID.
- Do not rename or rewrite the existing checkpoint identity. Scientific provenance keeps the run ID exactly as produced.
- Register `(100M, 2B, --r-sft)` in `chat.py` to resolve this artifact.
- Give R-SFT a dedicated artifact-source resolution path: prefer `SMALL_LLM_RSFT_HF_REPO_ID`, then fall back to `SMALL_LLM_SFT_HF_REPO_ID`, then `SMALL_LLM_HF_REPO_ID`.
- Chat loading must fail closed unless the checkpoint is complete, uses semantic vocabulary size 50,260, carries `pipeline_state.rsft_format` with `stage=r_sft_r0` and `delimiter_format=atomic`, and carries the exact accepted `<think>`, `</think>`, `<answer>` token contract.
- The textual delimiter checkpoint remains ablation evidence only and is not a registered R-SFT chat artifact or an accepted parent for later reasoning stages.

## Consequences

The accepted interactive command is now:

```bash
python chat.py --model_params 100M --num_tokens 2B --r-sft
```

The model lifecycle may refer to `100m-2b-rsft-r0-atomic-pilot-001` as the accepted R0 checkpoint despite the historical `pilot` suffix. The suffix records how the artifact was produced; it no longer means the checkpoint is unaccepted.

Future R-SFT expansion, deeper reasoning stages, reasoning qualification, and later RLVR remain separate decisions. This promotion does not change the training data, weights, tokenizer, or checkpoint bytes.
