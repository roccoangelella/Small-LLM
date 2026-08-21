---
status: accepted
date: 2026-08-21
supersedes: null
---

# ADR 0116 — Promote expanded 3-epoch R-SFT as the current default R0

## Context and problem statement

The completed historical accepted R-SFT R0 run `100m-2b-rsft-r0-12306-001` remains the default target selected by `chat.py --model_params 100M --num_tokens 2B --r-sft`. That run ends at `step-00000361` and was trained on the earlier 12,306-row corpus.

The expanded 16,716-row atomic production corpus has now also completed a three-epoch run, `100m-2b-rsft-r0-16716-e3-001`. W&B records the run as finished at logical optimizer step 1,251 (417 train blocks × 3 exact passes), with 40,262,469 consumed loss-bearing targets and final validation loss approximately 1.454987.

The user wants this completed expanded three-epoch model to replace the older R0 as the default model used for local R-SFT chat for now.

## Decision outcome

Promote `100m-2b-rsft-r0-16716-e3-001` as the current default/accepted R-SFT R0 chat target for the 100M/2B profile.

Concretely:

- `python chat.py --model_params 100M --num_tokens 2B --r-sft` resolves to `100m-2b-rsft-r0-16716-e3-001`.
- The older `100m-2b-rsft-r0-12306-001` remains preserved as a historical completed run and may still be loaded explicitly with `--run-id`.
- This is a provisional promotion ('for now'): future qualification or scaling evidence may replace the default again without deleting either completed artifact.
- No training recipe, dataset, tokenizer, checkpoint format, or inference decoding policy changes as part of this decision; only the accepted/default R-SFT run selection changes.

## Consequences

- Bare `--r-sft` chat now exercises the completed expanded-corpus, three-epoch model rather than the 12,306-row step-361 model.
- Tests and current project status should treat `100m-2b-rsft-r0-16716-e3-001` as the current default R-SFT R0 while retaining the older run as historical evidence.
- Explicit `--run-id` remains the mechanism for comparing historical or experimental R-SFT checkpoints.