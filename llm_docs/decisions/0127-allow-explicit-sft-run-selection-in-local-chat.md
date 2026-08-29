# ADR 0127: Allow explicit SFT run selection in local chat

- Status: Accepted
- Date: 2026-08-28

## Context

The 100M/2B SFT chat registry intentionally keeps the previously qualified S0 trajectory `100m-2b-sft-s0-001` as the bare `--sft` default. The newer 10% S0 experiment uses the separate run identity `100m-2b-sft-s0-10pct-001`, and the project owner wants to inspect that checkpoint directly without silently replacing the registered default before its full qualification is reviewed.

Before this decision, `chat.py --run-id` was restricted to R-SFT, so a completed non-default SFT trajectory could not be selected through the supported local chat path.

## Decision

Allow `chat.py --sft --run-id RUN_ID` to select an explicit SFT trajectory for an otherwise registered model/token profile.

Keep all existing fail-closed loading rules:

- explicit run IDs are allowed only for SFT and R-SFT, never pretrained artifacts;
- SFT uses the normal GPT-2 tokenizer contract;
- the selected checkpoint must carry the SFT S0 pipeline identity;
- the verified Hugging Face checkpoint must be complete, including equality between consumed loss-bearing targets and the frozen full schedule horizon;
- bare `--sft` continues to resolve the registered default and is not repointed by this change.

The 100M/2B 10% experiment can therefore be inspected with:

```bash
python chat.py --model_params 100M --num_tokens 2B --sft --run-id 100m-2b-sft-s0-10pct-001
```

## Consequences

- Experimental SFT checkpoints can be inspected without changing the qualified/default registry pointer.
- A stale or partial `latest` checkpoint still fails closed rather than being treated as a completed chat model.
- Promotion of the 10% SFT run to the bare `--sft` default remains a separate decision after qualification.

## Implementation

Implemented and pushed to `main` in commit `51e3ab1` (`Allow explicit SFT chat run selection`). The local chat runbook and focused resolver/parser coverage were updated in the same change.
