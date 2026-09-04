# ADR 0148: Register the completed 100M/10B SFT for local chat

- Status: Accepted
- Date: 2026-09-04

## Context

The SFT run `100m-10b-sft-s0-2b10pct-data-001` completed its 100M/10B-parent trajectory and published its final verified checkpoint. The root-level `chat.py` already supports completed SFT S0 checkpoints through the normal GPT-2 tokenizer and the verified Hugging Face `latest` checkpoint path, but the `(100M, 10B)` SFT profile was not present in `_SFT_CHAT_RUNS`. Because `_resolve_chat_run()` requires the model/token profile to be registered before applying an optional `--run-id`, the completed 100M/10B SFT artifact could not be selected through the supported local chat CLI.

## Decision

Register `(100_000_000, 10_000_000_000)` in the SFT chat registry with default run:

```text
100m-10b-sft-s0-2b10pct-data-001
```

The canonical invocation is:

```bash
python chat.py --model_params 100M --num_tokens 10B --sft
```

Keep the rest of the local-chat contract unchanged:

- the checkpoint source is SFT, using `SMALL_LLM_SFT_HF_REPO_ID` with the existing `SMALL_LLM_HF_REPO_ID` fallback;
- the checkpoint must pass verified-manifest loading;
- the pipeline identity must be `stage=sft_s0`;
- the trainer state must represent a fully completed supported schedule;
- the model must use the ordinary GPT-2 semantic vocabulary of 50,257 tokens;
- the standard SFT `GPT2ChatTemplate` remains the conversation serialization;
- no tokenizer conversion or inference-specific model conversion is introduced;
- the `(100M, 10B)` pretrained profile remains unregistered and therefore continues to fail closed under `--pre-trained`.

## Verification

Focused CLI coverage must assert that:

1. `10B` parses as `10_000_000_000`;
2. the `(100M, 10B, --sft)` resolver selects `100m-10b-sft-s0-2b10pct-data-001` from the SFT source;
3. the complete CLI argument path resolves the same run;
4. `(100M, 10B, --pre-trained)` remains rejected because no stable pretrained chat artifact has been registered for that profile.

## Consequences

The completed 100M/10B SFT model can now be used directly with the same fail-closed local chat path as the other SFT checkpoints. This is a registry/presentation change only; it does not alter training, evaluation, checkpoint publication, sampling, tokenizer behavior, or model architecture.
