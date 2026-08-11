---
status: accepted
date: 2026-08-11
supersedes: null
---

# 0036 — Add a local completed-SFT chat CLI

## Context and problem statement

The project now needs a minimal human-facing way to interact with whichever Small-LLM SFT profile has actually completed and been published to Hugging Face. The interface should select a model by parameter profile and parent pretraining token count, use the exact S0 chat serialization seen during fine-tuning, and run locally without creating a second inference/model implementation.

The current registered SFT profiles are 20M/500M and 20M/2B. The 500M-parent SFT lane is implemented but, at the time of this decision, has not yet been accepted as completed GPU evidence. Therefore the chat loader must not equate the existence of an intermediate `latest` checkpoint with completed SFT.

Recent sub-1B inference stacks commonly add KV/state caching, quantization, or a dedicated serving engine. Those are intentionally out of scope here: the current hybrid Small-LLM model does not yet expose a unified GDN-2/MHA generation-cache contract, and this CLI is a correctness/convenience surface rather than a new serving architecture.

## Considered options

- Build a separate inference/serving implementation with its own model loader and prompt format.
- Load the newest SFT checkpoint regardless of whether the SFT trajectory finished.
- Reuse the verified native checkpoint path, exact S0 template, and existing generation sampler, while refusing incomplete SFT checkpoints.

## Decision outcome

Chosen option: **add a root-level `chat.py` that reuses the verified native checkpoint format, exact `small-llm-s0-v1` generation template, and existing sampler, and only loads a completed Hugging Face SFT trajectory**.

The user-facing command is:

```bash
python chat.py --model_params 20M --num_tokens 500M
```

with `--model-params` / `--num-tokens` accepted as aliases. `--num_tokens` denotes the parent pretraining-token profile used to identify the SFT run, not the generation length.

Sampling is intentionally edited directly at the first lines of `chat.py`:

```python
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.95
```

The CLI resolves `SMALL_LLM_SFT_HF_REPO_ID`, falling back to `SMALL_LLM_HF_REPO_ID` when SFT and base checkpoints share one repository, and uses `HF_TOKEN` when authentication is required.

A downloaded checkpoint is accepted only when all of the following hold:

- the repository/run `latest` pointer resolves and the checkpoint tree passes existing manifest verification;
- the checkpoint pipeline identity says `stage=sft_s0`;
- the embedded trainer schedule is WSD;
- `consumed_tokens` exactly equals `warmup_tokens + stable_tokens + decay_tokens`.

Generation uses CUDA automatically when available and CPU otherwise. CUDA therefore follows the already-qualified FLA GDN-2 path; CPU follows the adaptive PyTorch fallback. Conversation history is serialized with the exact SFT template and old complete turns are dropped as needed to reserve answer space inside the 2,048-token context.

## Consequences

### Positive

- One simple command selects the intended SFT model profile and chats locally.
- Intermediate SFT publications cannot silently masquerade as a finished instruction model.
- Prompt formatting is training-compatible rather than an ad-hoc chat string.
- No second model implementation or checkpoint conversion format is introduced.
- Temperature, top-k, and top-p are immediately editable at the top of the script as requested.

### Negative or limiting

- The current model has no unified cached decode path, so generation recomputes the retained prefix for each new token and is not a production serving benchmark.
- CUDA first-use latency can include Triton/FLA JIT work when the local Triton cache has no matching compiled kernels.
- Only explicitly registered SFT profiles are accepted; future parameter/token profiles must be added deliberately.
- The native trainer checkpoint contains optimizer state, so the loader temporarily deserializes more data than a future inference-only export would require.

## Validation

- `tests/test_chat_cli.py` covers profile quantity parsing, fail-closed profile selection, and context-history trimming.
- After an SFT trajectory completes, run the CLI against its Hugging Face repository and verify that a completed final checkpoint loads while an earlier intermediate checkpoint fails the completion check.
- Exercise one CUDA chat session and one CPU session when practical; CUDA should use the qualified FLA backend and CPU should avoid Triton entirely.

## Links

- [`../runbooks/local_sft_chat.md`](../runbooks/local_sft_chat.md)
- [`../reference/post_training_sft.md`](../reference/post_training_sft.md)
- [`0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md`](0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md)
- [`0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md`](0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md)
