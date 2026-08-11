# Local completed-SFT chat

Use the root-level `chat.py` to download a completed SFT checkpoint from the configured private Hugging Face model repository and run an interactive local chat loop.

## Prerequisites

Install both model and post-training extras:

```bash
pip install -e '.[model,post-training]'
```

Configure Hugging Face access:

```bash
export HF_TOKEN=...
export SMALL_LLM_SFT_HF_REPO_ID=owner/repository
```

If SFT checkpoints are published into the same model repository as base checkpoints, `SMALL_LLM_HF_REPO_ID` is accepted instead of `SMALL_LLM_SFT_HF_REPO_ID`.

## Run

500M-parent SFT:

```bash
python chat.py --model_params 20M --num_tokens 500M
```

2B-parent SFT:

```bash
python chat.py --model_params 20M --num_tokens 2B
```

Hyphenated spellings are aliases:

```bash
python chat.py --model-params 20M --num-tokens 500M
```

`--num_tokens` identifies the parent pretraining-token profile. Generation length is controlled by `MAX_NEW_TOKENS` near the top of `chat.py`.

## Sampling settings

Edit the first lines of `chat.py` directly:

```python
TEMPERATURE = 0.8
TOP_K = 50
TOP_P = 0.95
MAX_NEW_TOKENS = 256
SEED = 17
```

`TEMPERATURE = 0` switches the existing sampler to greedy decoding. `TOP_K = 0` disables top-k filtering and `TOP_P = 1.0` disables nucleus filtering.

## Completion gate

The CLI does not accept merely-present intermediate SFT checkpoints. It downloads the run's verified Hugging Face `latest` checkpoint and requires:

1. valid checkpoint/local manifests;
2. SFT pipeline identity with `stage=sft_s0`;
3. a WSD trainer schedule;
4. `consumed_tokens == warmup_tokens + stable_tokens + decay_tokens`.

If no SFT pointer exists, the repository cannot be read, or the latest SFT checkpoint is still partial, the command exits with an error instead of falling back to the base model.

## Interactive commands

- `/clear` drops conversation history.
- `/quit` or `/exit` exits.
- `Ctrl-C` / EOF also exits cleanly.

The conversation is serialized with the exact `small-llm-s0-v1` SFT generation template. When history grows too long, the oldest complete user/assistant pairs are discarded until the retained prompt leaves room for `MAX_NEW_TOKENS` inside the model context.

## Device and Triton behavior

Device selection is automatic:

- CUDA available: FP16 autocast plus the qualified FLA GDN-2 backend.
- No CUDA: CPU FP32 plus the adaptive PyTorch GDN-2 fallback.

On CUDA, FLA calls Triton JIT kernels. A machine/environment with no matching Triton cache entry can pay compilation latency on first use. Triton maintains a persistent cache under its configured cache directory (by default beneath `~/.triton/`), so identical kernels normally do not need to be recompiled on every token or every process launch while that cache remains valid. Clearing the cache or changing relevant compiler/runtime/device/kernel specialization inputs can cause compilation again.

CPU execution does not use the FLA CUDA path and therefore does not compile Triton kernels.

## Current limitation

`SmallLLM` does not yet expose a unified cached decode contract across GDN-2 and MHA layers. The existing sampler therefore recomputes the retained prefix for each generated token. This CLI is intended for local qualitative interaction, not throughput benchmarking or production serving.
