# Local chat CLI

Use the root-level `chat.py` to download an explicitly registered completed Small-LLM artifact from Hugging Face and run an interactive local chat loop. The CLI supports completed SFT trajectories and selected stable pretrained artifacts.

## Prerequisites

Install both model and post-training extras:

```bash
pip install -e '.[model,post-training]'
```

Configure Hugging Face access:

```bash
export HF_TOKEN=...
export SMALL_LLM_HF_REPO_ID=owner/repository
```

If SFT checkpoints live in a separate repository, set it explicitly:

```bash
export SMALL_LLM_SFT_HF_REPO_ID=owner/sft-repository
```

SFT profiles prefer `SMALL_LLM_SFT_HF_REPO_ID` and fall back to `SMALL_LLM_HF_REPO_ID`. Stable pretrained artifacts always use `SMALL_LLM_HF_REPO_ID`.

## Run

500M-parent 20M SFT:

```bash
python chat.py --model_params 20M --num_tokens 500M
```

2B-parent 20M SFT:

```bash
python chat.py --model_params 20M --num_tokens 2B
```

Completed 100M / 2B pretrained base model:

```bash
python chat.py --model_params 100M --num_tokens 2B
```

The 100M / 2B entry resolves the stable Hugging Face model artifact for run `100m-2b-data-001`. It is a pretrained base model, not an SFT/instruction-tuned model.

Hyphenated spellings are aliases:

```bash
python chat.py --model-params 100M --num-tokens 2B
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

## Artifact and completion gates

For SFT profiles, the CLI downloads the verified Hugging Face `latest` checkpoint and requires:

1. valid checkpoint/local manifests;
2. SFT pipeline identity with `stage=sft_s0`;
3. a WSD trainer schedule;
4. `consumed_tokens == warmup_tokens + stable_tokens + decay_tokens`.

For the 100M / 2B pretrained profile, the CLI uses the canonical stable `models/100m-2b-data-001/...` artifact path, verifies its native `local_manifest.json`, and requires the same embedded WSD schedule-completion equality before loading the weights. It does not invent or search for a `100m-2b-sft-s0-001` run.

If an explicitly registered artifact cannot be resolved, verified, or shown complete, the command exits with an error rather than silently substituting another model.

## Interactive commands

- `/clear` drops conversation history.
- `/quit` or `/exit` exits.
- `Ctrl-C` / EOF also exits cleanly.

The conversation is serialized with the existing `small-llm-s0-v1` chat generation template. When history grows too long, the oldest complete user/assistant pairs are discarded until the retained prompt leaves room for `MAX_NEW_TOKENS` inside the model context. For the 100M / 2B base model this is only an interactive prompt format; it does not imply that the model was trained on SFT conversations.

## Streaming behavior

After the user submits a message, `chat.py` prints and flushes `assistant> ` immediately. Once the model samples its first non-EOS token, decoded text is printed token-by-token instead of buffering the entire response.

GPT-2 token boundaries do not always align with UTF-8 character boundaries, so the CLI incrementally decodes token bytes before printing. This prevents a multi-byte character split across tokens from being rendered as replacement characters.

Streaming changes presentation only. It does not reduce time-to-first-token or the model's per-token compute cost.

## Device, latency, and Triton behavior

Device selection is automatic:

- CUDA available: FP16 autocast plus the qualified FLA GDN-2 backend.
- No CUDA: CPU FP32 plus the adaptive PyTorch GDN-2 fallback.

Some delay after submitting the first message is expected. The first generated token requires a complete forward pass over the serialized prompt. On a cold CUDA process, the first FLA/Triton execution can additionally pay kernel import/JIT or cache-loading overhead. Later turns can also have noticeable prefill latency because the retained conversation history is longer.

On CUDA, FLA calls Triton JIT kernels. A machine/environment with no matching Triton cache entry can pay compilation latency on first use. Triton maintains a persistent cache under its configured cache directory (by default beneath `~/.triton/`), so identical kernels normally do not need to be recompiled on every token or every process launch while that cache remains valid. Clearing the cache or changing relevant compiler/runtime/device/kernel specialization inputs can cause compilation again.

CPU execution does not use the FLA CUDA path and therefore does not compile Triton kernels.

## Current limitation

`SmallLLM` does not yet expose a unified cached decode contract across GDN-2 and MHA layers. The sampler therefore recomputes the retained prefix for each generated token. Streaming makes this computation visible as tokens finish, but does not make the computation itself faster. This CLI is intended for local qualitative interaction, not throughput benchmarking or production serving.
