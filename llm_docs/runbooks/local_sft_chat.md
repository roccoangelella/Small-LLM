# Local chat CLI

Use the root-level `chat.py` to download an explicitly registered completed Small-LLM artifact from Hugging Face and run an interactive local chat loop. The CLI supports completed SFT trajectories, selected stable pretrained artifacts, and the accepted completed R-SFT trajectory.

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

If post-training checkpoints live in a separate repository, set it explicitly:

```bash
export SMALL_LLM_SFT_HF_REPO_ID=owner/sft-repository
```

R-SFT can use its own repository:

```bash
export SMALL_LLM_RSFT_HF_REPO_ID=owner/rsft-repository
```

R-SFT profiles prefer `SMALL_LLM_RSFT_HF_REPO_ID`, then `SMALL_LLM_SFT_HF_REPO_ID`, then `SMALL_LLM_HF_REPO_ID`. SFT profiles prefer `SMALL_LLM_SFT_HF_REPO_ID` and fall back to `SMALL_LLM_HF_REPO_ID`. Stable pretrained artifacts always use `SMALL_LLM_HF_REPO_ID`.

## Mandatory stage selection

Every invocation must select exactly one model stage:

```text
--pre-trained
--sft
--r-sft
```

The stage flags are mutually exclusive and there is no default. This is deliberate: pretrained/S0 artifacts use the ordinary GPT-2 vocabulary, while R-SFT artifacts use the extended reasoning-token vocabulary.

## Run

500M-parent 20M SFT:

```bash
python chat.py --model_params 20M --num_tokens 500M --sft
```

2B-parent 20M SFT:

```bash
python chat.py --model_params 20M --num_tokens 2B --sft
```

Completed 100M / 2B SFT default:

```bash
python chat.py --model_params 100M --num_tokens 2B --sft
```

A different completed SFT run in the same profile can be selected explicitly without changing the registered default. For example, the 10% S0 experiment is:

```bash
python chat.py --model_params 100M --num_tokens 2B --sft --run-id 100m-2b-sft-s0-10pct-001
```

The explicit SFT path uses the same normal GPT-2 tokenizer and completed-checkpoint verification as the registered SFT default. It fails closed if the selected Hugging Face checkpoint has not consumed its full frozen schedule.

Completed 100M / 2B pretrained base model:

```bash
python chat.py --model_params 100M --num_tokens 2B --pre-trained
```

The pretrained 100M / 2B entry resolves the stable Hugging Face model artifact for run `100m-2b-data-001`. It is a pretrained base model, not an SFT/instruction-tuned model.

Accepted 100M / 2B atomic R-SFT R0:

```bash
python chat.py --model_params 100M --num_tokens 2B --r-sft
```

This resolves the completed run `100m-2b-rsft-r0-12306-001`, whose verified Hugging Face `latest.json` points to `step-00000361`. The earlier atomic pilot, 10-epoch repeat probe, and textual pilot remain historical experiment identities in Git but their Hugging Face run namespaces have been deleted.

Hyphenated spellings remain aliases for the profile arguments, and `--pretrained` aliases `--pre-trained`:

```bash
python chat.py --model-params 100M --num-tokens 2B --pretrained
```

`--num_tokens` identifies the parent pretraining-token profile. Generation length is controlled by `MAX_NEW_TOKENS` near the top of `chat.py`.

## Tokenizer selection

`--pre-trained` and `--sft` use the unchanged GPT-2 tokenizer and require:

```text
semantic_vocab_size = 50,257
```

`--r-sft` uses `post_training/R-SFT/tokenizer.py` and requires:

```text
semantic_vocab_size = 50,260
reasoning-start ID = 50,257  <think>
reasoning-end ID   = 50,258  </think>
answer-start ID    = 50,259  <answer>
```

The three marker strings are now part of the accepted atomic R-SFT protocol. The verified R-SFT checkpoint must carry them in:

```text
checkpoint.json
  pipeline_state
    reasoning_tokenizer
```

and must identify its serialization as:

```text
pipeline_state.rsft_format.version = 1
pipeline_state.rsft_format.stage = r_sft_r0
pipeline_state.rsft_format.delimiter_format = atomic
```

The reasoning-tokenizer metadata records the GPT-2 base encoding, semantic vocabulary size, each marker string, and its fixed ID. `chat.py` reconstructs the extended encoder from this metadata. Missing/malformed metadata, wrong IDs, noncanonical marker spellings, a textual R-SFT format, or a model/tokenizer vocabulary mismatch aborts loading.

The extended encoder treats each configured marker as one atomic token, but delegates all ordinary text to the existing GPT-2 encoding. Its decoder and single-token byte decoder also understand the promoted IDs, so streamed output and retained assistant history round-trip through the same token contract.

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
4. `consumed_tokens == warmup_tokens + stable_tokens + decay_tokens`;
5. `semantic_vocab_size=50_257`.

For R-SFT profiles, the same verified post-training checkpoint path is used, but the loader additionally requires the atomic `r_sft_r0` format, the exact `<think>`, `</think>`, `<answer>` reasoning-token contract, and `semantic_vocab_size=50_260`.

For the 100M / 2B pretrained profile, the CLI uses the canonical stable `models/100m-2b-data-001/...` artifact path, verifies its native `local_manifest.json`, requires the same embedded WSD schedule-completion equality, and requires the ordinary 50,257-token vocabulary before loading the weights.

If an explicitly registered artifact cannot be resolved, verified, shown complete, or matched to the selected tokenizer stage, the command exits with an error rather than silently substituting another model/tokenizer.

## Interactive commands

- `/clear` drops conversation history.
- `/quit` or `/exit` exits.
- `Ctrl-C` / EOF also exits cleanly.

The conversation is serialized with the existing `small-llm-s0-v1` chat generation template. The encoder supplied to that template is stage-specific: normal GPT-2 for pretrained/SFT, or the reasoning-aware wrapper for R-SFT. When history grows too long, the oldest complete user/assistant pairs are discarded until the retained prompt leaves room for `MAX_NEW_TOKENS` inside the model context.

For the 100M / 2B base model the chat template is only an interactive prompt format; it does not imply that the model was trained on SFT conversations.

## Streaming behavior

After the user submits a message, `chat.py` prints and flushes `assistant> ` immediately. Once the model samples its first non-EOS token, decoded text is printed token-by-token instead of buffering the entire response.

GPT-2 token boundaries do not always align with UTF-8 character boundaries, so the CLI incrementally decodes token bytes before printing. The R-SFT encoder exposes the same byte-decoding interface for its special tokens. This prevents a multi-byte character split across tokens from being rendered as replacement characters.

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
