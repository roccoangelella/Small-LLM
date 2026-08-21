# R-SFT R0 atomic runbook

_Last updated: 2026-08-19 Europe/Rome_

The current accepted 100M / 2B R0 R-SFT artifact is the completed 12,306-row Superior/Gemini checkpoint:

```text
100m-2b-rsft-r0-12306-001
```

The verified Hugging Face latest pointer is the completed final optimizer boundary:

```text
step-00000361
```

It uses the frozen atomic special-token protocol:

```text
50257  <think>
50258  </think>
50259  <answer>
```

The historical 630-example delimiter pilot and 10-epoch repeat probe are no longer accepted model artifacts. Their Hugging Face run namespaces were deleted on 2026-08-19 after the 12,306-row run completed. Their experiment definitions remain reproducible from Git for audit purposes.

## 1. Current accepted R0 state

The frozen reasoning corpus is:

```text
artifacts/rsft-superior-instruction-r0-checkpoint-12306/reasoning.jsonl
```

It contains 12,306 unique normalized prompts:

- 7,683 unchanged, context-fit Superior instruction rows;
- 3,993 unique accepted Variant-D Superior rewrites;
- 630 frozen Gemini logic anchors.

The adjacent manifest records 28 accepted rewrites omitted because compression created normalized-prompt collisions, and 4,476 manually-kept over-context candidates that remain pending future compression. Every emitted row passes the exact atomic R-SFT serialization at the 2,048-token model context.

The native training bundle used 32,768 loss-bearing target tokens per optimizer block and one exact pass. It contained 361 train blocks and 11,609,452 train target tokens: 10,448,098 reasoning targets plus 1,161,354 S0-retention targets, preserving the approximately 90/10 reasoning/retention contract.

## 2. Chat with the accepted artifact

Configure Hugging Face access. R-SFT first checks `SMALL_LLM_RSFT_HF_REPO_ID`, then falls back to the SFT/base checkpoint repository variables. In the current shared setup, `SMALL_LLM_HF_REPO_ID=roccoangelella/small-llm-100m-qualification` is sufficient.

Run the registered current R-SFT model:

```bash
.venv/bin/python chat.py --model_params 100M --num_tokens 2B --r-sft
```

The explicit equivalent is:

```bash
.venv/bin/python chat.py \
  --model_params 100M \
  --num_tokens 2B \
  --r-sft \
  --run-id 100m-2b-rsft-r0-12306-001
```

`chat.py` fails closed unless the downloaded checkpoint is complete and carries:

```text
semantic_vocab_size = 50260
pipeline_state.rsft_format.version = 1
pipeline_state.rsft_format.stage = r_sft_r0
pipeline_state.rsft_format.delimiter_format = atomic
```

plus the exact `<think>`, `</think>`, `<answer>` token metadata at IDs 50257-50259.

## 3. Hugging Face R-SFT state

The checkpoint repository is:

```text
roccoangelella/small-llm-100m-qualification
```

The only retained 100M / 2B R-SFT run namespace is:

```text
run/100m-2b-rsft-r0-12306-001/
```

Its `latest.json` points to `step-00000361`. The completed S0 parent `100m-2b-sft-s0-001` is preserved.

The following superseded R-SFT namespaces were deliberately deleted from Hugging Face on 2026-08-19:

```text
100m-2b-rsft-r0-atomic-pilot-001
100m-2b-rsft-r0-atomic-repeat-e10-001
100m-2b-rsft-r0-textual-pilot-001
```

Do not treat those names as remotely loadable checkpoints. They remain historical experiment identities only.

## 4. Training/reproduction

The canonical production launch for the frozen 12,306-row checkpoint is:

```bash
python kaggle/launch_r_sft.py train --model 100M --tokens 2B
```

The launcher pins the committed corpus and implementation, resolves the completed 100M/2B S0 parent bundle, builds/verifies the atomic production bundle, and uses the distinct run ID `100m-2b-rsft-r0-12306-001`.

The production builder can be run directly with:

```bash
python post_training/R-SFT/build_atomic.py \
  --reasoning-jsonl artifacts/rsft-superior-instruction-r0-checkpoint-12306/reasoning.jsonl \
  --s0-bundle /path/to/100m-2b-sft-s0-bundle \
  --output-dir /path/to/rsft-r0-superior-instruction-checkpoint-12306
```

Do not relaunch this run under the same run ID unless intentionally resuming the same verified trajectory. Any expanded corpus must use a new corpus identity and a new R-SFT run ID.

## 5. Expanded-corpus adaptation lane

ADR 0106 resumes the over-context lane without changing the historical curation used by the completed model. Expansion work uses:

```text
artifacts/rsft-superior-instruction-r0-adaptation/manual-curation.expanded-v2.jsonl
```

Curation v2 contains 8,473 keepers, 829 code exclusions, 212 math exclusions, and 110 safety exclusions. The keeper-only resume reuses 4,009 valid historical accepted rewrites and sends only the remaining 4,464 keepers to Gemini.

Before any teacher request, require GemRouter to have `GEMROUTER_BACKEND_ORDER=gemini-api` and `GEMROUTER_NVIDIA_ENABLED=false`; `/health` must show only `gemini-api` and `fallbackEnabled=false`. Never enable NVIDIA for this dataset.

Prepare/status commands:

```bash
.venv/bin/python post_training/R-SFT/dataset/resume_superior_keep_adaptation.py prepare \
  --work-dir artifacts/rsft-superior-instruction-r0-adaptation \
  --manual-curation-jsonl artifacts/rsft-superior-instruction-r0-adaptation/manual-curation.expanded-v2.jsonl \
  --baseline-manifest artifacts/rsft-superior-instruction-r0/reasoning.jsonl.manifest.json

.venv/bin/python post_training/R-SFT/dataset/resume_superior_keep_adaptation.py status \
  --work-dir artifacts/rsft-superior-instruction-r0-adaptation
```

Provider batches and attempts under `keep-resume/` are generated local state and remain ignored by Git. The keeper lane completed on 2026-08-21 with `resume_pending_records=0`. Finalize deterministically with:

```bash
.venv/bin/python post_training/R-SFT/dataset/resume_superior_keep_adaptation.py finalize \
  --work-dir artifacts/rsft-superior-instruction-r0-adaptation \
  --baseline-jsonl artifacts/rsft-superior-instruction-r0/reasoning.jsonl \
  --baseline-manifest artifacts/rsft-superior-instruction-r0/reasoning.jsonl.manifest.json \
  --manual-curation-jsonl artifacts/rsft-superior-instruction-r0-adaptation/manual-curation.expanded-v2.jsonl \
  --output-jsonl artifacts/rsft-superior-instruction-r0-expanded/reasoning.jsonl
```

The frozen result is 16,716 rows at SHA-256 `d13052b6fc33108ec65511b790a75f6473144855059b16b55167b046f787c405`, after 70 normalized-prompt collision exclusions. Build the verified native bundle with:

```bash
.venv/bin/python post_training/R-SFT/build_atomic.py \
  --reasoning-jsonl artifacts/rsft-superior-instruction-r0-expanded/reasoning.jsonl \
  --s0-bundle /home/ubuntu/Projects/small-llm-work/small-llm-100m-2b-sft-bundle \
  --output-dir /home/ubuntu/Projects/small-llm-work/rsft-r0-superior-instruction-expanded-16716
```

The completed bundle has 417 train blocks and 13,420,823 train targets (12,077,733 reasoning + 1,343,090 S0 retention). Any training from this bundle requires a new run identity; do not resume the completed 12,306-row trajectory.
