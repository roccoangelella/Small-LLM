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

Do not relaunch this run under the same run ID unless intentionally resuming the same verified trajectory. A future corpus that incorporates the remaining 4,476 adapted keepers must use a new corpus identity and a new R-SFT run ID.
