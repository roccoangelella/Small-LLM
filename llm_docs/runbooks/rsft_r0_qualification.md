# Production R-SFT R0 qualification on Kaggle

Use this runbook to compare completed S0 `100m-2b-sft-s0-001` with accepted production R-SFT `100m-2b-rsft-r0-12306-001`.

## Required inputs

Attach the same frozen `eval_core_v1` Kaggle dataset used by the S0 qualification. The launcher discovers it by its verified `manifest.json`.

The completed S0 bundle is resolved automatically in this order:

1. `--s0-bundle` when supplied;
2. one matching attached Kaggle input;
3. private Kaggle dataset `roccoangelella/small-llm-100m-2b-sft-s0-001` through `kagglehub`.

The production R-SFT bundle does not need a separate attachment. Unless `--dataset-dir` is supplied, the launcher deterministically rebuilds and verifies it from the committed 12,306-row production reasoning corpus plus the resolved S0 retention source.

Hugging Face checkpoint access requires `HF_TOKEN`. Repository selection resolves through:

```text
SMALL_LLM_HF_REPO_ID
SMALL_LLM_SFT_HF_REPO_ID
SMALL_LLM_RSFT_HF_REPO_ID
```

The shared qualification repository may be used for all three when appropriate.

## Canonical full qualification

From the repository root in Kaggle:

```bash
python kaggle/launch_r_sft.py eval --model 100M --tokens 2B --suite full
```

In a notebook cell:

```bash
!python kaggle/launch_r_sft.py eval --model 100M --tokens 2B --suite full
```

The full suite compares S0→R-SFT and includes:

- full `eval_core_v1` on both checkpoints;
- S0 validation/test loss on both checkpoints;
- R-SFT production-bundle validation/test loss on R-SFT;
- the same 30 deterministic instruction-behavior probes used for S0, with R-SFT final-answer extraction;
- the same full 18-prompt greedy-32 qualitative regression (`T=0`, `top_p=1`, `top_k=0`, seed 17);
- the same full 18-prompt wider regression (`T=1.0`, `top_p=0.9`, `top_k=20`, seed 17, native budgets);
- 35 novel mechanically scored reasoning problems balanced across `INF`, `DED`, `REL`, `CSP`, `IND`, `ABD`, and `MAG`;
- reasoning greedy accuracy (`T=0`) and an eight-sample pass@1 estimate (`T=0.6`, `top_p=0.95`);
- atomic `<think>...</think><answer>...` protocol health and reasoning/answer length telemetry.

The report deliberately has no single master score.

## Fast smoke qualification

```bash
!python kaggle/launch_r_sft.py eval --model 100M --tokens 2B --suite fast
```

## Useful overrides

```bash
!python kaggle/launch_r_sft.py eval \
  --model 100M \
  --tokens 2B \
  --suite full \
  --reasoning-samples 16 \
  --reasoning-max-new-tokens 256 \
  --output /kaggle/working/post-rsft-full-qualification.json
```

Use more reasoning samples only when a tighter repeated-sampling estimate is worth the extra generation cost. Do not change the historical greedy/wider settings when making stage-to-stage comparisons.

## Output interpretation

Treat these as separate axes:

1. base-language retention (`eval_core_v1`);
2. S0 instruction retention;
3. instruction-behavior correctness;
4. novel reasoning final-answer accuracy;
5. R-SFT reasoning-token protocol health;
6. held-out R-SFT loss.

Do not interpret a fluent visible reasoning trace as evidence that the trace faithfully exposes the model's internal causal computation. The canonical suite does not use an LLM judge for that claim.
