---
date: 2026-08-21
status: observed-and-fixed
run_id: 100m-10b-deep-decay-from-step15500
---

# 100M/10B Kaggle deep-decay wrong Hugging Face repository routing

## Observed failure

The canonical Kaggle command:

```bash
python kaggle/launch.py deep-decay --model 100M --tokens 10B
```

completed its private `huggingface_hub==1.5.0` host bootstrap, then failed before dataset staging or GPU training with:

```text
RuntimeError: exact source 100m-10b-data-001/step-00015500 is unavailable in the HF model repository
```

The preceding pip dependency-resolver warnings were not the terminating failure.

## Root cause

The deep-decay subprocess inherited the generic `SMALL_LLM_HF_REPO_ID` Kaggle environment variable and the provider-neutral Beam runtime used that value for model-checkpoint lookup. On 2026-08-21 the R-SFT lane independently proved that the Kaggle environment can retain a stale generic repository value pointing at the 20M qualification repository. The R-SFT lane was hardened against that same failure class, but the deep-decay launcher still trusted the generic variable.

This explains why a checkpoint namespace known to have restored successfully on Kaggle on 2026-08-18 could suddenly appear unavailable without any T4, dataset, or checkpoint-format error.

## Fix

The canonical `kaggle/launch.py deep-decay` path now binds its child process to the 100M checkpoint repository explicitly:

1. use `SMALL_LLM_100M_HF_REPO_ID` when that dedicated override is set;
2. otherwise use `roccoangelella/small-llm-100m-qualification`;
3. overwrite only the child process's `SMALL_LLM_HF_REPO_ID`, so a stale generic Kaggle secret cannot redirect this 100M/10B trajectory;
4. print the selected repository before launching the deep-decay subprocess.

The repository-routing fix changes no model weights, optimizer/scaler/RNG state, data cursor, optimizer-block geometry, LR schedule, checkpoint identity, or T4 execution geometry.

A focused regression contract verifies that the deep-decay launcher keeps this dedicated 100M binding.

## Retry

After updating the repository, the canonical command remains unchanged:

```bash
python kaggle/launch.py deep-decay --model 100M --tokens 10B
```

The launcher should first report:

```text
[launch] deep-decay hf_repo=roccoangelella/small-llm-100m-qualification
```

If restore still fails after that line, the next investigation is the actual 100M repository pointer/inventory rather than Kaggle's generic repository routing.
