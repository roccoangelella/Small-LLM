# R-SFT Kaggle wrong-parent-repository fix — 2026-08-21

A committed expanded-corpus run reached 2xT4 DDP successfully, then both ranks failed in parent checkpoint resolution. The generated trainer command used:

```text
--parent-repo-id roccoangelella/small-llm-20m-qualification
--parent-run-id 100m-2b-sft-s0-001
--checkpoint-repo-id roccoangelella/small-llm-20m-qualification
```

The resulting error was `Hugging Face model repository contains no artifact for run '100m-2b-sft-s0-001'`. This was not a dataset, bundle, DDP, CUDA, or S0-Kaggle-download failure; repository fallback selected a stale cross-profile Kaggle secret.

The fixed 100M/2B R-SFT resolver now ignores `SMALL_LLM_HF_REPO_ID` and `SMALL_LLM_SFT_HF_REPO_ID`. It defaults parent and checkpoint resolution to `roccoangelella/small-llm-100m-qualification`, with dedicated `SMALL_LLM_100M_HF_REPO_ID` / `SMALL_LLM_RSFT_HF_REPO_ID` and explicit CLI overrides available. Eval uses the same resolver.

A stale-environment dry run with both generic variables deliberately set to the 20M repository emitted the 100M repository for both `--parent-repo-id` and `--checkpoint-repo-id`. Focused R-SFT tests passed. An authenticated Hugging Face inventory check independently confirmed that `roccoangelella/small-llm-100m-qualification` contains 51 files under `run/100m-2b-sft-s0-001/` and includes `run/100m-2b-sft-s0-001/latest.json`.
