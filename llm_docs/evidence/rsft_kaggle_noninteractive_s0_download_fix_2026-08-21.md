# R-SFT Kaggle non-interactive S0 download fix — 2026-08-21

A committed Kaggle R-SFT run failed before bundle construction while resolving private S0 dataset `roccoangelella/small-llm-100m-2b-sft-s0-001`. KaggleHub selected its Kaggle-cache resolver and attempted to attach the dataset dynamically. Kaggle rejected that operation because new datasets cannot be attached in a non-interactive session.

The R-SFT S0 downloader now sets `DISABLE_KAGGLE_CACHE=true` only for the KaggleHub download subprocess. KaggleHub therefore skips its notebook attachment/cache resolver and uses the authenticated HTTP resolver. Resolution order remains explicit `--s0-bundle`, then an already attached matching `/kaggle/input` bundle, then the private HTTP download fallback.

Focused launcher/autoprep regression coverage verifies that the private-download subprocess receives `DISABLE_KAGGLE_CACHE=true`, `PYTHONUNBUFFERED=1`, and `UV_LINK_MODE=copy`. The focused Kaggle R-SFT launcher/autoprep suite passes 20 tests after the fix.

A live KaggleHub 1.0.2 resolver smoke test simulated `KAGGLE_KERNEL_RUN_TYPE=Batch`, pointed `KAGGLE_DATA_PROXY_URL` at a deliberately unreachable local endpoint, and set `DISABLE_KAGGLE_CACHE=true`. `kagglehub.dataset_download(..., path="abalone.csv")` successfully downloaded the 216,903-byte public file through the HTTP cache path, confirming that the disabled Kaggle-cache resolver was not consulted.
