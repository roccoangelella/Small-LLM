---
status: current
last_reviewed: 2026-09-08
---

# Current roadmap

## Current position

- The 20M scaling series through 2B is complete.
- 100M/2B pretraining is complete at 2,001,000,448 consumed target tokens.
- 100M/10B deep-decay pretraining is complete at `step-00076294` / 10,000,007,168 consumed target tokens.
- Evaluation v2 is active under ADRs 0140 and 0141.
- The 100M/10B S0 SFT trajectory `100m-10b-sft-s0-2b10pct-data-001` has been restarted after a Kaggle T4 session-time interruption; that interruption is infrastructure evidence, not a model-quality result.
- ADR 0144 defines the current post-completion pretraining diagnostic: one launcher, two constant-LR holds (`1e-5`, `2e-5`), 3,000 updates per branch, preferred source `step-00071750`, strict current-best fallback from the same dedicated best-model repository, and no rolling-latest fallback.
- ADR 0153 authorizes 100M/100B on Modal H100 and Beam RTX 4090.
- ADR 0158 freezes a complete prebuilt public-HF `100b-b64-dataset-001`; paid providers do not build this corpus.

## Immediate priorities

1. Build and fully publish `100b-b64-dataset-001` to its dedicated public HF bucket, then verify its terminal manifest/READY/frontier and immutable shard inventory.
2. Complete or exactly resume the 100M/10B SFT trajectory under its existing checkpoint/data contract.
3. Run evaluation-v2 SFT qualification after completion and compare the SFT model with the 100M/10B parent using the primary Behavior v2 suite, frozen `eval_core_v1`, masked-loss diagnostics, and the defined sampled-robustness view.
4. Run the ADR-0144 `hold-1e-5` and `hold-2e-5` pretraining probes from the same source/data continuation and compare their validation trajectories.
5. Run provider CPU-stage/live-smoke checks against the completed 100B bucket before allocating a full training segment.

## Next decision gate

Do not authorize another long pretraining trajectory solely because an execution path is available. The next scaling decision should use, at minimum:

- the 100M/10B endpoint relative to 100M/2B under evaluation v2;
- the two low-LR probe trajectories from ADR 0144;
- the completed 100M/10B SFT qualification.

ADR 0153 resolved this gate in favor of more data at fixed 100M scale. The 100B trajectory is authorized only on its frozen Modal H100 and Beam RTX 4090 lanes; corpus completion and provider CPU-stage/live-smoke verification remain prelaunch gates.

## Frozen boundaries still in force

- Context remains 2,048 for the current comparison family.
- Production CUDA GDN-2 uses `fla-core==0.5.2`, saved chunk 32 / FLA internal chunk 64.
- New dataset durability uses Hugging Face Storage Buckets, not Google Drive.
- Live exact-resume checkpoints and strict validation-loss best artifacts remain separate according to ADR 0132.
- Canonical sampled qualitative decoding is `temperature=1`, `top_p=1`, `top_k=0`.
- Pretraining EOS termination is not a metric.
- Teacher-forced confidence and masked SFT losses remain diagnostics rather than headline capability scores.
