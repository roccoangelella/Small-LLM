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
- ADR 0160 supersedes ADR 0159 and selects approximately 200M parameters / 100B target tokens for the next major pretraining trajectory.
- ADR 0161 retains the successful 100M/10B deep-decay/WSqD-style LR structure but requires a modestly broader numerical LR range at every major anchor for 200M/100B; exact values and phase boundaries remain open.
- ADR 0158 remains in force: the complete public-HF `100b-b64-dataset-001` corpus is built to 100B and the next major run is intended to consume the full corpus; paid providers do not build this corpus.

## Immediate priorities

1. Build and fully publish `100b-b64-dataset-001` to its dedicated public HF bucket, then verify its terminal manifest/READY/frontier and immutable shard inventory.
2. Plan and freeze the approximately-200M / 100B scientific contract before implementation: exact architecture/parameter count, optimizer and LR schedule, block/microbatch/update geometry, checkpoint cadence, and provider limits.
3. Select exact widened LR anchors and their token boundaries under ADR 0161. The starting evidence is the successful 100M/10B sequence `3e-4 -> 1e-4 -> 1e-5 -> 5e-6`; for 200M/100B, peak should be only modestly higher and each later anchor only modestly lower, while preserving meaningful decay over the much longer horizon.
4. Qualify the frozen 200M geometry and run contract with local/provider smoke tests before any long GPU trajectory is launched.
5. Complete or exactly resume outstanding 100M/10B qualification/probe work where it remains scientifically useful for interpreting the scaling transition.

## Next decision gate

ADR 0160 fixes the scale target and ADR 0161 fixes the qualitative LR-range direction, but neither freezes the 200M model or exact training recipe. The immediate gate is architectural and numerical optimization design.

Before wiring production launchers, explicitly decide and understand:

- whether 200M is a pure width/depth scale of the existing `[GDN-2, GDN-2, GDN-2, gated full MHA]` hybrid family or introduces a new architecture change;
- exact `d_model`, depth, FFN width, attention/GDN head geometry, and resulting learned parameter count;
- optimizer routing and the exact widened LR anchors;
- warmup and the shape/rate of LR decay over the full 100B horizon, incorporating the 100M aggressive-decay evidence;
- global tokens per optimizer update and provider-specific microbatch slicing;
- checkpoint/evaluation cadence and deterministic exact-resume behavior;
- deterministic full-corpus consumption of `100b-b64-dataset-001`.

No long 200M/100B launch is authorized until those items are frozen and provider smoke-tested.

## Frozen boundaries still in force unless explicitly superseded

- Context remains 2,048 for the current comparison family.
- Production CUDA GDN-2 uses `fla-core==0.5.2`, saved chunk 32 / FLA internal chunk 64.
- New dataset durability uses Hugging Face Storage Buckets, not Google Drive.
- Live exact-resume checkpoints and strict validation-loss best artifacts remain separate according to ADR 0132.
- Canonical sampled qualitative decoding is `temperature=1`, `top_p=1`, `top_k=0`.
- Pretraining EOS termination is not a metric.
- Teacher-forced confidence and masked SFT losses remain diagnostics rather than headline capability scores.
