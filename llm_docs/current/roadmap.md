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
- ADR 0161 retains the successful 100M/10B deep-decay/WSqD-style LR structure but requires a modestly broader numerical LR range at every major anchor for 200M/100B; exact numerical values remain open.
- ADR 0162 freezes the 200M/100B LR phase timing as a 10x proportional scaling of the successful 100M/10B trajectory: warmup endpoint step 38,147 (~5.000B targets), deep-decay start step 155,000 (20.31616B), aggressive-settle endpoint step 177,890 (23.31639808B), terminal-cooldown start step 732,420 (95.99975424B), and final step 762,940 (100.00007168B).
- ADR 0158 remains in force: the complete public-HF `100b-b64-dataset-001` corpus is built to 100B and the next major run is intended to consume the full corpus; paid providers do not build this corpus.

## Immediate priorities

1. Build and fully publish `100b-b64-dataset-001` to its dedicated public HF bucket, then verify its terminal manifest/READY/frontier and immutable shard inventory.
2. Plan and freeze the approximately-200M / 100B scientific contract before implementation: exact architecture/parameter count, optimizer and exact LR anchors, block/microbatch/update geometry, checkpoint cadence, and provider limits.
3. Freeze the exact widened LR values under ADR 0161 and solve the decay function/exponent across ADR 0162's fixed proportional phase boundaries. The starting evidence is the successful 100M/10B sequence `3e-4 -> 1e-4 -> 1e-5 -> 5e-6`; for 200M/100B, peak should be only modestly higher and each later anchor only modestly lower.
4. Qualify the frozen 200M geometry and run contract with local/provider smoke tests before any long GPU trajectory is launched.
5. Complete or exactly resume outstanding 100M/10B qualification/probe work where it remains scientifically useful for interpreting the scaling transition.

## Next decision gate

ADR 0160 fixes the scale target, ADR 0161 fixes the qualitative LR-range direction, and ADR 0162 fixes the proportional LR phase timing. The immediate gate is now exact architecture plus exact numerical optimization design.

Before wiring production launchers, explicitly decide and understand:

- whether 200M is a pure width/depth scale of the existing `[GDN-2, GDN-2, GDN-2, gated full MHA]` hybrid family or introduces a new architecture change;
- exact `d_model`, depth, FFN width, attention/GDN head geometry, and resulting learned parameter count;
- optimizer routing and the exact widened LR anchors;
- the continuous decay function/exponent connecting those LR anchors over ADR 0162's fixed phase boundaries;
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
