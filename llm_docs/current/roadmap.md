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
- ADR 0159 supersedes ADR 0153's next-run scale target and selects approximately 200M parameters / 50B target tokens for the next major pretraining trajectory.
- ADR 0158 remains in force: the complete public-HF `100b-b64-dataset-001` corpus is still built to 100B even though the immediate training horizon is 50B; paid providers do not build this corpus.

## Immediate priorities

1. Build and fully publish `100b-b64-dataset-001` to its dedicated public HF bucket, then verify its terminal manifest/READY/frontier and immutable shard inventory.
2. Plan and freeze the approximately-200M / 50B scientific contract before implementation: exact architecture/parameter count, optimizer and LR schedule, block/microbatch/update geometry, checkpoint cadence, provider limits, and deterministic 50B consumption mapping onto the 100B corpus.
3. Qualify the frozen 200M geometry and run contract with local/provider smoke tests before any long GPU trajectory is launched.
4. Complete or exactly resume outstanding 100M/10B qualification/probe work where it remains scientifically useful for interpreting the scaling transition.

## Next decision gate

ADR 0159 fixes the scale target but deliberately does not freeze the 200M model or training recipe. The immediate gate is therefore architectural and optimization design, not another scale-choice debate.

Before wiring production launchers, explicitly decide and understand:

- whether 200M is a pure width/depth scale of the existing `[GDN-2, GDN-2, GDN-2, gated full MHA]` hybrid family or introduces a new architecture change;
- exact `d_model`, depth, FFN width, attention/GDN head geometry, and resulting learned parameter count;
- optimizer routing and peak LR;
- warmup/stable/decay shape over exactly 50B targets;
- global tokens per optimizer update and provider-specific microbatch slicing;
- checkpoint/evaluation cadence and deterministic exact-resume behavior;
- the exact 50B prefix/slice contract drawn from `100b-b64-dataset-001`.

No long 200M/50B launch is authorized until those items are frozen and provider smoke-tested.

## Frozen boundaries still in force unless explicitly superseded

- Context remains 2,048 for the current comparison family.
- Production CUDA GDN-2 uses `fla-core==0.5.2`, saved chunk 32 / FLA internal chunk 64.
- New dataset durability uses Hugging Face Storage Buckets, not Google Drive.
- Live exact-resume checkpoints and strict validation-loss best artifacts remain separate according to ADR 0132.
- Canonical sampled qualitative decoding is `temperature=1`, `top_p=1`, `top_k=0`.
- Pretraining EOS termination is not a metric.
- Teacher-forced confidence and masked SFT losses remain diagnostics rather than headline capability scores.
