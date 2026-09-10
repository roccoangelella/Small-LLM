---
status: current
last_reviewed: 2026-09-10
---

# Current roadmap

MoE branch: qualify and run the paired M0/M1 pilot per [ADR 0168](../decisions/0168-moe-paired-pilot-controller-and-observation.md) and the [pilot runbook](../runbooks/moe-paired-pilot.md), after explicit launch authorization. This is the MoE branch priority; the dense/SFT lifecycle below retains its earlier scope.

## MoE branch: execution efficiency gates and open decisions (2026-09-10)

Accepted on this branch: ADRs 0170–0172 (batched Muon, fused attention + chunked loss, sorted
dispatch), CPU-equivalent, GPU-unmeasured. Before any many-expert 100B run (ADR 0168 on `main`):

1. **GPU profile of the accepted E64/Top-2/h352/d256/L8/V8k geometry** with the largest fitting
   microbatch, telemetry off, four clocks reported; compare launches/syncs per token against the
   2026-09-08 baseline. Requires explicit launch authorization and a cost cap.
2. **Batched expert GEMM** (pad sorted slices, one `bmm` per layer) after the profile shows the
   expert loop dominating; then CUDA graphs on static shapes. FP8 last.
3. **Router contract to settle with the `main` owner**: the 2026-09-10 `main` history records two
   accepted contracts for the same experiment — sigmoid expert scores (commit `433039d`) and
   `sqrt(softplus(z))` scores (commit `e0621b6`, later) — both with Top-K over `s + b`, Quantile
   Balancing bias for selection only, unbiased normalized mixture weights, FP32 router on AdamW, no
   z-loss. Both files were removed from the `main` tree by ADR 0172 (`main` numbering, "experiment
   memory on dedicated branches") and are on no branch tip. With elementwise scores the router rows
   of unselected experts receive no gradient from the LM loss; balance rests entirely on the bias.
4. **Merge of `main` into this branch**: 81 files changed on `main` since the 2026-09-06 merge base,
   9 touched on both sides (`dataset/incremental_cache.py`, `dataset/incremental_frontier.py`,
   `trainer/cli_args.py`, `pyproject.toml`, `kaggle/src/launch.py`, `llm_docs/current/*`,
   `llm_docs/decisions/README.md`, `tests/test_incremental_frontier.py`). `main` also carries the
   offline test-health fix (51 files) and the shared rolling-dataset producer refactor.
5. **Proposal, not adopted**: save checkpoints at log-spaced consumed-token counts (1, 2, 5, 10,
   20, 50, 100 B) during the long run so later studies (knowledge-vs-tokens curves, expert
   specialization, retrofit experiments) reuse one trajectory; cost is disk only.

## Current position

- The 20M scaling series through 2B is complete.
- 100M/2B pretraining is complete at 2,001,000,448 consumed target tokens.
- 100M/10B deep-decay pretraining is complete at `step-00076294` / 10,000,007,168 consumed target tokens.
- Evaluation v2 is active under ADRs 0140 and 0141.
- The 100M/10B S0 SFT trajectory `100m-10b-sft-s0-2b10pct-data-001` has been restarted after a Kaggle T4 session-time interruption; that interruption is infrastructure evidence, not a model-quality result.
- ADR 0144 defines the current post-completion pretraining diagnostic: one launcher, two constant-LR holds (`1e-5`, `2e-5`), 3,000 updates per branch, preferred source `step-00071750`, strict current-best fallback from the same dedicated best-model repository, and no rolling-latest fallback.

## Immediate priorities

1. Complete or exactly resume the 100M/10B SFT trajectory under its existing checkpoint/data contract.
2. Run evaluation-v2 SFT qualification after completion and compare the SFT model with the 100M/10B parent using the primary Behavior v2 suite, frozen `eval_core_v1`, masked-loss diagnostics, and the defined sampled-robustness view.
3. Run the ADR-0144 `hold-1e-5` and `hold-2e-5` pretraining probes from the same source/data continuation and compare their validation trajectories. The purpose is to test whether the apparent 10B tail plateau is explained by terminal LR decay rather than by model/data saturation.
4. Re-evaluate completed pretrained checkpoints with evaluation v2 where needed so subsequent scale decisions use one protocol.
5. Keep completed 100M/10B provider/run procedures as reproduction and recovery references, not as active launch authorization.

## Next decision gate

Do not authorize another long pretraining trajectory solely because an execution path is available. The next scaling decision should use, at minimum:

- the 100M/10B endpoint relative to 100M/2B under evaluation v2;
- the two low-LR probe trajectories from ADR 0144;
- the completed 100M/10B SFT qualification.

After those results are available, choose explicitly between more data at fixed 100M scale, a geometry/architecture change, or shifting additional effort toward post-training. Record that choice in a new ADR. No 50B- or 100B-token pretraining trajectory is authorized by this roadmap.

## Frozen boundaries still in force

- Context remains 2,048 for the current comparison family.
- Production CUDA GDN-2 uses `fla-core==0.5.2`, saved chunk 32 / FLA internal chunk 64.
- New dataset durability uses Hugging Face Storage Buckets, not Google Drive.
- Live exact-resume checkpoints and strict validation-loss best artifacts remain separate according to ADR 0132.
- Canonical sampled qualitative decoding is `temperature=1`, `top_p=1`, `top_k=0`.
- Pretraining EOS termination is not a metric.
- Teacher-forced confidence and masked SFT losses remain diagnostics rather than headline capability scores.
