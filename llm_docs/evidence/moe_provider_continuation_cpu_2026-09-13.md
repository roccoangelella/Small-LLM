# MoE provider continuation: local verification, 2026-09-13

Baseline: clean `29cabac0488f379ff6b5b13fb904df17f4fc52cb`, branch `edo/production-run`.
Scope and accepted behavior: [ADR 0183](../decisions/0183-continue-moe-training-across-provider-accounts.md).
Procedure: [provider continuation](../runbooks/moe-provider-continuation.md).

## Verified locally

`tests/test_moe_provider_continuation.py`: 16 tests passed. Real tiny version-3 MoE,
real trainer CLI and checkpoint serializer/publisher; byte-accurate fake HF Bucket
SDK and fake W&B. No cloud action or GPU allocation.

- Train, publish, restore into an empty account, change microbatch, continue.
- Exact saved model/router, optimizer, scaler and RNG tensors; exact data cursor,
  update count and token/LR clock. Subsequent microbatch-partitioned CPU update
  agrees within 2e-5; this is not a cross-GPU numerical equivalence claim.
- CLI updates 1–4 across same-volume and empty-volume resumes; stable W&B identity,
  resumptions use `must`, cadence and final checkpoints publish.
- Missing continuation checkpoint, scientific configuration drift, corpus/source/
  telemetry identity drift, corrupt remote bytes and interrupted uploads fail closed.
- Retention protects latest and best (including legacy `/best` trees).
- Recover missing publication sidecars after a local volume commit; repair an
  interrupted best-pointer update, including after the final update on CPU.
- Restore honors the existing Beam distributed-volume no-fsync policy.

The affected regression selection passed **168 tests and 26 subtests** (14 existing
TorchScript deprecation warnings) in 260.40 seconds. It covers production launch/
wiring/streaming/schedule, compile lane, observation, accepted geometry, checkpoint
transport/sequence/retention, trainer configuration and W&B. A subsequent completed
streaming CPU-gate regression also passed. The final receipt schedule fallback
reuses the existing geometry-only contract resolver and has its own regression.
Final rerun of continuation, schedule and streaming: **39 passed in 5.42 seconds**.
Across these selections, 170 distinct tests passed (plus the 26 subtests above).

## Baseline limitations

Whole-tree pytest collection fails with the same nine errors on the untouched
baseline and the edited checkout: missing historical Kaggle modules for
build_and_push_100m, gdn2_fla_fp32_qualification, 20m_100m console/launcher,
dual_t4 autotune/production/qualification, sft_publication; plus the historical
modal_split_checkpoint_transport `runtime` import collision. Baseline collected
833 tests before interruption. These unrelated files were not rebuilt.

Two existing fixture failures were reproduced on the detached clean baseline:
test_remote_checkpoint omitted the Python RNG state that the coordinator already
saves, and one test_trainer_best_checkpoint namespace omitted checkpoint_dir.
Only those fixtures were corrected; checkpoint behavior was not weakened.

## Residual verification

HF upload latency, live W&B continuity, provider secret permissions and the actual
Beam→Modal GPU/account handoff remain unmeasured for this integration. Exercise
that handoff in the next authorized useful segment; no new sweep is required.
A hard kill can lose work and unflushed telemetry after the last durable snapshot.
No current claim of GPU launch readiness overrides the pending corpus discussion.
