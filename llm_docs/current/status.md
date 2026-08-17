---
status: current
last_reviewed: 2026-08-16
---

# Current project status

This file is the high-freshness summary. Measurements live in `../evidence/`, durable choices in `../decisions/`, detailed contracts in `../reference/`, and commands in `../runbooks/`.

## Completed pretraining endpoints

| endpoint | checkpoint | consumed training targets | status |
|---|---|---:|---|
| 20M / 100M | `step-00003053` | 100,018,176 | completed historical scaling point |
| 20M / 500M | `step-00015264` | 500,156,416 | completed; stable HF model artifact |
| 20M / 2B | `step-00061066` | 2,001,000,448 | completed 20M data-scaling endpoint |
| 100M / 2B | `step-00015267` | 2,001,000,448 | completed final Modal/H100 endpoint; stable HF model artifact |

The approximately-20M geometry is 20,637,592 learned parameters (`d_model=256`, `d_ff=704`, 8 layers). The approximately-100M hybrid is 101,252,280 learned parameters (`d_model=512`, `d_ff=1408`, 20 layers). Both use context 2,048.

The final 100M/2B artifact records microbatch 16. Its canonical repository and
stable path are:

```text
roccoangelella/small-llm-100m-qualification
  models/100m-2b-data-001/step-00015267
```

## Frozen three-way intrinsic comparison

The 20M/500M, 20M/2B, and 100M/2B full bundles all use `eval_core_v1` manifest SHA-256:

```text
aa7b6157e5f420dd53a99552685eaed01962ee45c23cbe438e1321a886422792
```

| metric | 20M / 500M | 20M / 2B | 100M / 2B |
|---|---:|---:|---:|
| loss | 4.007289 | 3.894576 | **3.338815** |
| perplexity | 54.997550 | 49.135214 | **28.185701** |
| BPB | 1.250808 | 1.215627 | **1.042155** |
| top-1 | 0.343950 | 0.355129 | **0.398875** |
| top-5 | 0.547491 | 0.561084 | **0.618154** |
| top-10 | 0.620226 | 0.634112 | **0.692041** |
| cluster macro loss | 4.031874 | 3.899664 | **3.349121** |
| mixture-weighted cluster loss | **3.529477** | 3.589138 | **3.042600** |

At fixed 20M capacity, 500M→2B improves ordinary full-eval loss but only 14/19 clusters and mildly regresses the final context buckets. At fixed 2B training tokens, 20M→100M improves all 19/19 clusters and all 8/8 position buckets. The capacity increase yields about 4.9x the absolute loss gain of the 4x data increase at 20M. Current interpretation: **the 20M model is capacity-constrained by the 2B endpoint**.

Canonical evidence: [`../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md`](../evidence/scaling/20m_500m_20m_2b_100m_2b_full_eval_2026-08-13.md).

## Behavioral qualification and 10B authorization

The exact ADR-0025 greedy-32 comparison is complete. Strict QA remains mixed at
0/12 for 20M/500M, 2/12 for 20M/2B, and 2/12 for 100M/2B, but the 100M endpoint
stops before the cap on 10/12 QA prompts versus 3/12 at 20M/2B, is markedly less
repetitive, exposes more facts under matched sampled decoding, and has the
strong uniform intrinsic gains above. See
[`../evidence/scaling/100m_2b_behavioral_qualification_2026-08-13.md`](../evidence/scaling/100m_2b_behavioral_qualification_2026-08-13.md).

ADR 0071 records the user's explicit decision that this evidence is sufficient
to launch the fresh 100M/10B trajectory. The full run is now active on one Beam
RTX4090 from source commit `1f9dff920ecc45ce2fdb43fd875514a18391273d`. Its
current segment resumed exactly from the verified HF step-3,000 checkpoint
after repeated Beam RTX5090 worker/startup failures; the earlier step-250
infrastructure migration from launch source `42b0376` remains part of the
checkpoint ancestry.
Training is uncapped and will run through the full 76,294-update plan without a
5B pause. The approximately-5B checkpoint will be evaluated concurrently on
Kaggle and will not act as a continuation gate.

## GDN-2 production execution

Production CUDA GDN-2 execution is mixed FLA on `fla-core==0.5.2` with FP32 master parameters plus CUDA FP16 autocast. Saved/configured `gdn_chunk_size` is 32; FLA's internal runtime chunk is 64. The adaptive PyTorch recurrence remains the correctness/reference fallback. See [`../reference/gdn2_fla_backend.md`](../reference/gdn2_fla_backend.md).

Kaggle production training uses exact-batch two-T4 DDP under ADR 0056. Modal remains the one-H100 lane. Beam is now an alternate single-GPU lane under ADR 0061/0062, restricted to serverless `RTX5090`, `RTX4090`, or `A10G`, with RTX5090 as the default. On the live RTX5090, microbatch 8 exceeded VRAM. Explicit microbatch 4 then passed four finite updates at median 42,018 target tokens/s with 20,904,411,136 peak reserved bytes, 62.08% of the 33,670,758,400-byte device. The frozen optimizer block remains 64 sequences.

## Dataset and checkpoint durability

For **new** dataset production, Hugging Face Storage Buckets are the only remote dataset durability backend under ADR 0054. Google Drive is historical only; legacy fields such as `drive_manifest.json` remain readable provider-neutral compatibility identifiers for already-built artifacts.

Model durability is unified on `SMALL_LLM_HF_REPO_ID` under ADR 0055:

```text
run/<run_id>/...       live two-phase exact-resume checkpoints
models/<run_id>/...    stable completed model artifacts
```

Stable `models/...` artifacts are verified with their native `local_manifest.json`. `checkpoint_manifest.json` is publication metadata for the live two-phase `run/...` protocol and is **not required** for stable model artifacts.

## 100M / 10B execution

ADR 0058 defines the incremental producer/consumer path: approximately-1-GiB immutable HF dataset shards, monotonic READY frontier, frozen 16-block validation prefix, cheap CPU production/staging before H100 allocation, current+successor lead window, and exact ordered H100 consumption. The frozen whole-block training horizon is 76,294 updates / 10,000,007,168 target tokens with standard WSD 3,815 warmup / 57,220 stable / 15,259 decay updates.

Dataset production is complete. HF and Beam both contain 21 train and 21
validation shards; HF reports `target_reached=true`, `producer_complete=true`,
and final manifest SHA-256
`d23e7e4641e30c25b56189093bf1270cd11e85efc8b26bc4660af1873edb96f1`.
Canonical evidence is
[`../evidence/scaling/100m_10b_dataset_completion_2026-08-14.md`](../evidence/scaling/100m_10b_dataset_completion_2026-08-14.md).

The VPS-fed Beam wrapper is active on RTX4090 with no session cap. ADR 0072
pins the live gateway-compatible `beam-client==0.2.207`. The training run ID is
`100m-10b-data-001`, and W&B is online on the unchanged resumable run. Finite
production metrics were observed through step 250, followed by 16-block validation loss
8.827006. The first Beam checkpoint then hung after its atomic rename during a
POSIX parent-directory `fsync`. The checkpoint independently verified on CPU as
`step-00000250`, with next block 250, and source `1f9dff9` resumed it under an
exact one-time infrastructure migration. One later RTX5090 container
disappeared without a trainer traceback after finite updates beyond step 1,650;
an exact step-1,500 recovery then remained finite through step 3,250 and
validation loss 3.722389 before failing as an incomplete checkpoint staging
directory was created. A clean RTX5090 retry subsequently failed before the
trainer command. The active segment therefore uses the supported RTX4090 lane,
resumed exact HF `step-00003000`, and advanced through at least step 3,009 with
finite loss 3.639936. The RTX4090 segment advanced through step 4,535 before
the container stopped around 14:30 UTC. Checkpoint `step-00004500` was verified
intact on Hugging Face (validation loss 3.440967, perplexity 31.217146). The run
was cleanly relaunched on Beam RTX4090 from pinned commit `1f9dff9`, exact-restoring
`step-00004500` with 71,794 planned steps remaining and resuming W&B online
telemetry in `running` state. The active path keeps Triton compilation on
container-local scratch, makes the VPS preseed guard initialize fail-closed, and
retries Beam Volume `EAGAIN` during CPU staging.
Canonical launch evidence is
[`../evidence/scaling/100m_10b_beam_launch_2026-08-14.md`](../evidence/scaling/100m_10b_beam_launch_2026-08-14.md).
The step-250 checkpoint incident and verified resume are recorded in
[`../evidence/scaling/100m_10b_step250_beam_fsync_resume_2026-08-14.md`](../evidence/scaling/100m_10b_step250_beam_fsync_resume_2026-08-14.md).
The later worker disappearance and verified step-1,500 recovery are recorded in
[`../evidence/scaling/100m_10b_beam_worker_loss_step1500_resume_2026-08-14.md`](../evidence/scaling/100m_10b_beam_worker_loss_step1500_resume_2026-08-14.md).
The step-3,250 failure, failed RTX5090 startup retry, and verified RTX4090
step-3,000 failover are recorded in
[`../evidence/scaling/100m_10b_beam_step3250_failure_rtx4090_failover_2026-08-14.md`](../evidence/scaling/100m_10b_beam_step3250_failure_rtx4090_failover_2026-08-14.md).
The step-4,535 container exit and verified step-4,500 resume are recorded in
[`../evidence/scaling/100m_10b_beam_step4500_resume_2026-08-14.md`](../evidence/scaling/100m_10b_beam_step4500_resume_2026-08-14.md).
After a Beam account/workspace change, the exact HF `step-00015500` resume was
restarted on 2026-08-16 in the new workspace. Fresh Beam volumes required a
one-time runtime-contract seed before the RTX4090 task could start; the seed
did not change scientific identity or checkpoint bytes. The resumed task ran
from source `1f9dff920ecc45ce2fdb43fd875514a18391273d` with microbatch 4, then
an earlier pre-reset billing baseline caused the deterministic notional cap to
stop the first attempt at an estimated `$51.28` before any new checkpoint was
published. That old-account estimate is no longer used. At
`2026-08-16T15:13:23Z`, the supervisor reset its billing baseline for the new
Beam account, recorded `$0.00` since reset, and relaunched a fresh bounded
segment from `step-00015500` under the `$30` notional cap. The account-cost
basis remains `$0.00`; the notional RTX4090/CPU/RAM estimate remains the hard
stop. The hourly job continues to stop an active task at the cap and relaunch
from the latest HF checkpoint after a crash while budget remains.
See [`../evidence/scaling/100m_10b_beam_account_zero_resume_2026-08-16.md`](../evidence/scaling/100m_10b_beam_account_zero_resume_2026-08-16.md)
and [`../evidence/scaling/100m_10b_beam_billing_reset_2026-08-16.md`](../evidence/scaling/100m_10b_beam_billing_reset_2026-08-16.md).

The repository-wide unit-test job is still red for unrelated existing/concurrent failures outside this lane (including test modules that import unavailable `pytest`, stale eval-entrypoint/eval-core expectations, historical ADR-shape failures, and an older remote-checkpoint state-equality regression). Do not interpret the global red job as a failure of the incremental 10B path, but also do not describe the repository as globally green.

Technical contract: [`../reference/100m_10b_incremental_dataset.md`](../reference/100m_10b_incremental_dataset.md). Active operational procedure: [`../runbooks/100m_10b_beam.md`](../runbooks/100m_10b_beam.md).

## Post-training status

The completed 20M/500M S0 SFT learned the masked SFT objective but failed behavioral qualification: 0/30 deterministic instruction cases, 0% EOS termination, 100% runaway, with modest broad `eval_core_v1` regression. Do not promote the unchanged S0 recipe solely from its lower SFT validation/test loss.

Evidence: [`../evidence/20m/20m_500m_sft_full_qualification_2026-08-11.md`](../evidence/20m/20m_500m_sft_full_qualification_2026-08-11.md).

The 100M/2B S0 bundle is built and privately published at `roccoangelella/small-llm-100m-2b-sft-s0-001`. The authenticated Kaggle round-trip passed full bundle verification and exact tree identity: 15 files, 347,155,440 bytes, tree SHA-256 `aa1f4c2bb98c9218e390e9be5ebe5152e8d20fd1938b03f044667ced259f6818`; anonymous access is denied. This records dataset readiness only, not SFT training or behavioral qualification.

The first live 100M/2B SFT hardware start OOMed during no-step backward prewarm
at per-rank microbatch 4. Microbatch 2 then passed prewarm and completed 250
finite optimizer updates, reaching 8,044,261 loss-bearing targets at 3,690
targets/s on the boundary step. Rank 1 subsequently entered the next NCCL
barrier while rank 0 owned validation/behavior work and hit the 600-second NCCL
watchdog. Evaluation, local save, and remote publication had not completed, so
no SFT checkpoint exists and the next launch starts fresh from the parent. The
pinned runtime now uses a one-hour CPU/Gloo control group for rank-zero cadence
side effects; the next live gate is successful step-250 evaluation plus local
and verified remote checkpoint publication. See
[`../evidence/scaling/100m_2b_sft_t4_microbatch4_oom_2026-08-13.md`](../evidence/scaling/100m_2b_sft_t4_microbatch4_oom_2026-08-13.md)
and [`../evidence/scaling/100m_2b_sft_step250_nccl_timeout_2026-08-13.md`](../evidence/scaling/100m_2b_sft_step250_nccl_timeout_2026-08-13.md).

## Source of truth

- Immediate priorities and gates: [`roadmap.md`](roadmap.md)
- Decisions: [`../decisions/README.md`](../decisions/README.md)
- Current architecture/backend: [`../reference/model_architecture.md`](../reference/model_architecture.md), [`../reference/gdn2_fla_backend.md`](../reference/gdn2_fla_backend.md)
- Dataset contract: [`../reference/dataset_and_tokenization.md`](../reference/dataset_and_tokenization.md)
- Training/evaluation contract: [`../reference/training_and_evaluation.md`](../reference/training_and_evaluation.md)
