---
status: current
last_reviewed: 2026-08-30
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

ADR 0071 authorized the original fresh 100M/10B trajectory. ADR 0095 later
froze the current deep-decay schedule from the exact uncooled step-15,500
state, and ADR 0114 authorizes its execution on one Modal H100. ADR 0132 now
splits durability by role: exact-resume `latest` is in the mutable checkpoint
Bucket and strict validation-loss `best` is in a dedicated recreate-on-improvement
model repository. The previous detached app is no longer live; the newest verified
durable continuation is Bucket `step-00061500`. A new Modal invocation is currently
blocked by the workspace spend limit, not by checkpoint validity. The unchanged
run ID remains `100m-10b-deep-decay-from-step15500` through the full 76,294-update
plan without a 5B continuation gate.

## GDN-2 production execution

Production CUDA GDN-2 execution is mixed FLA on `fla-core==0.5.2` with FP32 master parameters plus CUDA FP16 autocast. Saved/configured `gdn_chunk_size` is 32; FLA's internal runtime chunk is 64. The adaptive PyTorch recurrence remains the correctness/reference fallback. See [`../reference/gdn2_fla_backend.md`](../reference/gdn2_fla_backend.md).

The current production continuation uses one exact Modal H100 under ADR 0114.
It preserves the 64-sequence optimizer block and changes only execution slicing
to four ordered microbatch-16 accumulations. Kaggle two-T4 DDP and Beam remain
historical/alternate execution evidence, not the live lane.

## Dataset and checkpoint durability

For **new** dataset production, Hugging Face Storage Buckets are the only remote dataset durability backend under ADR 0054. Google Drive is historical only; legacy fields such as `drive_manifest.json` remain readable provider-neutral compatibility identifiers for already-built artifacts.

Under ADR 0132, rolling exact-resume state and selected model state use different Hugging Face transports. `latest` checkpoints use a mutable private Storage Bucket, by default `<SMALL_LLM_HF_REPO_ID>-checkpoints`, with only the newest verified checkpoint retained per run. Strict validation-loss `best` uses a dedicated per-run model repository derived as `<owner>/<base>-best-<run_id>`; every strict improvement marker-verifies the existing dedicated repo, deletes it, recreates it, and publishes the new best in one fresh repository history. Stable completed `models/...` artifacts remain model artifacts and are not moved by this transport change.

Legacy model-repository `run/...` namespaces remain restore/migration sources until their valuable state is safely classified or moved. Shared model repositories that also contain stable artifacts or other runs are not safe for wholesale deletion. Stable model artifacts are verified with their native `local_manifest.json`; `checkpoint_manifest.json` remains publication metadata for the live two-phase exact-resume protocol.

Canonical Kaggle SFT and R-SFT commands use rolling latest-only remote
retention. Each verified publication prunes superseded checkpoints only within
its own run namespace and super-squashes the Git-backed model repository, while
preserving other runs and stable `models/...` artifacts in the current tree.
Detached SFT, scaled-SFT training, and R-SFT worktrees are pinned to transport
implementation commit `184adccc1c12437046594ac674bc8d61eb710125`.

## 100M / 10B execution

ADR 0058 defines the incremental producer/consumer path: approximately-1-GiB immutable HF dataset shards, monotonic READY frontier, frozen 16-block validation prefix, cheap CPU production/staging before H100 allocation, current+successor lead window, and exact ordered H100 consumption. The frozen whole-block training horizon is 76,294 updates / 10,000,007,168 target tokens with standard WSD 3,815 warmup / 57,220 stable / 15,259 decay updates.

Dataset production is complete. HF and Beam both contain 21 train and 21
validation shards; HF reports `target_reached=true`, `producer_complete=true`,
and final manifest SHA-256
`d23e7e4641e30c25b56189093bf1270cd11e85efc8b26bc4660af1873edb96f1`.
Canonical evidence is
[`../evidence/scaling/100m_10b_dataset_completion_2026-08-14.md`](../evidence/scaling/100m_10b_dataset_completion_2026-08-14.md).

The historical VPS-fed Beam wrapper ran on RTX4090 with no session cap. ADR 0072
pinned the gateway-compatible `beam-client==0.2.207`. The training run ID was
`100m-10b-data-001`, and W&B was online on the unchanged resumable run. Finite
production metrics were observed through step 250, followed by 16-block validation loss
8.827006. The first Beam checkpoint then hung after its atomic rename during a
POSIX parent-directory `fsync`. The checkpoint independently verified on CPU as
`step-00000250`, with next block 250, and source `1f9dff9` resumed it under an
exact one-time infrastructure migration. One later RTX5090 container
disappeared without a trainer traceback after finite updates beyond step 1,650;
an exact step-1,500 recovery then remained finite through step 3,250 and
validation loss 3.722389 before failing as an incomplete checkpoint staging
directory was created. A clean RTX5090 retry subsequently failed before the
trainer command. That recovery segment therefore used the supported RTX4090 lane,
resumed exact HF `step-00003000`, and advanced through at least step 3,009 with
finite loss 3.639936. The RTX4090 segment advanced through step 4,535 before
the container stopped around 14:30 UTC. Checkpoint `step-00004500` was verified
intact on Hugging Face (validation loss 3.440967, perplexity 31.217146). The run
was cleanly relaunched on Beam RTX4090 from pinned commit `1f9dff9`, exact-restoring
`step-00004500` with 71,794 planned steps remaining and resuming W&B online
telemetry in `running` state. That path kept Triton compilation on
container-local scratch, made the VPS preseed guard initialize fail-closed, and
retried Beam Volume `EAGAIN` during CPU staging.
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
On 2026-08-17, the exact uncooled `step-00015500` checkpoint was found in the
old Beam `jourme` workspace and copied into the then-active workspace with all five
files present: the four manifests plus the 871.54 MiB `trainer_state.pkl`.
The local and remote checkpoint identities were verified before allocation.
The initial aggressive GPU worker progressed through locally valid
`step-00016750`; Hugging Face had fully published `step-00016500`. The worker
was lost during `step-00017000` staging, leaving only an ignored incomplete
`.step-00017000...` directory. Beam retried that same GPU invocation with its
original immutable `step-00015500` argument, so the duplicate retry was stopped
at step 15,630 rather than allowed to repeat work. The replacement launcher resolved
the highest manifest-verified continuation checkpoint inside every GPU worker
start, before it invokes the trainer. The prior replacement task ran from clean
local commit `af0ff1ea207ba775e23278fc86217ae8c86e2a67`; its CPU and visibility
gates and GPU-side retry resolver selected exact `step-00016750`, leaving
59,544 steps. The function requested RTX4090, but Beam's worker reported
`NVIDIA A10G` in the trainer telemetry tag; this allocation mismatch remains
visible rather than being hidden by the requested label. It ran in tmux
session `aggressive-wsqd-10b`. The A10G worker later reached step 16,785, but
was intentionally cancelled before its next checkpoint cadence to switch lanes;
`step-00016750` remains the highest valid durable restart point. The queued
RTX5090 task `73070b42-8257-462d-a038-b64aeba02018` was superseded by an
immediate RTX4090 switch at the user's direction. The
account-zero billing baseline was reset at `2026-08-17T12:04:26Z`; the monitor
recorded the A10G interval and retains a conservative RTX5090-equivalent rate
after the RTX4090 switch, preserving the `$30` notional hard stop. The dedicated
`ops/monitor_aggressive_wsqd_10b_beam.py` ran every five minutes in the
historical `small-llm-billing-guard` tmux supervisor, recognized only this
aggressive task, stopped it at the cap, and relaunched the same RTX4090 tmux
command after a crash when control-plane state was readable.

The first ADR-0114 Modal segment advanced to a locally valid
`step-00061500`, then stopped because Hugging Face rejected its checkpoint
upload at the private-repository storage limit. The trainer had finite state.
After the earlier model-repository repair, ADR 0132 moved the verified step-61,500
exact-resume tree into the dedicated checkpoint Storage Bucket. The Bucket's
`latest.json` now resolves `step-00061500`; the 913,885,544-byte trainer state
matches manifest SHA-256 `a3c8b018f49f3315a3443eb73810712dfc2adbb53bc3c49774ef693d32cf43ff`,
and the rewritten transport metadata is bound to implementation commit
`5b942181163ce5ca3f74e1ae61da4f9bcbb4e92b`. The legacy model repo has only an
incomplete step-61,750 staging attempt and its pointer remains step 61,500.
The persisted historical best validation loss is `2.8437069645151496` at step
59,250, while step 61,500 is worse at `2.8463459765771404`; because the retained
step-59,250 bytes are unavailable, no dedicated best repo was falsely initialized.
The global block remains 64 and microbatch 16 remains the execution-only four-slice
adaptation. Canonical transport-migration evidence is
[`../evidence/scaling/100m_10b_modal_hf_bucket_best_split_2026-08-30.md`](../evidence/scaling/100m_10b_modal_hf_bucket_best_split_2026-08-30.md); the preceding quota incident remains recorded in
[`../evidence/scaling/100m_10b_modal_step61500_hf_quota_repair_resume_2026-08-30.md`](../evidence/scaling/100m_10b_modal_step61500_hf_quota_repair_resume_2026-08-30.md).

The repository-wide unit-test job is still red for unrelated existing/concurrent failures outside this lane (including test modules that import unavailable `pytest`, stale eval-entrypoint/eval-core expectations, historical ADR-shape failures, and an older remote-checkpoint state-equality regression). Do not interpret the global red job as a failure of the incremental 10B path, but also do not describe the repository as globally green.

Technical contract: [`../reference/100m_10b_incremental_dataset.md`](../reference/100m_10b_incremental_dataset.md). Active operational procedure: [`../runbooks/100m_10b_deep_decay_modal.md`](../runbooks/100m_10b_deep_decay_modal.md).

## Post-training status

The 20M/500M S0 experiment remains a failed behavioral qualification despite lower held-out SFT loss; do not use that result as evidence that the unchanged S0 recipe is generally sufficient.

For the 100M/2B line, the completed S0 parent is `100m-2b-sft-s0-001`. Its frozen S0 bundle remains privately published at `roccoangelella/small-llm-100m-2b-sft-s0-001` and is the retention/parent source used by the current R-SFT trajectory.

ADR 0117 supersedes ADR 0116's provisional promotion after the completed full qualification. The expanded three-epoch checkpoint remains preserved as an experimental landmark, but it is **not a qualified model improvement or qualified default on model quality**:

```text
run ID:       100m-2b-rsft-r0-16716-e3-001
HF repo:      roccoangelella/small-llm-100m-qualification
latest step:  step-00001251
```

W&B records this run as finished after 1,251 logical optimizer steps (417 train blocks × 3 exact passes), with 40,262,469 consumed loss-bearing targets and final in-distribution validation loss `1.4549868323180837`. The frozen qualification nevertheless regresses against S0 on every headline comparison axis: eval-core loss `+0.163692`, perplexity `+5.334050`, top-1 `-0.015369`, top-5 `-0.017188`, top-10 `-0.016909`, instruction-behavior pass rate `0.066667 → 0.0`, novel-reasoning greedy accuracy `0.457143 → 0.257143`, novel-reasoning sampled pass@1 `0.392857 → 0.282143`, and frozen S0 validation loss `+0.088378`.

The run did learn part of the intended R-SFT generation protocol. In the dedicated wrapper test, the trained chat wrapper starts reasoning in `14/14` cases and is fully well formed in `9/14`; both reasoning-start and well-formed rates fall to zero under plain and `Question: … Answer:` wrappers. The project interpretation is therefore: **protocol acquisition is real, but improved reasoning is not**. Reasoning-shaped text and reasoning correctness must remain separate evaluation axes.

The root chat registry currently selects this run for the 100M/2B `--r-sft` profile. That operational pointer is not a qualification endorsement after ADR 0117; changing the runtime registry is a separate implementation action. Canonical local chat is:

```bash
.venv/bin/python chat.py --model_params 100M --num_tokens 2B --r-sft
```

The previous accepted run `100m-2b-rsft-r0-12306-001` remains preserved as a historical completed R0 and is still loadable explicitly with `--run-id`. Its historical training corpus had SHA-256 `e7d83f9809a65bcb50a6dea3087813d92fea1950a716b3c1eb13e87bfe263a5e` and contained 12,306 unique normalized prompts: 7,683 unchanged Superior instruction rows, 3,993 accepted unique Variant-D rewrites, and 630 Gemini logic anchors. The intermediate corpus file was removed from the current tree after the 16,716-row expansion became canonical; it remains recoverable from Git history at `2ae60bfa135017353f39da2ef34a6124cda465dc`. Its verified native bundle used one exact pass, 32,768 loss-bearing target tokens per optimizer block, and 361 train blocks for 11,609,452 total targets; its completed Hugging Face `latest.json` pointer resolves `step-00000361`.

The superseded Hugging Face R-SFT trial namespaces `100m-2b-rsft-r0-atomic-pilot-001`, `100m-2b-rsft-r0-atomic-repeat-e10-001`, and `100m-2b-rsft-r0-textual-pilot-001` were deleted after the earlier accepted run completed. The e3 experimental run, the historical 12,306-row run, and completed S0 parent remain preserved.

The expanded over-context corpus is complete under ADR 0106. Historical v1 curation and the 1,122 accepted Variant-D batches remain immutable evidence for the completed 12,306-row model. Expansion curation v2 (`manual-curation.expanded-v2.jsonl`, SHA-256 `fb4da2929b47ececbde839da199437144677e4c7e1ea52ef2e8f6d4525ae1cde`) retained 8,473 keepers. All 4,464 previously missing keepers now have accepted compressed supervision, so the keeper-resume status is `resume_pending_records=0` and 1,116/1,116 resume batches are complete. One stubborn candidate in batch 305 was recovered as an audited safe-refusal compression after Gemini repeatedly returned empty completions for an appended unsafe image-generation request involving minors; no alternate provider was used.

The frozen expanded corpus is `artifacts/rsft-superior-instruction-r0-expanded/reasoning.jsonl`, SHA-256 `d13052b6fc33108ec65511b790a75f6473144855059b16b55167b046f787c405`. It contains 16,716 unique normalized prompts: 7,683 unchanged Superior instruction rows, 8,403 accepted unique simplified Superior rows, and 630 Gemini logic anchors. The finalizer excluded 70 otherwise accepted rewrites because their normalized prompts collided with the baseline or another accepted rewrite. Every row fits the exact atomic 2,048-token serialization; the observed serialized-token range is 61–2,048.

The verified expanded native bundle is `/home/ubuntu/Projects/small-llm-work/rsft-r0-superior-instruction-expanded-16716`. Its train split has 417 optimizer blocks / 20,313 packed records and 13,420,823 loss-bearing targets: 12,077,733 reasoning targets plus 1,343,090 completed-S0 retention targets. Validation and test each contain four blocks. ADR 0108 promotes this corpus to the standard Kaggle `train` input. The one-epoch command defaults to run ID `100m-2b-rsft-r0-16716-001`; ADR 0111 also permits exact production replay through `--num-epochs N`, with automatic epoch-specific IDs (`--num-epochs 2` → `100m-2b-rsft-r0-16716-e2-001`) and 834 steps for two passes. ADR 0116's provisional e3 promotion is superseded by ADR 0117 after full qualification; no new R-SFT training recipe is selected by that decision.

Canonical evidence: [`../evidence/rsft_e3_full_qualification_2026-08-24.md`](../evidence/rsft_e3_full_qualification_2026-08-24.md), [`../evidence/rsft_r0_12306_training_completion_2026-08-19.md`](../evidence/rsft_r0_12306_training_completion_2026-08-19.md), [`../evidence/rsft_expansion_resume_2026-08-20.md`](../evidence/rsft_expansion_resume_2026-08-20.md), and [`../evidence/rsft_expanded_corpus_completion_2026-08-21.md`](../evidence/rsft_expanded_corpus_completion_2026-08-21.md). Active procedure: [`../runbooks/rsft_r0_atomic_production.md`](../runbooks/rsft_r0_atomic_production.md).

## Source of truth

- Immediate priorities and gates: [`roadmap.md`](roadmap.md)
- Decisions: [`../decisions/README.md`](../decisions/README.md)
- Current architecture/backend: [`../reference/model_architecture.md`](../reference/model_architecture.md), [`../reference/gdn2_fla_backend.md`](../reference/gdn2_fla_backend.md)
- Dataset contract: [`../reference/dataset_and_tokenization.md`](../reference/dataset_and_tokenization.md)
- Training/evaluation contract: [`../reference/training_and_evaluation.md`](../reference/training_and_evaluation.md)
