---
status: current
last_reviewed: 2026-08-12
---

# Current project status

This file is high-freshness working memory. Detailed measurements and investigation chronology live in evidence/reference/archive and are linked rather than duplicated here.

## Completed 20M / 100M pretraining

```text
W&B run ID: 20m-100m-data-004
optimizer updates: 3,053
consumed training target tokens: 100,018,176
final validation loss: 4.252758495143203
final validation perplexity: 70.29906475797992
final checkpoint: step-00003053
```

The trajectory remained trainable but suffered a large late-run throughput collapse in the old adaptive PyTorch GDN-2 backend. Historical operating records are archived under `../archive/20m_100m/`; measured incidents/results are under `../evidence/20m_100m/`.

## Completed 20M / 500M pretraining

```text
W&B run ID: 20m-500m-data-001
final checkpoint: step-00015264
consumed training target tokens: 500,156,416
architecture: gdn2_hybrid
d_model: 256
d_ff: 704
layers: 8
context: 2,048
precision: FP16 autocast with FP32 master parameters
```

The accepted 500M checkpoint chain used the adaptive backend through step 4000 and mixed FLA for the accepted continuation. Treat throughput and very fine loss-curve comparisons across that boundary accordingly.

The frozen post-pretraining qualitative suite showed materially stronger answer-shaped/schema continuation than the earlier smoke-scale evidence, including stable Q/A and Alice/Ben surface structure, while factual/arithmetic answering and open-ended semantic stability remained weak. Under ADR 0027 this is enough evidence to keep the approximately-20M model fixed through the already-authorized 2B probe, not evidence for unlimited token scaling.

Canonical qualitative evidence: [`../evidence/20m/20m_500m_post_pretraining_full_suite_2026-08-10.md`](../evidence/20m/20m_500m_post_pretraining_full_suite_2026-08-10.md)

## Qualified GDN-2 production backend

Production CUDA execution is **mixed FLA on `fla-core==0.5.2`** under FP32 master parameters plus CUDA FP16 autocast.

```text
GPU qualified: Tesla T4 / SM75
PyTorch: 2.10.0+cu128
CUDA runtime: 12.8
Triton: 3.6.0
fla-core: 0.5.2
saved/configured gdn_chunk_size: 32
FLA internal runtime chunk: 64
```

The corrected deterministic qualification passed the requested synthetic decay sweep and the exact real step-4000 next-block forward/backward gate. Warmed true-block throughput measured 22,765.80 target tok/s for mixed FLA versus 1,964.75 target tok/s for the adaptive FP32 recurrence. The adaptive backend remains the correctness/reference fallback; full-FP32 FLA remains diagnostic/fallback.

Current contract: [`../reference/gdn2_fla_backend.md`](../reference/gdn2_fla_backend.md)

## 100M / 2B Modal trajectory and Hugging Face artifact transport

ADR 0041 authorizes the approximately-100M-parameter / 2B-token Modal trajectory using the byte-preserving block-64 derivative of the verified 2B corpus. ADR 0043 keeps all Kaggle-to-Modal preparation and Modal operation on the VPS. **ADR 0055 now unifies Modal checkpoint durability and stable model artifacts on `SMALL_LLM_HF_REPO_ID`, superseding the separate checkpoint-Storage-Bucket decisions ADR 0047 and ADR 0052.** ADR 0056 explicitly excludes Modal from the Kaggle dual-T4 execution change: Modal training remains **one H100 per training run**.

By 2026-08-11 16:10 Europe/Rome, the user reported that the production training run had been started on an **H100 GPU**. On 2026-08-12 the original Modal account exhausted its credits and the user moved operational setup to a new Modal account/workspace. Modal Volumes do not carry across that account boundary, so cross-workspace continuity depends on a verified Hugging Face checkpoint or stable artifact.

```text
model preset: 100M / trainer substantive
W&B run ID: 100m-2b-data-001
dataset profile: modal-2b-b64
dataset run ID: modal-2b-b64-dataset-001
context: 2,048
prepared optimizer block: 64 sequences
full-block target tokens: ~131,072
historical planned optimizer updates in the original status record: 15,259
precision: FP16 autocast with FP32 master parameters
microbatch qualification candidates: 16, 32, 48, 64
execution topology: single H100; no DDP from ADR 0056
GPU actually reported for original live run: H100
same-workspace checkpoint cadence: Modal Volume every 250 successful updates + final
cross-workspace checkpoint cadence for future Modal execution: HF model repository every 500 successful updates + final
live HF namespace: run/100m-2b-data-001/
live HF retention: latest resumable checkpoint, with superseded paths pruned and repository history squashed
stable HF namespace: models/100m-2b-data-001/<checkpoint_id>/
validation cadence: every 250 successful updates
```

The original runtime probed 16/32/48/64 before optimizer step 1 and froze the fastest safe measured execution microbatch. The selected live microbatch has not yet been copied into project memory; use surviving checkpoint/W&B evidence rather than guessing it.

On 2026-08-12 the user reported manually moving the surviving 100M/2B checkpoint from the former checkpoint bucket to:

```text
models/100m-2b-data-001/step-00015267
```

The stable-model evaluator now accepts that canonical `models/...` layout directly. If `models/100m-2b-data-001/artifact.json` is absent because the checkpoint was moved manually, it discovers the highest `step-XXXXXXXX` directory under the run namespace, downloads the complete tree, verifies `local_manifest.json` and the embedded checkpoint publication manifest, and only then loads the native model state.

For future Modal execution, the current runtime first resumes from a verified checkpoint in the current workspace's `small-llm-runs` Volume. If that is empty it checks the model repository's `run/<run_id>/latest.json` two-phase pointer. Live checkpoint objects are verified before `latest.json` advances. Rolling cleanup removes superseded checkpoint paths and squashes model-repository history so periodic checkpoint publication does not accumulate an unbounded Git history. A live remote restore remains fail-closed on source-commit mismatch.

A completed Modal run also publishes its verified final checkpoint under `models/<run_id>/<checkpoint_id>` and updates `models/<run_id>/artifact.json`. The former private checkpoint Storage Bucket is retained only as a legacy restore source for already-produced checkpoints; new checkpoint writes do not use it. `SMALL_LLM_HF_CHECKPOINT_BUCKET_ID` is therefore legacy compatibility configuration only.

This checkpoint decision does **not** retire Hugging Face Storage Buckets generally. The rolling 10B dataset uses a dataset Storage Bucket because large mutable shard/object transport is a different workload from versioned model checkpoints. Its dataset-bucket path remains active under the rolling-dataset decisions.

The external ten-minute `modal/publish_hf.py` loop remains retired for live durability. Its explicit final-publication behavior is now generalized by the stable model-artifact helper and automatic final Modal publication.

The VPS dataset preparation account-migration bug is closed by ADR 0048: `modal/prepare_dataset.py` no longer trusts `.modal-2b-b64-upload.json` to skip transfer. Every upload-enabled run resolves the actually authenticated Modal workspace/environment, opens or creates `small-llm-data`, verifies the canonical destination from remote file sizes plus the manifest SHA-256, uploads only when needed, and verifies again after `modal volume put`. A workspace switch therefore uses the normal `python modal/prepare_dataset.py` command; `--force-upload` is reserved for explicit re-upload repair.

The checkpoint-transport changes are infrastructure-only relative to the scientific trajectory: they do not change model geometry, optimizer math, schedule semantics, dataset token ordering, or checkpoint bytes.

This is a distinct trajectory from the historical/ongoing 20M / 2B Kaggle run. The 2B sequence bytes and train/validation ordering are preserved by the reblock, while the optimizer batch is intentionally larger for Hopper utilization.

## Active experiment — fresh 20M / 2B

ADR 0023 authorizes a new independent approximately-2B-token data-scaling trajectory. ADR 0027 adds the completed 500M qualitative result as justification for keeping model size fixed through this point. ADR 0056 now makes exact-batch two-T4 DDP the standard Kaggle execution backend after the live qualification passed every numerical and throughput gate.

```text
profile: 20m-2b-data-scaling-v1
dataset run ID: 20m-2b-dataset-001
W&B run ID: 20m-2b-data-001
target accepted source tokens: 2,000,000,000
minimum: 1,800,000,000
maximum: 2,200,000,000
producer durable checkpoint cadence: 80,000,000 source tokens
fresh initialization seed: 17
training microbatch: 4
global optimizer block: 16 sequences
Kaggle execution: 2 x Tesla T4 DDP, 8 sequences/rank, 2 local microbatches/rank
DDP sync: no_sync on first local microbatch, synchronized second local backward
DDP gradient normalization: world_size * local_loss_sum / global_target_tokens
training durability / validation / remote publication cadence: 250 updates
```

The accepted dual-T4 qualification measured 20,183.50 target tok/s median on one warmed T4 and 34,292.22 target tok/s under warmed two-T4 DDP, a 1.6990x speedup above the predeclared 1.60x gate. Maximum loss delta was 4.77e-6 and maximum gradient relative delta was 7.54e-6; final parameter and optimizer-state comparisons also passed. Production adds synchronized GradScaler/non-finite agreement before any optimizer mutation, rank-zero-only side effects, and raw-model checkpoint serialization. Canonical evidence: [`../evidence/20m/20m_2b_dual_t4_ddp_qualification_2026-08-12.md`](../evidence/20m/20m_2b_dual_t4_ddp_qualification_2026-08-12.md).

The 2B trajectory is fresh relative to 500M and the superseded 1B setup. It starts from seed 17 and uses the qualified mixed FLA CUDA backend from optimizer update 1. The exact optimizer-update count and WSD boundaries come from the completed verified manifest rather than a nominal hard-coded count.

Data path remains:

```text
pinned Nemotron-ClimbMix source
  -> VPS deterministic finite build
  -> immutable uint16 shards + verified Google Drive mirror
  -> private Kaggle publication + round-trip byte verification
  -> attached Kaggle dataset
  -> exact-batch 2 x T4 DDP training from Kaggle-local input
```

Current operational state: **the 2B trajectory has been launched and user-observed W&B data on 2026-08-11 covered optimizer steps 23,438 through 28,438; the exact live step may be later.** The consolidated dataset identity/geometry remains centralized in `dataset.qualification` (`--profile 20m-2b`) under ADR 0037. Existing topology-neutral checkpoints are intended to resume through the Kaggle DDP adapter without changing model keys, TrainerConfig identity, optimizer batch geometry, or dataset cursor semantics.

A fresh Kaggle clone after the dataset/Kaggle cleanup exposed a launcher-path regression: direct `python kaggle/launch.py ...` execution could not import the top-level `dataset` package because only the `kaggle/` script directory was guaranteed on `sys.path`. Commit `27406f4a12fa450902e6ead1d9f95fbe51da6fce` fixes the human-facing launcher by adding the repository root before runtime/profile resolution.

The subsequent repository-wide dataset-layout audit found a second latent boundary bug: the pinned per-experiment training worktree predates the consolidated `dataset.qualification` module, while the shared training engine tried to run current dataset verification/plan commands from that historical worktree. Commit `356c924ee5f39f5365e6608c7a8b9f3a070fd0d0` now keeps model/trainer execution pinned but routes `dataset.main` and `dataset.qualification` control-plane subprocesses through the clean controlling checkout. The dual-T4 production adapter preserves this split: model/trainer semantics remain sourced from the pinned experiment worktree while only the Kaggle execution topology is injected from the controlling checkout. The audit also removed retired flat `train.bin`/`validation.bin` path constants, refreshed active launcher documentation, and added repository-wide regression coverage for deleted dataset modules, the schema-v2 layout, packaging, active commands, and direct-launch import behavior.

## Frozen decisions still in force for the 2B probe

- Keep the pinned source revision, GPT-2 token IDs, and programming-cluster-11 exclusion policy.
- Keep context length 2,048 and the existing 20M `gdn2_hybrid` geometry.
- Keep saved/model `gdn_chunk_size=32`; CUDA FLA executes internal chunk 64.
- Keep adaptive PyTorch as correctness/reference fallback; do not change learned decay solely for runtime behavior.
- Use microbatch 4 on the qualified T4 path.
- Use exact-batch two-T4 DDP as the standard Kaggle training topology; preserve the 16-sequence global optimizer block.
- Keep Modal training single-H100; ADR 0056 does not authorize Modal DDP.
- Preserve fail-closed FP16 scaler/atomic-block behavior.
- Preserve `eval_core_v1` plus free-generation and teacher-forced confidence/rank diagnostics.
- New finite scaling trajectories start fresh unless a later ADR explicitly authorizes continuation.
- Revisit additional fixed-size token scaling and model enlargement after the frozen 2B comparison; SFT qualification is authorized in parallel under ADR 0032.

## Completed 500M-parent SFT qualification

ADR 0032 authorized the completed 20M/500M checkpoint as the first SFT qualification parent and froze the 4%-of-parent scaling rule. ADR 0033 froze the comprehensive parent-versus-SFT scorecard and the pretraining-equivalent T4 microbatch/cadence defaults.

The full qualification is now complete for `20m-500m-sft-s0-001`:

```text
parent checkpoint: step-00015264
parent consumed targets: 500,156,416
SFT checkpoint: step-00000621
SFT consumed/train loss-bearing targets: 20,006,234
bundle status: verified
validation targets: 526,446
test targets: 526,473
```

The SFT objective was learned strongly on held-out SFT data: validation loss improved from 3.212253 to 2.639931 and test loss from 3.185668 to 2.609139. However, this did **not** translate into usable deterministic instruction following. The 30-case instruction suite remained 0/30 passed, with 0% EOS termination and 100% runaway generation in every evaluated checkpoint. Mean trigram repetition improved from 0.5742 to 0.4626, but no behavior category acquired a passing case.

Base capability also regressed modestly and broadly on unchanged `eval_core_v1`: loss increased from 4.007289 to 4.047304 (+0.040016), perplexity from 54.997550 to 57.242943 (+2.245393), top-1 accuracy fell by 0.004536, and calibration ECE worsened by 0.012486. Eighteen of nineteen reported clusters regressed in loss and all eight position buckets worsened.

Current interpretation: **the 500M-parent S0 run is a failed behavioral SFT qualification despite its strong held-out SFT-loss improvement.** It validates that the pipeline optimizes the intended masked target distribution, but it is not evidence that the unchanged S0 recipe should be promoted solely on SFT validation/test loss. The next SFT recipe/selection decision remains open.

Canonical evidence: [`../evidence/20m/20m_500m_sft_full_qualification_2026-08-11.md`](../evidence/20m/20m_500m_sft_full_qualification_2026-08-11.md)

## Current source of truth

- Roadmap: [`roadmap.md`](roadmap.md)
- Durable decisions: [`../decisions/README.md`](../decisions/README.md)
- Modal 100M / 2B block-64 decision: [`../decisions/0041-use-block64-modal-corpus-and-probe-microbatch-16-32-48-64.md`](../decisions/0041-use-block64-modal-corpus-and-probe-microbatch-16-32-48-64.md)
- VPS-only Modal preparation decision: [`../decisions/0043-prepare-modal-block64-corpus-on-vps.md`](../decisions/0043-prepare-modal-block64-corpus-on-vps.md)
- Final HF artifact decision: [`../decisions/0044-publish-100m-2b-final-model-to-hugging-face.md`](../decisions/0044-publish-100m-2b-final-model-to-hugging-face.md)
- Unified Modal/HF model-repository checkpoint decision: [`../decisions/0055-unify-modal-checkpoints-on-hf-model-repository.md`](../decisions/0055-unify-modal-checkpoints-on-hf-model-repository.md)
- Kaggle dual-T4 / Modal single-H100 execution decision: [`../decisions/0056-adopt-exact-batch-dual-t4-ddp-for-kaggle-only.md`](../decisions/0056-adopt-exact-batch-dual-t4-ddp-for-kaggle-only.md)
- Dual-T4 qualification evidence: [`../evidence/20m/20m_2b_dual_t4_ddp_qualification_2026-08-12.md`](../evidence/20m/20m_2b_dual_t4_ddp_qualification_2026-08-12.md)
- Modal dataset workspace-verification decision: [`../decisions/0048-verify-modal-dataset-in-active-workspace.md`](../decisions/0048-verify-modal-dataset-in-active-workspace.md)
- Modal training runbook: [`../runbooks/modal_training_launcher.md`](../runbooks/modal_training_launcher.md)
- 2B decision: [`../decisions/0023-run-2b-20m-probe-via-vps-kaggle-dataset.md`](../decisions/0023-run-2b-20m-probe-via-vps-kaggle-dataset.md)
- 500M scaling interpretation: [`../decisions/0027-use-500m-schema-gains-to-justify-fixed-20m-token-scaling-through-2b.md`](../decisions/0027-use-500m-schema-gains-to-justify-fixed-20m-token-scaling-through-2b.md)
- Unified Kaggle runtime decision: [`../decisions/0030-consolidate-kaggle-profile-wrappers-behind-one-runtime.md`](../decisions/0030-consolidate-kaggle-profile-wrappers-behind-one-runtime.md)
- Dataset tooling consolidation: [`../decisions/0037-consolidate-dataset-profile-tools-and-retire-one-off-qualification-code.md`](../decisions/0037-consolidate-dataset-profile-tools-and-retire-one-off-qualification-code.md)
- Memory-governance decision: [`../decisions/0031-govern-project-memory-with-progressive-disclosure.md`](../decisions/0031-govern-project-memory-with-progressive-disclosure.md)
- SFT scaling decision: [`../decisions/0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md`](../decisions/0032-scale-sft-budget-with-pretraining-and-qualify-on-500m-first.md)
- SFT scorecard/cadence decision: [`../decisions/0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md`](../decisions/0033-use-comprehensive-post-sft-qualification-and-pretraining-cadence.md)
- 2B runbook: [`../runbooks/20m_2b_runbook.md`](../runbooks/20m_2b_runbook.md)
- SFT runbook: [`../runbooks/sft_s0_runbook.md`](../runbooks/sft_s0_runbook.md)
- Dataset contract: [`../reference/dataset_and_tokenization.md`](../reference/dataset_and_tokenization.md)
- FLA backend contract: [`../reference/gdn2_fla_backend.md`](../reference/gdn2_fla_backend.md)