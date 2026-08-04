# Approximately-20M First Pretraining Launch Decisions

_Last updated: 2026-08-04_

## Purpose

This document records the remaining user decisions for the first approximately-20M integrated pretraining qualification. It complements `20m_dataset_scope.md`, `20m_training_readiness.md`, and `20m_qualification_protocol.md`.

The run remains an engineering qualification. It is not a model-quality or architecture-ranking experiment.

## Trainer pass count — fixed

The first qualification performs exactly one pass over the finite qualification dataset.

```text
passes: 1
implicit wraparound: forbidden
repeated presentation: forbidden for this run
```

The trainer stops at the end of the verified manifest. Repeating the same finite dataset is a separate later experiment.

## Dataset lifecycle — fixed

The finite qualification dataset is completed before the trainer starts.

For this decision, **completed** means that the separate approximately-10M accepted-source-token dataset has:

1. reached its approved 10M / 9M / 11M accepted-source-token envelope;
2. finalized every schema-v2 train and validation shard with `context_length=2048` and `sequences_per_block=16`;
3. durably uploaded every finalized shard to Google Drive;
4. verified every remote object by identity, size, and checksum;
5. produced a completed, fail-closed verified manifest.

It does not mean building the future 90B production corpus.

The first training qualification therefore consumes a fixed completed manifest. Producer/trainer overlap, queue starvation, and live-cache lifecycle are qualified later as a separate operational test so they cannot obscure trainer, optimizer, FP16, checkpoint, or resume failures in the first run.

## Validation distribution — fixed

Validation remains deterministic and keeps the accepted cluster distribution policy untouched.

The build continues to use the frozen document-identity hash split and the existing exact cluster-mixture scheduler. The qualification must not:

- change accepted or excluded cluster IDs;
- change production cluster weights;
- rebalance validation with new per-cluster quotas;
- oversample a cluster to make the small validation set look more balanced;
- move a document between train and validation after the build.

After the dataset is complete, the project freezes:

- the verified dataset and Drive-manifest hashes;
- the complete ordered validation block-ID list used by the qualification;
- validation token and document counts by cluster;
- a deterministic generation-prompt file and its SHA-256.

Because the frozen validation probability is small, the validation result is a functional health signal rather than a strong estimate of model quality. Any absent or noisy cluster representation is reported rather than silently corrected by changing the mixture.

## Seed policy — fixed for engineering qualification

The first integrated qualification uses only seed `17`.

This is acceptable because the run asks whether one exact model/data/optimizer/checkpoint configuration executes correctly, remains numerically healthy, and resumes consistently. The same-seed uninterrupted reference and A/A repeatability runs are the controls for implementation and hardware nondeterminism.

This decision does **not** authorize single-seed model-quality claims. The approximately-100M architecture comparison should use a staged multi-seed policy: one screening seed for all candidates, then additional seeds for close or finalist configurations before selecting an architecture.

## Empty-environment restore prefetch — fixed initial test

The first remote-recovery qualification prefetches two consecutive train shards, starting with the shard that contains the next unconsumed block.

```text
prefetch_shards: 2
```

The recovery test must measure post-restore data wait. If two shards do not keep the T4 supplied while later shards are fetched, the window may be increased based on measured shard size, download latency, and consumption rate. The initial value is a qualification input, not a permanent production constant.

## Exact-commit verification — deferred to T4 connection

GPU-specific exact-commit verification occurs after the launch implementation is frozen and the NVIDIA T4 session is available.

The operator must record the exact Git commit, pull that commit on the T4 environment, run the complete offline suite there, and then run the bounded T4 preflight. There is no benefit in treating the pre-remote-publication commit as the final launch commit.

## Qualification telemetry in plain language

Qualification telemetry is the training run's health log. It answers practical questions while the model trains:

- Is the loss finite and generally moving in the expected direction?
- Is FP16 overflowing, reducing its scale, or skipping updates?
- Are gradients being clipped frequently because updates are too large?
- Are the Muon and AdamW parameter groups both receiving sensible updates?
- Is GPU memory safely below the out-of-memory limit?
- Is the GPU computing, or waiting for data?
- How much time is spent saving checkpoints, validating, and uploading recovery points?
- Exactly which code, data, model, optimizer routing, and schedule produced the log?

These signals do not improve the model by themselves. They make failures diagnosable and prevent a run from being called successful merely because the process stayed alive.

## Live remote checkpoint publication — implemented

The trainer CLI now supports synchronous fail-closed two-phase publication to a private Hugging Face repository. Enabling publication requires a verified Drive manifest. At each publication boundary the CLI first creates the local atomic joint checkpoint, uploads and verifies the checkpoint tree, and only then advances the remote `latest.json` pointer.

The relevant CLI controls are:

```text
--remote-publish-every-steps <N>
--remote-drive-manifest <verified-drive-manifest.json>
--remote-checkpoint-repo <owner/private-repository>
--remote-checkpoint-revision <optional-revision>
--remote-token-env HF_TOKEN
--remote-create-repo
```

`SMALL_LLM_HF_REPO_ID` remains the environment fallback for the repository ID. A clean run also publishes its final checkpoint when the final update is not already a publication boundary.

Remote publication failure is fatal to the command and does not erase the already completed local checkpoint or advance the remote pointer.
