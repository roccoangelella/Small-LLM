# Project Status

_Last updated: 2026-08-05 15:34 Europe/Rome_

## Current phase

The approximately-20M engineering qualification, complete 10M-token one-pass run, remote checkpoint lifecycle, and post-pretraining prompt suite are complete and accepted.

The authorized next experiment remains the same approximately-20M model trained for one pass on a fixed approximately-100M-token finite dataset. No model enlargement or later logarithmic stage is authorized yet.

```text
status: 100m_build_and_private_kaggle_publish_ready
current authorization: 20m_model_on_100m_tokens_only
future dataset convention: 10M -> 100M -> 1B -> 10B -> approximately 90B
execution venue: Kaggle NVIDIA T4
model enlargement authorized: no
```

## Completed 20M / 10M result

```text
parameters: 20,637,592
architecture: gdn2_hybrid
context: 2,048
precision: FP16
optimizer: hybrid whole-matrix Muon + AdamW
seed: 17
accepted source tokens: 10,000,662
optimizer updates: 306
status: completed
FP16 overflow events: 0
final validation loss: 6.136690
final validation perplexity: 462.520157
final remote checkpoint: step-00000306
```

The checkpoint is accepted as an engineering and learning-signal success, not as a capable chatbot. Detailed evidence remains in the 20M qualification, repeatability, resume, remote-recovery, and qualitative-result records.

## Fixed 100M dataset identity

```text
producer: dataset.qualification_100m
report: dataset.qualification_100m_report
run ID: 20m-100m-dataset-001
target accepted source tokens: 100,000,000
minimum: 90,000,000
hard maximum: 110,000,000
context: 2,048
stored tokens per sequence: 2,049
sequences per optimizer update: 16
target tokens per full update: 32,768
target shard size: 8 MiB
producer durable checkpoint cadence: 20,000,000 source tokens
remote durability: required
passes: 1
implicit wraparound: forbidden
```

## One-command VPS build and Kaggle publication

The user decided to use the official `kagglehub` Python library, authenticated with `KAGGLE_API_TOKEN` from `.env`, and one command for dataset production and private Kaggle publication.

```bash
cd /path/to/Small-LLM
git switch main
git pull --ff-only
bash kaggle/build_and_push_100m.sh
```

Implemented on `main`:

```text
wrapper: kaggle/build_and_push_100m.sh
implementation: kaggle/build_and_push_100m.py
dependency pin: kagglehub==1.0.2
environment template: kaggle/100m-publish.env.example
offline tests: tests/test_build_and_push_100m.py
```

The suite automatically builds or resumes the canonical producer, performs a full scan, derives the exact plan, stages only `manifest.json`, `drive_manifest.json`, `qualification_plan.json`, `train/`, and `validation/`, uploads with `kagglehub.dataset_upload`, downloads the complete dataset back, checks byte-identical tree identity, reruns the full scan, and requires anonymous access to be denied.

New Kaggle datasets created by the pinned `kagglehub` release are explicitly marked private. The additional anonymous-access probe prevents silently publishing into an existing public handle.

The suite records an idempotent verified receipt and skips duplicate Kaggle versions when the same tree was already published.

## Frozen training recipe

```text
parameters: 20,637,592
model size: smoke
architecture: gdn2_hybrid
layer pattern: [GDN-2, GDN-2, GDN-2, full gated MHA] x 2
context: 2,048
GDN-2 chunk size: 32
initialization: normal
precision: FP16
seed: 17
optimizer: hybrid whole-matrix Muon + AdamW
base LR: 3e-4
weight decay: 0.1
Muon momentum: 0.95
Muon target direction RMS: 0.18
global gradient clipping: 1.0
schedule: one-pass WSD
minimum LR ratio: 0.1
```

## Microbatch and segmented Kaggle execution

```text
baseline microbatch: 1
candidate microbatch: 4
effective optimizer block: unchanged at 16 sequences
probe: first 8 blocks
minimum median throughput improvement: 5%
maximum per-step loss delta: 0.05
maximum relative gradient-norm delta: 5%
maximum reserved T4 memory: 90%
FP16 overflow tolerance: zero
fallback: none; fail closed
```

Each Kaggle invocation runs at most 749 additional updates, publishes an explicit final private checkpoint, and preserves one W&B run. Later accounts restore model, optimizer, scheduler, scaler, RNG, and exact data cursor from the private Hugging Face latest pointer.

Official training command:

```bash
cd /kaggle/working/Small-LLM
git switch main
git pull --ff-only
python kaggle/run_20m_100m.py
```

Pinned evidence-producing commit:

```text
43190cb72443a2de290dc8e6f2c54f29d8dff501
```

## Immediate next actions

```text
1. Add KAGGLE_API_TOKEN and KAGGLE_USERNAME or an exact dataset handle to .env.
2. Run bash kaggle/build_and_push_100m.sh on the VPS.
3. Require build-and-push-summary.json to report completed or already_published.
4. Require kaggle-publish-state.json to report verified.
5. Attach that private dataset to a T4 notebook.
6. Run python kaggle/run_20m_100m.py and review the live microbatch gate.
7. Re-run the same training entry point after every clean published segment until completion.
```

Detailed records:

```text
llm_docs/20m_100m_data_scaling_plan.md
llm_docs/20m_100m_runbook.md
llm_docs/100m_kagglehub_publication_suite.md
```
