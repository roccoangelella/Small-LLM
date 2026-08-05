# Project Status

_Last updated: 2026-08-05 15:26 Europe/Rome_

## Current phase

The approximately-20M engineering qualification, complete 10M-token one-pass run, remote checkpoint lifecycle, and post-pretraining prompt suite are complete and accepted.

The project is now preparing the authorized **20M-model / 100M-token data-scaling experiment**. The model is not being enlarged in this stage. The only scientific scale change is the finite accepted-source-token envelope from approximately 10M to approximately 100M; microbatch 4 is an execution optimization that must pass an explicit first-session gate while preserving the same 16-sequence optimizer block.

```text
status: 100m_experiment_implementation_complete_dataset_build_pending
current authorization: 20m_model_on_100m_tokens_only
future logarithmic dataset convention: 10M -> 100M -> 1B -> 10B -> approximately 90B
later stages authorized now: no
model enlargement authorized now: no
execution venue: Kaggle
accelerator target: NVIDIA Tesla T4
```

## Completed 20M / 10M qualification

```text
parameters: 20,637,592
architecture: gdn2_hybrid
context: 2,048
precision: FP16
GDN-2 chunk size: 32
optimizer: hybrid whole-matrix Muon + AdamW
seed: 17
accepted source tokens: 10,000,662
planned train target tokens: 10,006,528
optimizer updates: 306
status: completed
launcher exit code: 0
trainer exit code: 0
FP16 overflow events: 0
final validation loss: 6.136690
final validation perplexity: 462.520157
final remote checkpoint: step-00000306
```

The W&B history was complete and contiguous. Validation improved at every recorded boundary. Gradient clipping was concentrated early rather than sustained through the run. Dataset consumption, optimizer telemetry, checkpoint cadence, final remote publication, and operational overhead passed their project gates.

The post-pretraining qualitative suite confirmed non-random local English next-token structure, stable generation, punctuation and formatting behavior, and local topic associations. It did not demonstrate reliable factual retrieval, instruction following, or useful chatbot behavior. The checkpoint is accepted as an engineering and learning-signal success, not as a capable language model.

Detailed records:

```text
llm_docs/20m_training_readiness.md
llm_docs/20m_repeatability_results.md
llm_docs/20m_local_resume_results.md
llm_docs/20m_remote_recovery_results.md
llm_docs/20m_post_pretraining_qualitative_results.md
```

## Authorized 100M-token experiment

### Dataset identity

```text
producer module: dataset.qualification_100m
report module: dataset.qualification_100m_report
run ID: 20m-100m-dataset-001
accepted-source-token target: 100,000,000
minimum: 90,000,000
hard maximum: 110,000,000
context length: 2,048
stored tokens per sequence: 2,049
sequences per optimizer block: 16
target tokens per full optimizer update: 32,768
target shard size: 8 MiB
producer durable checkpoint cadence: 20,000,000 source tokens
remote durability: required
passes: 1
implicit wraparound: forbidden
```

The completed dataset will be attached once as a private Kaggle Dataset and read from immutable local shards under `/kaggle/input`. Google Drive remains the durable mirror and recovery source; ordinary training does not stream shards over the network.

### Frozen model and optimizer

The 20M model recipe remains unchanged:

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
Muon LR multiplier: 1.0
Muon target direction RMS: 0.18
Muon weight decay: 0.1
global gradient clipping: 1.0
schedule: one-pass WSD
minimum LR ratio: 0.1
```

### Microbatch qualification

```text
baseline microbatch: 1
candidate microbatch: 4
effective optimizer block: unchanged at 16 sequences
probe prefix: first 8 blocks
throughput requirement: at least 5% median improvement
maximum per-step loss delta: 0.05
maximum relative gradient-norm delta: 5%
maximum reserved-memory fraction: 90% of T4
FP16 overflow tolerance: zero
fallback on gate failure: none; fail closed
```

### Kaggle segmentation and resume

The 100M run is structured as bounded exact segments so it does not depend on an abrupt Kaggle session timeout.

```text
maximum additional updates per invocation: 749
local checkpoint cadence: 250 updates
validation cadence: 500 updates
periodic remote publication cadence: 500 updates
cross-session authority: private Hugging Face latest pointer
W&B run ID: 20m-100m-data-001
resume policy: must
```

Each invocation either starts fresh when no remote pointer exists or restores the latest verified checkpoint, checks its embedded Drive-manifest identity against the attached dataset, restores optimizer/scheduler/scaler/RNG/data cursor state, and continues the same W&B run. Every normal segment exit requires an explicit final remote publication.

### Official entry point

```text
wrapper: kaggle/run_20m_100m.py
implementation: kaggle/run_20m_100m_data_scaling.py
pinned evidence-producing commit: 43190cb72443a2de290dc8e6f2c54f29d8dff501
wrapper commit: 3f085e57260205bf9c0f9d30873fe97c1cbc2f27
```

Operator command:

```bash
cd /kaggle/working/Small-LLM
git switch main
git pull --ff-only
python kaggle/run_20m_100m.py
```

Run the same command after every successfully published segment until the summary reports `status: completed` and `remaining_steps: 0`.

## Implementation and test state

Completed on `main`:

```text
canonical fixed 100M producer: complete
exact generic finite-profile report: complete
100M report and Drive-manifest binding: complete
microbatch-1 versus microbatch-4 gate: complete
bounded segment planning: complete
verified remote restore and W&B resume: complete
single pinned Kaggle wrapper: complete
duplicate 100M profiles removed: complete
dedicated VPS/Kaggle runbook: complete
```

Offline pure launcher tests passed for segment planning, resume arguments, microbatch acceptance/rejection, dataset-profile identity, and explicit final-publication evidence. No repository CI workflow is currently attached to these commits. T4 throughput/memory qualification and live remote behavior remain intentionally pending because they require the completed attached 100M dataset and Kaggle secrets.

## Immediate next actions

```text
1. Build 20m-100m-dataset-001 on the VPS with dataset.qualification_100m.
2. Run the literal full scan and derive qualification_plan.json.
3. Publish the unchanged completed directory as a private Kaggle Dataset.
4. Attach it to a T4 notebook and run kaggle/run_20m_100m.py.
5. Review the microbatch gate before treating the training segment as authorized by evidence.
6. Re-run the same entry point after each published segment until completion.
```

Detailed plan and commands:

```text
llm_docs/20m_100m_data_scaling_plan.md
llm_docs/20m_100m_runbook.md
```
