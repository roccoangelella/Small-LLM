# Project Status

_Last updated: 2026-08-05 11:50 Europe/Rome_

## Current phase

The frozen approximately-20M engineering qualification ladder is complete, the full run was authorized, and the user has now confirmed that the complete Kaggle run is live and has reached at least optimizer step 150.

```text
status: running
qualification: passed_remote_empty_environment_recovery
authorization: full_306_run_authorized
execution venue: Kaggle
accelerator target: NVIDIA Tesla T4
run state: running_user_reported_step_150
W&B run ID: 20m-one-pass-001
latest reported optimizer step: 150 / 306
latest reported validation perplexity: approximately 576
corresponding validation loss: approximately 6.356108
reported at: 2026-08-05 11:50 Europe/Rome
```

The live-state record above is based on the user's direct observation of the active run. Completion, final checkpoint publication, and the final metric set remain unverified until the run terminates successfully and its artifacts are inspected.

The authorization was given at 2026-08-05 10:04 Europe/Rome. Architecture selection and the frozen recipe are not being reopened.

Detailed evidence and the launch decision are recorded in:

```text
llm_docs/20m_kaggle_preflight_results.md
llm_docs/20m_repeatability_results.md
llm_docs/20m_local_resume_results.md
llm_docs/20m_remote_recovery_results.md
llm_docs/20m_kaggle_launch_authorization.md
```

## Live complete-run observations

The user reported the following validation trajectory from the active 306-update one-pass run:

```text
step 50 validation perplexity: approximately 2,739.35
step 50 validation loss: approximately 7.915476
step 150 validation perplexity: approximately 576
step 150 validation loss: approximately 6.356108
perplexity reduction from step 50 to 150: approximately 4.756x
validation-loss reduction from step 50 to 150: approximately 1.559 nats
nominal full-block target tokens committed by step 150: 4,915,200
fraction of planned train target tokens: approximately 49.12%
```

This is strong midpoint evidence of continued learning on the fixed validation block. It is not yet a final model-quality result because the validation set is very small and the current metric includes every stored target position, including any synthetic padding positions in the final partial validation sequence.

The run's configured persistence cadence remains:

```text
local joint checkpoint: every 25 successful optimizer updates
validation: every 50 successful optimizer updates
remote Hugging Face publication: every 50 successful optimizer updates
```

No statement about the success of the live run's step-50, step-100, or step-150 remote publications is recorded here without direct artifact or log verification.

## Frozen model and optimizer

```text
parameters: 20,637,592
architecture: [GDN-2, GDN-2, GDN-2, full gated MHA] repeated
context length: 2,048
precision: FP16
GDN-2 backend: ordinary PyTorch chunkwise
GDN-2 chunk size: 32
initialization: normal
seed: 17
optimizer: hybrid whole-matrix Muon + AdamW
base LR: 3e-4
AdamW betas: 0.9 / 0.95
AdamW epsilon: 1e-8
weight decay: 0.1
Muon momentum: 0.95
Muon LR multiplier: 1.0
Muon target direction RMS: 0.18
Muon weight decay: 0.1
global gradient clipping: 1.0
```

The user accepted universal clipping for this frozen 20M qualification because it was bounded, exactly repeatable, accompanied by improving loss, and produced no FP16 overflows. This does not transfer automatically to later models or authorize silent hyperparameter changes.

## Accepted qualification dataset

```text
run ID: 20m-qualification-dataset-001
accepted source tokens: 10,000,662
train source tokens: 9,991,872
validation source tokens: 8,790
train shards: 6
validation shards: 1
train sequences: 4,886
validation sequences: 5
train blocks / one-pass optimizer updates: 306
stored uint16 tokens: 10,021,659
manifest SHA-256: 1e5ee8f372b77b6728288610dbe7cce74d833be21e53d1538bc5a890229b18bb
Drive manifest SHA-256: fbb29ee0d0102658e1274e39d6647cf56a6dcb685e0f566b1736847dcc4fbe84
```

The mounted dataset repeatedly passed literal full scans and exact one-pass-plan reproduction.

## Exact one-pass plan

```text
schedule: WSD
passes: 1
steps: 306
full-block target tokens: 32,768
warmup: 16 updates / 524,288 target tokens
stable: 228 updates / 7,471,104 target tokens
decay: 62 updates / 2,011,136 target tokens
minimum LR ratio: 0.1
validation blocks: 1
train target tokens: 10,006,528
```

The final training block is partial. Silent data wraparound remains forbidden.

## Frozen launch identity

```text
launch commit: 45d1da4a1ac3f18cf6ce02b8439672f10e2c8b4c
GPU qualification target: NVIDIA Tesla T4
selected execution venue: Kaggle
```

Evidence-producing trainer work used clean detached worktrees at the frozen launch commit.

## Passed qualification gates

```text
offline suite: 229 passed, 1 expected live-remote skip
corrected T4 harness: passed
dataset full scan and exact plan reproduction: passed
20-update integrated trainer preflight: passed
50-update uninterrupted reference: passed
50-update same-T4 A/A repeatability: passed
actual-process SIGTERM at update 25: passed
fresh-process local resume through update 50: passed
private Hugging Face checkpoint publication: passed
empty-environment checkpoint restore: passed
two-shard Google Drive prefetch and verification: passed
remote-restored continuation through update 30: passed
```

## Repeatability result

Two independent 50-update runs produced exact non-runtime telemetry and validation equality:

```text
compared numerical values: 10,650
differing numerical values: 0
maximum absolute difference: 0.0
maximum relative difference: 0.0
discrete trajectory exact: true
validation exact: true
training loss: 10.845867 -> 8.090633
validation loss: 7.915478
GradScaler: 65,536 throughout
FP16 overflow events / retries: 0 / 0
gradient clipping: 50 / 50 updates
```

The first-10 and last-10 median pre-clip norms were `1.399718` and `1.385033`; the final norm was `1.344303`.

## Local interruption and resume result

The actual trainer process group was terminated after the complete step-25 checkpoint:

```text
signal: SIGTERM
exit code: 143
forced SIGKILL: false
process group gone: true
last consumed block: 24
next resumed block: 25
```

A fresh process resumed updates 26-50 and matched the uninterrupted reference exactly:

```text
compared numerical values: 10,650
differing numerical values: 0
numeric trajectory exact: true
discrete trajectory exact: true
validation exact: true
resume class: exact_local_resume
```

Decoded semantic checkpoint state was exact at steps 25 and 50:

```text
tensors compared per checkpoint: 383
tensor elements compared: 54,184,616
semantic differences: 0
```

Raw checkpoint-tree differences with exact decoded state are accepted as serialization-byte variability rather than training-state divergence.

## Remote empty-environment recovery result

The bounded final gate executed 35 updates total:

```text
publisher segment: updates 1-25
local reference continuation: updates 26-30
remote-restored continuation: updates 26-30
```

Final result:

```text
status: passed_remote_empty_environment_recovery
authorization: full_306_run_ready_for_explicit_launch
resume class: exact_remote_empty_environment_recovery
evidence: /kaggle/working/small-llm-remote-recovery-controller/small-llm-remote-recovery-20260805T073059Z
```

The source step-25 checkpoint was published privately, restored into an empty destination, and verified. The restored state pointed to block 25 as the next block. Exactly two required train shards were downloaded from Google Drive and matched their immutable byte sizes and SHA-256 identities.

Source versus remote step-25 semantic comparison:

```text
tensors: 383
tensor elements: 54,184,616
scalars: 1,112
differences: 0
semantic exact: true
```

Local versus remote-restored updates 26-30:

```text
compared numerical values: 1,065
differing numerical values: 0
maximum absolute difference: 0.0
maximum relative difference: 0.0
numeric trajectory exact: true
discrete trajectory exact: true
```

Local versus remote step-30 semantic comparison also had zero differences across the same 383 tensors and 54,184,616 tensor elements.

W&B run identities:

```text
publisher: 20m-t4-remote-20260805-073105-publisher
local reference: 20m-t4-remote-20260805-073105-local-reference
remote restored: 20m-t4-remote-20260805-073105-remote-restored
project: Small-LLM
entity observed in Kaggle logs: rocchissimo936-none
```

## Current readiness verdict

**Dataset gate: passed.**

**Offline, T4, integrated trainer, and numerical-stability gates: passed.**

**Same-T4 repeatability: passed with exact trajectory equality.**

**Actual-process local interruption and resume: passed with exact trajectory and semantic state.**

**Private remote publication and empty-environment recovery: passed with exact trajectory and semantic state.**

**Complete run state: running.** The user has confirmed progress through step 150 of 306 with validation perplexity approximately 576.

**Not yet completed:** final validation, final local checkpoint, final remote publication, W&B finalization, and complete-run acceptance still require direct artifact and log verification.