# Project Status

_Last updated: 2026-08-04_

## Current phase

The finite approximately-20M engineering qualification dataset is built, durably mirrored, fully scanned, and accepted. The exact-commit Kaggle T4 gates, integrated preflight, same-T4 repeatability test, and actual-process local interruption/resume test have passed.

The project is now in **final remote empty-environment recovery qualification**.

Architecture selection is not being reopened. The complete 306-update one-pass segment remains unauthorized until the remote recovery test passes and the user explicitly authorizes launch. The approximately-100M architecture comparison and complete 90B dataset build also remain unauthorized until the 20M ladder is complete.

Detailed evidence is recorded in:

```text
llm_docs/20m_kaggle_preflight_results.md
llm_docs/20m_repeatability_results.md
llm_docs/20m_local_resume_results.md
llm_docs/20m_remote_recovery_test.md
```

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
weight decay: 0.1
Muon momentum: 0.95
Muon target direction RMS: 0.18
global gradient clipping: 1.0
```

The user accepted the observed universal clipping for this frozen 20M engineering qualification. The decision is based on bounded and exactly repeatable gradient norms, decreasing loss, stable optimizer telemetry, and zero FP16 overflows. It does not silently authorize a different LR, clipping threshold, optimizer, or transfer of the decision to later models.

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

The mounted Kaggle dataset has repeatedly passed literal full scans and exact 306-update plan reproduction.

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

## Passed gates

All evidence-producing trainer work has used a clean detached worktree at:

```text
launch commit: 45d1da4a1ac3f18cf6ce02b8439672f10e2c8b4c
GPU: NVIDIA Tesla T4
```

Passed:

```text
offline suite: 229 passed, 1 expected live-remote skip
corrected T4 harness: passed
dataset full scan and plan reproduction: passed
20-update integrated trainer preflight: passed
50-update uninterrupted reference: passed
50-update same-T4 A/A repeat: passed
actual-process SIGTERM at update 25: passed
fresh-process local resume through update 50: passed
```

## Repeatability result

The two independent 50-update runs produced exact non-runtime telemetry and validation equality:

```text
compared numerical values: 10,650
differing numerical values: 0
maximum absolute difference: 0.0
maximum relative difference: 0.0
discrete trajectory exact: true
validation exact: true
```

Both produced:

```text
training loss: 10.845867 -> 8.090633
validation loss: 7.915478
GradScaler: 65,536 throughout
FP16 overflow events / retries: 0 / 0
gradient clipping: 50 / 50 updates
```

The first-10 and last-10 median pre-clip norms were `1.399718` and `1.385033`; the final norm was `1.344303`. Universal clipping was therefore accepted as bounded for this qualification.

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

A fresh process resumed updates 26-50. The combined path matched the uninterrupted reference exactly:

```text
compared numerical values: 10,650
differing numerical values: 0
numeric trajectory exact: true
discrete trajectory exact: true
validation exact: true
resume class: exact_local_resume
```

Decoded checkpoint semantics were exact at both step 25 and step 50:

```text
tensors compared per checkpoint: 383
tensor elements compared: 54,184,616
semantic differences: 0
```

Raw checkpoint-tree hashes differed while decoded state was exact. This is accepted as serialization-byte variability rather than state divergence.

Local-resume evidence:

```text
status: passed_local_interruption_resume
authorization: remote_recovery_only
evidence directory: /kaggle/working/small-llm-local-resume-controller/small-llm-local-resume-20260804T162921Z
summary: /kaggle/working/small_llm_local_resume_summary.json
```

## Final remaining gate

The repository-native launcher is:

```text
kaggle/run_20m_remote_recovery_from_clone.py
```

The remote test is deliberately shorter because 50-step stability and exact local recovery are already established:

```text
publisher training: updates 1-25
local continuation: updates 26-30
remote-restored continuation: updates 26-30
total executed updates: 35
```

It must prove:

1. verified two-phase private Hugging Face publication of `step-00000025`;
2. restore into a destination with no prior checkpoint or data cache;
3. exact checkpoint-manifest and semantic-state verification;
4. exact two-shard Google Drive prefetch by immutable file ID, size, and SHA-256;
5. restored next block equal to 25;
6. exact local-versus-remote continuation for updates 26-30;
7. exact semantic step-30 checkpoint equality.

Required Kaggle secrets:

```text
WANDB_API_KEY
HF_TOKEN
SMALL_LLM_HF_REPO_ID
GOOGLE_DRIVE_OAUTH_TOKEN_JSON
```

Optional: `WANDB_ENTITY`.

A successful result must report:

```text
status: passed_remote_empty_environment_recovery
authorization: full_306_run_ready_for_explicit_launch
```

## Current readiness verdict

**Dataset gate: passed.**

**Exact-commit offline/T4/integrated preflight: passed.**

**Same-T4 50-update repeatability: passed with exact trajectory equality.**

**Actual-process local interruption and resume: passed with exact trajectory and semantic checkpoint equality.**

**Ready next:** run the bounded private remote publication and empty-environment recovery test.

**Not yet authorized:** the complete 306-update one-pass segment, because the final remote recovery gate has not yet passed.