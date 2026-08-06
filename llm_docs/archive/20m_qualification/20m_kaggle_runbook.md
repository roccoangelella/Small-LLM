# 20M Qualification: VPS Dataset and Kaggle T4 Runbook

_Last updated: 2026-08-04_

This runbook separates dataset production from GPU qualification:

1. build and verify the fixed finite dataset on the VPS;
2. attach the completed directory as a private Kaggle Dataset;
3. run exact-commit offline and T4 evidence in the Kaggle notebook;
4. run the 20-update trainer preflight;
5. stop for threshold review before the complete one-pass segment.

GitHub Actions is not required.

## Part A — Build the finite dataset on the VPS

### 1. Prepare the exact checkout

```bash
cd /path/to/Small-LLM
git switch main
git pull --ff-only
git status --short
git rev-parse HEAD
```

The working tree must be clean.  Record the commit in the build log.

### 2. Prepare local secrets

The ignored `.env` file must contain:

```env
SMALL_LLM_GOOGLE_OAUTH_TOKEN=.secrets/google-drive-authorized-user.json
SMALL_LLM_DRIVE_FOLDER_ID=<existing-dataset-shards-folder-id>
```

The authorized-user JSON must already have passed the real Drive smoke test.

### 3. Select paths

```bash
export SMALL_LLM_REPO=/path/to/Small-LLM
export WEIGHTS_FILE=/data/climbmix-mixture-calibration/climbmix_code_free_weights.json
export DATASET_DIR=/data/small-llm-20m-qualification-001
export OPS_DIR=/data/small-llm-20m-qualification-ops
mkdir -p "$OPS_DIR/logs"
```

`DATASET_DIR` must be new for the first invocation.

### 4. Run the fail-closed qualification producer

```bash
cd "$SMALL_LLM_REPO"
set -o pipefail
uv run \
  --env-file .env \
  --with-requirements dataset/requirements-remote.txt \
  python -m dataset.qualification_20m \
  --weights-file "$WEIGHTS_FILE" \
  --output-dir "$DATASET_DIR" \
  --run-id 20m-qualification-dataset-001 \
  2>&1 | tee "$OPS_DIR/logs/dataset-build.log"
status=${PIPESTATUS[0]}
echo "$status" > "$OPS_DIR/dataset-build.exit-code"
test "$status" -eq 0
```

The wrapper fixes:

```text
10M target / 9M minimum / 11M hard maximum
context 2,048
16 sequences per block
8 MiB target shards
2M-source-token durable checkpoints
remote durability required
```

Resume an interrupted build with the same arguments plus `--resume`.

### 5. Run full local verification

```bash
set -o pipefail
uv run \
  --with-requirements dataset/requirements-remote.txt \
  python -m dataset.main verify \
  --output-dir "$DATASET_DIR" \
  --full-scan \
  2>&1 | tee "$OPS_DIR/logs/dataset-verify.log"
status=${PIPESTATUS[0]}
echo "$status" > "$OPS_DIR/dataset-verify.exit-code"
test "$status" -eq 0
```

### 6. Derive the exact trainer plan

```bash
set -o pipefail
uv run python -m dataset.qualification_20m_report \
  --dataset-dir "$DATASET_DIR" \
  --drive-manifest "$DATASET_DIR/drive_manifest.json" \
  --output "$DATASET_DIR/qualification_plan.json" \
  2>&1 | tee "$OPS_DIR/logs/qualification-plan.log"
status=${PIPESTATUS[0]}
echo "$status" > "$OPS_DIR/qualification-plan.exit-code"
test "$status" -eq 0
```

The report fails unless the completed manifest exposes the fixed profile and
every Drive shard is marked remotely durable.  It writes the exact:

- train and validation source/target tokens;
- ordered block IDs;
- shard identities;
- one-pass step count;
- warmup, stable, and decay updates;
- warmup, stable, and decay token horizons;
- validation block count;
- manifest and Drive-manifest hashes.

### 7. Attach the completed data to Kaggle

Create a **private** Kaggle Dataset containing the complete `DATASET_DIR`, at
least:

```text
manifest.json
drive_manifest.json
qualification_plan.json
train/
validation/
```

Do not alter file names, bytes, or directory layout.  The notebook will rerun a
full scan against the attached read-only copy.

## Part B — Prepare the Kaggle notebook

In notebook settings:

```text
Accelerator: NVIDIA T4
Internet: On
```

Add these Kaggle Secrets:

```text
GITHUB_TOKEN
WANDB_API_KEY
WANDB_ENTITY                 optional
HF_TOKEN
SMALL_LLM_HF_REPO_ID
```

Attach the private qualification dataset created in Part A.

### 1. Load secrets and clone without printing them

```python
import base64
import os
import pathlib
import subprocess

from kaggle_secrets import UserSecretsClient

secrets = UserSecretsClient()
for name in (
    "WANDB_API_KEY",
    "WANDB_ENTITY",
    "HF_TOKEN",
    "SMALL_LLM_HF_REPO_ID",
):
    try:
        value = secrets.get_secret(name)
    except Exception:
        value = None
    if value:
        os.environ[name] = value

github_token = secrets.get_secret("GITHUB_TOKEN")
authorization = base64.b64encode(
    f"x-access-token:{github_token}".encode("utf-8")
).decode("ascii")

repo = pathlib.Path("/kaggle/working/Small-LLM")
if not repo.exists():
    subprocess.run(
        [
            "git",
            "-c",
            f"http.extraHeader=AUTHORIZATION: basic {authorization}",
            "clone",
            "https://github.com/roccoangelella/Small-LLM.git",
            str(repo),
        ],
        check=True,
    )
```

### 2. Check out one exact commit

Replace `<FINAL_COMMIT_SHA>` only after implementation is frozen:

```python
LAUNCH_COMMIT = "<FINAL_COMMIT_SHA>"
subprocess.run(["git", "fetch", "origin", LAUNCH_COMMIT], cwd=repo, check=True)
subprocess.run(["git", "checkout", "--detach", LAUNCH_COMMIT], cwd=repo, check=True)
actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
assert actual == LAUNCH_COMMIT, (actual, LAUNCH_COMMIT)
print("Exact launch commit:", actual)
```

### 3. Install `uv` and Python 3.13

```bash
python -m pip install -q uv
cd /kaggle/working/Small-LLM
uv python install 3.13
uv run --python 3.13 python --version
```

## Part C — Exact-commit evidence gates

### 1. Complete offline suite

```bash
cd /kaggle/working/Small-LLM
set -o pipefail
uv run \
  --python 3.13 \
  --extra model \
  --with wandb==0.26.1 \
  --with-requirements dataset/requirements-remote.txt \
  python -m unittest discover -v \
  2>&1 | tee /kaggle/working/offline-tests.log
status=${PIPESTATUS[0]}
echo "$status" > /kaggle/working/offline-tests.exit-code
test "$status" -eq 0
```

Do not continue on failure.

### 2. Corrected T4 harness

```bash
cd /kaggle/working/Small-LLM
set -o pipefail
uv run --python 3.13 --extra model python -m tests.t4_qualification \
  --require-t4 \
  --chunk-sizes 16 32 64 \
  --precisions fp32 fp16 \
  --sequence-length 2048 \
  --batch-size 1 \
  --warmup-steps 1 \
  --measure-steps 3 \
  --include-plan-b \
  --output /kaggle/working/t4_qualification.json \
  2>&1 | tee /kaggle/working/t4_qualification.log
status=${PIPESTATUS[0]}
echo "$status" > /kaggle/working/t4_qualification.exit-code
test "$status" -eq 0
```

### 3. Verify the attached dataset again

Set the mounted directory after inspecting `/kaggle/input`:

```bash
export DATASET_DIR=/kaggle/input/<private-dataset-slug>/<dataset-root>
cd /kaggle/working/Small-LLM

set -o pipefail
uv run --python 3.13 python -m dataset.main verify \
  --output-dir "$DATASET_DIR" \
  --full-scan \
  2>&1 | tee /kaggle/working/kaggle-dataset-verify.log
status=${PIPESTATUS[0]}
echo "$status" > /kaggle/working/kaggle-dataset-verify.exit-code
test "$status" -eq 0

uv run --python 3.13 python -m dataset.qualification_20m_report \
  --dataset-dir "$DATASET_DIR" \
  --drive-manifest "$DATASET_DIR/drive_manifest.json" \
  --output /kaggle/working/qualification_plan.json
```

## Part D — 20-update constant-LR trainer preflight

This is the first actual integrated trainer run.  It is not the complete one-pass
segment.

```python
import json
import os
import pathlib
import subprocess

repo = pathlib.Path("/kaggle/working/Small-LLM")
dataset_dir = pathlib.Path(os.environ["DATASET_DIR"])
plan = json.loads(
    pathlib.Path("/kaggle/working/qualification_plan.json").read_text()
)
validation_blocks = int(plan["trainer"]["validation_blocks"])

command = [
    "uv", "run", "--python", "3.13", "--extra", "model",
    "--with", "wandb==0.26.1",
    "--with-requirements", "dataset/requirements-remote.txt",
    "python", "-m", "trainer",
    "--dataset-dir", str(dataset_dir),
    "--dataset-manifest", str(dataset_dir / "manifest.json"),
    "--checkpoint-dir", "/kaggle/working/checkpoints-preflight",
    "--steps", "20",
    "--sequences-per-block", "16",
    "--model-size", "smoke",
    "--architecture", "gdn2_hybrid",
    "--gdn-chunk-size", "32",
    "--initialization", "normal",
    "--optimizer", "hybrid_muon_adamw",
    "--device", "cuda",
    "--precision", "fp16",
    "--microbatch-size", "1",
    "--learning-rate", "3e-4",
    "--weight-decay", "0.1",
    "--muon-momentum", "0.95",
    "--muon-lr-multiplier", "1.0",
    "--muon-update-rms", "0.18",
    "--muon-weight-decay", "0.1",
    "--max-grad-norm", "1.0",
    "--schedule", "constant",
    "--evaluation-every-steps", "20",
    "--validation-blocks", str(validation_blocks),
    "--seed", "17",
    "--wandb-mode", "online",
    "--wandb-project", "Small-LLM",
    "--wandb-run-id", "20m-t4-preflight-001",
    "--wandb-run-name", "20M T4 preflight",
    "--wandb-tags", "20m", "t4", "preflight",
]

with open("/kaggle/working/trainer-preflight.log", "w", encoding="utf-8") as log:
    result = subprocess.run(
        command,
        cwd=repo,
        stdout=log,
        stderr=subprocess.STDOUT,
        text=True,
    )
pathlib.Path("/kaggle/working/trainer-preflight.exit-code").write_text(
    f"{result.returncode}\n"
)
assert result.returncode == 0
```

Review both the saved log and W&B.  Required observations include finite loss,
no exhausted overflow retries, scaler behavior, clipping frequency, Muon and
AdamW gradient norms, actual update RMS/weight ratios, per-Muon-matrix update
statistics, memory, data wait, validation, and final checkpoint timing.

## Stop gate after the preflight

Do **not** start the full one-pass run immediately.  First:

1. run the same-hardware uninterrupted/A-A controls;
2. derive and commit empirical thresholds;
3. qualify local interruption/resume;
4. qualify remote publication and two-shard empty-environment restore.

## Part E — Complete one-pass command construction

After those gates pass, construct the command from the report rather than
copying approximate numbers:

```python
import json
import os
import pathlib
import subprocess

repo = pathlib.Path("/kaggle/working/Small-LLM")
dataset_dir = pathlib.Path(os.environ["DATASET_DIR"])
plan = json.loads(pathlib.Path("/kaggle/working/qualification_plan.json").read_text())
trainer_plan = plan["trainer"]

command = [
    "uv", "run", "--python", "3.13", "--extra", "model",
    "--with", "wandb==0.26.1",
    "--with-requirements", "dataset/requirements-remote.txt",
    "python", "-m", "trainer",
    "--dataset-dir", str(dataset_dir),
    "--dataset-manifest", str(dataset_dir / "manifest.json"),
    "--checkpoint-dir", "/kaggle/working/checkpoints-one-pass",
    "--steps", str(trainer_plan["steps"]),
    "--sequences-per-block", "16",
    "--model-size", "smoke",
    "--architecture", "gdn2_hybrid",
    "--gdn-chunk-size", "32",
    "--initialization", "normal",
    "--optimizer", "hybrid_muon_adamw",
    "--device", "cuda",
    "--precision", "fp16",
    "--microbatch-size", "1",
    "--learning-rate", "3e-4",
    "--weight-decay", "0.1",
    "--muon-momentum", "0.95",
    "--muon-lr-multiplier", "1.0",
    "--muon-update-rms", "0.18",
    "--muon-weight-decay", "0.1",
    "--max-grad-norm", "1.0",
    "--schedule", "wsd",
    "--warmup-tokens", str(trainer_plan["warmup_tokens"]),
    "--stable-tokens", str(trainer_plan["stable_tokens"]),
    "--decay-tokens", str(trainer_plan["decay_tokens"]),
    "--minimum-lr-ratio", "0.1",
    "--checkpoint-every-steps", "25",
    "--evaluation-every-steps", "50",
    "--validation-blocks", str(trainer_plan["validation_blocks"]),
    "--remote-publish-every-steps", "50",
    "--remote-drive-manifest", str(dataset_dir / "drive_manifest.json"),
    "--remote-token-env", "HF_TOKEN",
    "--seed", "17",
    "--wandb-mode", "online",
    "--wandb-project", "Small-LLM",
    "--wandb-run-id", "20m-one-pass-001",
    "--wandb-run-name", "20M one-pass qualification",
    "--wandb-tags", "20m", "t4", "qualification", "one-pass",
]

# Add --remote-create-repo only when the configured private repository does not
# already exist.  Execute only after the preflight/recovery threshold gate.
print("Prepared exact one-pass steps:", trainer_plan["steps"])
print("Prepared exact token horizons:", {
    key: trainer_plan[key]
    for key in ("warmup_tokens", "stable_tokens", "decay_tokens")
})
```

The trainer now runs final validation whenever the final step is not already an
evaluation boundary, so a typical approximately-305-step run validates and
checkpoints the actual final model rather than stopping with only step-300
validation evidence.
