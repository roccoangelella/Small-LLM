# Beam training

`beam/launch.py` is the Beam counterpart of `modal/launch.py`. It keeps the same Small-LLM scientific and checkpoint contract while binding execution to Beam Functions, GPUs, secrets, and distributed Volumes.

Beam is a Python SDK plus CLI and is included by the project's canonical `uv sync` environment. Run the launcher from the repository root so Beam syncs the complete checkout.

## Setup

```bash
cd ~/Projects/Small-LLM
uv sync
source .venv/bin/activate
beam configure default --token YOUR_BEAM_TOKEN
```

Create Beam secrets named `WANDB_API_KEY`, `HF_TOKEN`, and `SMALL_LLM_HF_REPO_ID` using the values already used by the project. The launcher injects those three names into training containers.

The repository `.beamignore` excludes `.env` and `.secrets/` from Beam source sync. Treat an `Added .../.env` or `Added .../.secrets/...` line during sync as a failed security preflight and stop before continuing.

Volumes are created automatically when referenced. They may also be created explicitly:

```bash
beam volume create small-llm-data
beam volume create small-llm-runs
beam volume create small-llm-cache
beam machine list
```

The adapter deliberately allows only the Beam serverless lane recorded by the current project decision: `RTX5090`, `RTX4090`, and `A10G`. `RTX5090` is the default. `H100` is intentionally rejected by the Beam launcher so Beam credits cannot accidentally spill into an on-demand H100; use the existing Modal lane when H100 is the intended comparison.

## Dry run and scientific launch gate

The Beam adapter can be resolved without allocating a GPU:

```bash
python beam/launch.py --model 100M --tokens 10B --gpu RTX5090 --dry-run
```

The canonical `100m-10b-data-001` trajectory is still behind ADR 0050's behavioral launch gate. **Do not use `--max-steps-this-session` as a GPU smoke test before that gate closes**: even a short segment would create/resume the canonical checkpoint and W&B identity.

Once the scientific gate is explicitly closed, launch the real trajectory with:

```bash
python beam/launch.py --model 100M --tokens 10B --gpu RTX5090
```

A fresh Beam trajectory probes real forward/backward execution at microbatch `8, 12, 16`. OOM, non-finite, and >90%-reserved-memory candidates are rejected; the fastest safe candidate is frozen. The optimizer block remains exactly 64 sequences.

Under ADR 0065, the first authorized real RTX5090 allocation is the compatibility qualification and continues into training when the startup probe succeeds. No separate paid GPU smoke is required solely for Beam compatibility.

## 10B CPU-first allocation order

The incremental path is:

```text
headless Beam CPU producer -> immutable HF READY frontier
                               |
Beam CPU stager ---------------+
        |
        v
current + successor shard downloaded and SHA256 verified
        |
        v
fresh Beam CPU container verifies distributed-volume visibility
        |
        v
only now allocate RTX5090 / RTX4090 / A10G
```

The producer is a `headless=True` Beam Function and may continue producing while training consumes earlier shards. Beam documents that distributed-volume writes can take up to roughly 60 seconds to appear in another container, so the launcher never uses a local volume write as the readiness signal. The HF READY frontier remains authoritative and a fresh CPU visibility check is mandatory before GPU dispatch.

## Checkpoints and resume

Beam Volumes are local acceleration/durability layers. Cross-provider exact resume still uses the unified Hugging Face model repository:

```text
run/<run-id>/latest.json          live exact-resume pointer
models/<run-id>/<checkpoint-id>  stable completed model artifact
```

The historical transport schema name `modal-hf-checkpoint-v1` is retained deliberately as a compatibility identifier so Beam can read existing Modal checkpoints without a second namespace. W&B keeps the canonical run ID and adds a `beam` provider tag.

## Existing finite 2B corpus

The 10B lane streams HF shards and does not upload a complete dataset to Beam. If a finite 2B run is needed, create or reuse the existing block-64 derivative locally and copy it to the Beam data Volume:

```bash
python modal/prepare_dataset.py --no-upload
beam cp ~/small-llm-data/modal-2b-b64-dataset-001 beam://small-llm-data/modal-2b-b64-dataset-001
```

The frozen profile name `modal-2b-b64` is retained for dataset identity compatibility; it does not force execution on Modal.

## Runtime

The Beam adapter has two GPU image lanes. RTX5090 uses CUDA 12.8.1 plus the official PyTorch 2.10 `cu128` wheel because Blackwell first gains CUDA toolkit support in CUDA 12.8. RTX4090/A10G use the CUDA 12.4.1 host family plus PyTorch 2.10 `cu126`. Both keep `fla-core==0.5.2`. Triton caches are keyed by compute capability, so Blackwell, Ada, and Ampere compile/cache independently. The first paid RTX5090 segment is still a live compatibility qualification for FLA/Triton; the image choice does not assume the kernel will pass.

The scientific contract remains FP16 autocast with FP32 master parameters, GDN-2 hybrid, context 2048, block 64, hybrid Muon+AdamW, manifest-derived WSD, seed 17, checkpoint/evaluation cadence 250, and exact data-cursor resume.
