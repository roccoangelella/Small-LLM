#!/usr/bin/env python3
"""Modal CPU producer for the accepted 100B SuperBPE MoE corpus."""

from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset.moe_100b import RUN_ID


def _load_existing_launcher():
    source = _REPO_ROOT / "modal" / "launch.py"
    spec = importlib.util.spec_from_file_location("small_llm_modal_base_launch", source)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load canonical Modal launcher: {source}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_base = _load_existing_launcher()
_modal = _base.modal
DATASET_IMAGE = _base.IMAGE.uv_pip_install(
    "tiktoken==0.14.0",
    "tokenizers==0.23.2",
)
app = _modal.App("small-llm-moe-100b-superbpe-dataset", image=DATASET_IMAGE)

CACHE_ROOT = _base.CACHE_ROOT
CACHE_VOLUME = _base.CACHE_VOLUME
TRAINING_SECRET = _base.TRAINING_SECRET
REMOTE_REPO = _base.REMOTE_REPO


def _hf_bucket_identity(explicit_bucket_id: str = "") -> tuple[str, str]:
    token = os.environ.get("HF_TOKEN", "").strip()
    if not token:
        raise RuntimeError("HF_TOKEN is required for the 100B dataset producer")
    explicit = explicit_bucket_id.strip() or os.environ.get(
        "SMALL_LLM_HF_DATASET_BUCKET_ID", ""
    ).strip()
    if explicit:
        return explicit, token
    repo_id = os.environ.get("SMALL_LLM_HF_REPO_ID", "").strip()
    if not repo_id:
        raise RuntimeError(
            "SMALL_LLM_HF_DATASET_BUCKET_ID or SMALL_LLM_HF_REPO_ID is required"
        )
    return f"{repo_id}-datasets", token


@app.function(
    cpu=8,
    memory=32768,
    timeout=24 * 60 * 60,
    retries=_modal.Retries(max_retries=10, initial_delay=1.0),
    single_use_containers=True,
    secrets=[TRAINING_SECRET],
    volumes={str(CACHE_ROOT): CACHE_VOLUME},
)
def produce_superbpe_100b(
    dataset_bucket_id: str,
    reader_workers: int = 8,
    max_in_flight_work_items: int = 32,
) -> dict[str, object]:
    """Resume or build the frozen 100B corpus and publish READY shards to HF."""

    sys.path.insert(0, str(REMOTE_REPO))
    from dataset.incremental_frontier import SHARD_FRONTIER_FILENAME
    from dataset.moe_100b import main as production_main
    from dataset.src.hf_bucket_shards import HuggingFaceBucketShardStore

    output = CACHE_ROOT / "producer" / RUN_ID
    output.mkdir(parents=True, exist_ok=True)
    weights = REMOTE_REPO / "dataset" / "climbmix_code_free_weights.json"
    if not weights.is_file():
        raise RuntimeError(f"missing production weights: {weights}")

    args = [
        "--weights-file", str(weights),
        "--output-dir", str(output),
        "--reader-workers", str(reader_workers),
        "--max-in-flight-work-items", str(max_in_flight_work_items),
    ]
    if (output / "work_plan.json").is_file():
        args.append("--resume")

    code = production_main(args, durable_progress_hook=CACHE_VOLUME.commit)
    if code:
        raise RuntimeError(f"100B SuperBPE producer exited with status {code}")
    CACHE_VOLUME.commit()

    bucket_id, token = _hf_bucket_identity(dataset_bucket_id)
    store = HuggingFaceBucketShardStore(
        bucket_id,
        token=token,
        private=False,
        create_bucket=False,
    )
    store.verify_bucket_visibility()
    frontier = store._read_json(store.object_key(RUN_ID, SHARD_FRONTIER_FILENAME))
    if not isinstance(frontier, dict) or frontier.get("producer_complete") is not True:
        raise RuntimeError("producer returned without a completed remote READY frontier")
    return {
        "status": "complete",
        "run_id": RUN_ID,
        "dataset_bucket_id": bucket_id,
        "producer_complete": True,
        "planned_train_blocks": frontier.get("planned_train_blocks"),
        "last_ready_train_block_id": frontier.get("last_ready_train_block_id"),
        "local_producer_dir": str(output),
    }


@app.local_entrypoint()
def main(
    dataset_bucket_id: str = "",
    reader_workers: int = 8,
    max_in_flight_work_items: int = 32,
    dry_run: bool = False,
) -> None:
    if reader_workers <= 0 or max_in_flight_work_items <= 0:
        raise ValueError("reader worker counts must be positive")
    resolved_bucket_id, _ = _hf_bucket_identity(dataset_bucket_id)
    source_commit = _base._local_source_commit()
    payload = {
        "run_id": RUN_ID,
        "source_commit": source_commit,
        "provider": "modal",
        "dataset_bucket_id": resolved_bucket_id,
        "work": "CPU GPT-2 decode -> SuperBPE encode -> stratify -> schema-v2 -> HF READY",
        "reader_workers": reader_workers,
        "max_in_flight_work_items": max_in_flight_work_items,
        "resume": "automatic from durable work_plan/progress",
    }
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)
    if dry_run:
        return
    result = produce_superbpe_100b.remote(
        resolved_bucket_id,
        reader_workers,
        max_in_flight_work_items,
    )
    print(json.dumps(result, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    raise SystemExit("use `modal run modal/moe_100b_dataset.py ...`")
