#!/usr/bin/env python3
"""Build 100M/10B data on the VPS and mirror READY shards to Beam.

Usage (run from the repository root, preferably inside tmux):

    python beam/vps_dataset_producer.py --model 100M --tokens 10B

Hugging Face remains the authoritative durable dataset store/frontier. Before a
newly durable shard is published READY, this wrapper copies the same verified
local shard into the fixed Beam cache Volume used by training. Beam therefore
does not need to pay a CPU worker to build the corpus.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections.abc import Mapping
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
BEAM_DIR = Path(__file__).resolve().parent
sys.path[:0] = [str(ROOT), str(BEAM_DIR)]

from profiles import resolve_presets  # noqa: E402
from rolling_producer import APPROVED_WEIGHTS_SHA256  # noqa: E402
from dataset import config  # noqa: E402
from dataset.production.cli import main as production_main  # noqa: E402
from dataset.qualification import get_profile, production_arguments  # noqa: E402
from dataset.src.remote import sha256_path  # noqa: E402
from dataset.src.storage import read_json, write_json_atomic  # noqa: E402

BEAM_VOLUME = "small-llm-cache"
DEFAULT_ROOT = Path.home() / ".cache" / "small-llm" / "beam-producer"


def _logical_name(value: object) -> str:
    if not isinstance(value, str) or not value or "\\" in value:
        raise RuntimeError(f"invalid shard filename: {value!r}")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise RuntimeError(f"invalid shard filename: {value!r}")
    return path.as_posix()


def _mirror_hook(output: Path, run_id: str):
    state_path = output.parent / f"{run_id}.beam-mirror.json"

    def mirror() -> None:
        progress = read_json(output / config.PROGRESS_FILENAME)
        if not isinstance(progress, Mapping) or not isinstance(progress.get("finalized_shards"), list):
            raise RuntimeError("producer progress is unavailable at the Beam mirror boundary")
        if state_path.is_file():
            raw_state = read_json(state_path)
            if not isinstance(raw_state, Mapping) or raw_state.get("run_id") != run_id:
                raise RuntimeError("Beam mirror state belongs to another run")
            synced = dict(raw_state.get("shards", {}))
        else:
            synced = {}

        for row in progress["finalized_shards"]:
            if not isinstance(row, Mapping):
                raise RuntimeError("finalized shard metadata is malformed")
            name = _logical_name(row.get("filename"))
            checksum = row.get("checksum", row.get("local_sha256"))
            size = row.get("byte_size")
            if not isinstance(checksum, str) or len(checksum) != 64:
                raise RuntimeError(f"finalized shard has no SHA-256: {name}")
            if isinstance(size, bool) or not isinstance(size, int) or size <= 0:
                raise RuntimeError(f"finalized shard has invalid size: {name}")
            previous = synced.get(name)
            identity = {"sha256": checksum, "byte_size": size}
            if previous is not None:
                if previous != identity:
                    raise RuntimeError(f"immutable Beam mirror identity changed: {name}")
                continue

            local = output / name
            if local.is_symlink() or not local.is_file():
                raise RuntimeError(f"local finalized shard disappeared before Beam mirror: {name}")
            if local.stat().st_size != size or sha256_path(local) != checksum:
                raise RuntimeError(f"local finalized shard failed verification: {name}")
            destination = f"beam://{BEAM_VOLUME}/datasets/{run_id}/{name}"
            subprocess.run(["beam", "cp", str(local), destination], check=True)
            synced[name] = identity
            write_json_atomic(
                state_path,
                {"version": 1, "run_id": run_id, "beam_volume": BEAM_VOLUME, "shards": synced},
            )
            print(
                json.dumps(
                    {"beam_vps_dataset_feed": "shard_mirrored", "shard": name, "bytes": size},
                    sort_keys=True,
                ),
                flush=True,
            )

    return mirror


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--reader-workers", type=int)
    args = parser.parse_args()

    _, token_preset = resolve_presets(args.model, args.tokens)
    profile = get_profile(token_preset.dataset_profile)
    if not profile.incremental_frontier or profile.run_id is None:
        raise RuntimeError("VPS-fed Beam production requires an incremental dataset profile")
    if not os.environ.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for authoritative HF dataset durability")

    weights = ROOT / "dataset" / "climbmix_code_free_weights.json"
    if not weights.is_file() or sha256_path(weights) != APPROVED_WEIGHTS_SHA256:
        raise RuntimeError("vendored ClimbMix weights are missing or invalid")

    output = args.output_root.expanduser().resolve() / profile.run_id
    output.parent.mkdir(parents=True, exist_ok=True)
    producer_args = ["--weights-file", str(weights), "--output-dir", str(output)]
    if (output / config.WORK_PLAN_FILENAME).is_file():
        producer_args.append("--resume")
    if args.reader_workers is not None:
        if args.reader_workers <= 0:
            raise ValueError("--reader-workers must be positive")
        producer_args.extend(["--reader-workers", str(args.reader_workers)])

    print(
        json.dumps(
            {
                "beam_vps_dataset_feed": "start",
                "run_id": profile.run_id,
                "local_output": str(output),
                "beam_destination": f"beam://{BEAM_VOLUME}/datasets/{profile.run_id}",
                "resume": "--resume" in producer_args,
            },
            indent=2,
            sort_keys=True,
        ),
        flush=True,
    )
    code = production_main(
        production_arguments(profile, producer_args),
        durable_progress_hook=_mirror_hook(output, profile.run_id),
    )
    if code:
        raise RuntimeError(f"VPS dataset producer exited with status {code}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
