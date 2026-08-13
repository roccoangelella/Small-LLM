#!/usr/bin/env python3
"""Launch Beam training while dataset production runs on the VPS.

Run the producer in a persistent VPS session first:

    python beam/vps_dataset_producer.py --model 100M --tokens 10B

Then launch training with the ordinary Beam arguments:

    python beam/vps_train.py --model 100M --tokens 10B --gpu RTX5090

This wrapper deliberately replaces the incremental Beam CPU producer with the
external VPS feed. The GPU-side trainer also requires dataset shard bytes to be
preseeded in the Beam Volume; HF remains frontier/durability metadata only and
cannot silently become a dataset-byte fallback on this path.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from beam import function  # noqa: E402
from beam import launch as base  # noqa: E402


def _stage_from_vps_feed(model: str, tokens: str) -> tuple[dict[str, object], object]:
    staged = base._require_remote_mapping(
        base.stage_rolling_dataset_remote.remote(model, tokens),
        label="VPS-fed dataset stage",
    )
    return staged, {
        "status": "external_vps_beam_volume_feed",
        "beam_cpu_dataset_producer_allocated": False,
    }


def _install_preseed_guard() -> None:
    """Make trainer subprocesses consume dataset bytes only from the Beam Volume."""

    site_dir = ROOT / "beam" / "vps_site"
    if not (site_dir / "sitecustomize.py").is_file():
        raise RuntimeError("VPS-fed trainer preseed guard was not synced")
    current = [value for value in os.environ.get("PYTHONPATH", "").split(os.pathsep) if value]
    if str(site_dir) not in current:
        current.insert(0, str(site_dir))
    os.environ["PYTHONPATH"] = os.pathsep.join(current)
    os.environ["SMALL_LLM_DATASET_REQUIRE_PRESEEDED"] = "1"
    # Beam documents up to roughly 60 seconds for distributed-Volume
    # propagation. Double that bound before failing rather than downloading the
    # same dataset shard again from HF while a GPU is allocated.
    os.environ["SMALL_LLM_DATASET_PRESEED_WAIT_SECONDS"] = "120"


def _train_vps_impl(
    model: str,
    tokens: str,
    source_commit: str,
    dataset_dir: str,
    max_steps_this_session: int,
    microbatch_size: int,
    precision: str,
) -> dict[str, object]:
    _install_preseed_guard()
    return base._train_impl(
        model,
        tokens,
        source_commit,
        dataset_dir,
        max_steps_this_session,
        microbatch_size,
        precision,
    )


@function(
    name="small-llm-vps-train-rtx5090",
    gpu="RTX5090",
    image=base.BLACKWELL_IMAGE,
    **base._GPU_FUNCTION_KWARGS,
)
def train_vps_rtx5090_remote(
    model: str,
    tokens: str,
    source_commit: str,
    dataset_dir: str = "",
    max_steps_this_session: int = 0,
    microbatch_size: int = 0,
    precision: str = base.DEFAULT_PRECISION,
) -> dict[str, object]:
    return _train_vps_impl(
        model, tokens, source_commit, dataset_dir, max_steps_this_session, microbatch_size, precision
    )


@function(
    name="small-llm-vps-train-rtx4090",
    gpu="RTX4090",
    image=base.LEGACY_SERVERLESS_IMAGE,
    **base._GPU_FUNCTION_KWARGS,
)
def train_vps_rtx4090_remote(
    model: str,
    tokens: str,
    source_commit: str,
    dataset_dir: str = "",
    max_steps_this_session: int = 0,
    microbatch_size: int = 0,
    precision: str = base.DEFAULT_PRECISION,
) -> dict[str, object]:
    return _train_vps_impl(
        model, tokens, source_commit, dataset_dir, max_steps_this_session, microbatch_size, precision
    )


@function(
    name="small-llm-vps-train-a10g",
    gpu="A10G",
    image=base.LEGACY_SERVERLESS_IMAGE,
    **base._GPU_FUNCTION_KWARGS,
)
def train_vps_a10g_remote(
    model: str,
    tokens: str,
    source_commit: str,
    dataset_dir: str = "",
    max_steps_this_session: int = 0,
    microbatch_size: int = 0,
    precision: str = base.DEFAULT_PRECISION,
) -> dict[str, object]:
    return _train_vps_impl(
        model, tokens, source_commit, dataset_dir, max_steps_this_session, microbatch_size, precision
    )


VPS_GPU_FUNCTIONS = {
    "RTX5090": train_vps_rtx5090_remote,
    "RTX4090": train_vps_rtx4090_remote,
    "A10G": train_vps_a10g_remote,
}


def main() -> int:
    # The canonical launcher calls this hook only for incremental-frontier
    # datasets. Replacing it here keeps every other launch gate unchanged while
    # making accidental paid Beam dataset production impossible on this path.
    base._stage_with_incremental_producer = _stage_from_vps_feed
    base.GPU_FUNCTIONS = VPS_GPU_FUNCTIONS
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
