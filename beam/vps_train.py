#!/usr/bin/env python3
"""Launch Beam training while dataset production runs on the VPS.

Run the producer in a persistent VPS session first:

    python beam/vps_dataset_producer.py --model 100M --tokens 10B

Then launch training with the ordinary Beam arguments:

    python beam/vps_train.py --model 100M --tokens 10B --gpu RTX5090

This wrapper deliberately replaces the incremental Beam CPU producer with the
external VPS feed. All import, staging, visibility, checkpoint, microbatch, and
GPU-dispatch gates remain the canonical implementation in ``beam.launch``.
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

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


def main() -> int:
    # The canonical launcher calls this hook only for incremental-frontier
    # datasets. Replacing it here keeps every other launch gate unchanged while
    # making accidental paid Beam dataset production impossible on this path.
    base._stage_with_incremental_producer = _stage_from_vps_feed
    return base.main()


if __name__ == "__main__":
    raise SystemExit(main())
