"""Derive the exact one-pass plan for the 20M-model/1B-token scaling run."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

from dataset.qualification_1b import (
    CHECKPOINT_SOURCE_TOKENS,
    CONTEXT_LENGTH,
    MAXIMUM_SOURCE_TOKENS,
    MINIMUM_SOURCE_TOKENS,
    SEQUENCES_PER_BLOCK,
    TARGET_SHARD_BYTES,
    TARGET_SOURCE_TOKENS,
)
from dataset.qualification_report import (
    QualificationProfile,
    derive_plan as derive_profile_plan,
    run_cli,
)

PROFILE = QualificationProfile(
    name="20m-1b-data-scaling-v1",
    target_source_tokens=TARGET_SOURCE_TOKENS,
    minimum_source_tokens=MINIMUM_SOURCE_TOKENS,
    maximum_source_tokens=MAXIMUM_SOURCE_TOKENS,
    checkpoint_source_tokens=CHECKPOINT_SOURCE_TOKENS,
    context_length=CONTEXT_LENGTH,
    sequences_per_block=SEQUENCES_PER_BLOCK,
    target_shard_bytes=TARGET_SHARD_BYTES,
)


def derive_plan(
    manifest: Mapping[str, object],
    *,
    manifest_path: Path | None = None,
    drive_manifest_path: Path | None = None,
) -> dict[str, object]:
    return derive_profile_plan(
        manifest,
        profile=PROFILE,
        manifest_path=manifest_path,
        drive_manifest_path=drive_manifest_path,
    )


def main(argv: Sequence[str] | None = None) -> int:
    return run_cli(
        argv,
        profile=PROFILE,
        description=(
            "Derive the exact 20M-model/1B-token trainer plan from its "
            "verified manifest."
        ),
    )


if __name__ == "__main__":
    raise SystemExit(main())
