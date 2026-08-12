"""Profile-driven finite dataset production and trainer-plan CLI.

This is the single experiment-facing dataset surface.  The schema-v2 producer
lives in :mod:`dataset.production`; this module only freezes the identity and
geometry of approved finite scaling profiles and dispatches the shared producer
and report engines.

Examples::

    python -m dataset.qualification profiles
    python -m dataset.qualification build --profile 20m-2b \
        --weights-file weights.json --output-dir /data/20m-2b
    python -m dataset.qualification report --profile 20m-2b \
        --dataset-dir /data/20m-2b --drive-manifest /data/20m-2b/drive_manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

from dataset import config
from dataset.production.cli import main as production_main
from dataset.qualification_report import (
    QualificationProfile,
    derive_plan as derive_qualification_plan,
)
from dataset.src.storage import write_json_atomic


@dataclass(frozen=True, slots=True)
class DatasetProfile:
    """One immutable finite dataset identity plus its trainer-plan contract."""

    key: str
    run_id: str | None
    plan: QualificationProfile
    production_enabled: bool = True

    @property
    def target_source_tokens(self) -> int:
        return self.plan.target_source_tokens

    @property
    def minimum_source_tokens(self) -> int:
        return self.plan.minimum_source_tokens

    @property
    def maximum_source_tokens(self) -> int:
        return self.plan.maximum_source_tokens

    @property
    def checkpoint_source_tokens(self) -> int:
        return self.plan.checkpoint_source_tokens

    @property
    def context_length(self) -> int:
        return self.plan.context_length

    @property
    def sequences_per_block(self) -> int:
        return self.plan.sequences_per_block

    @property
    def target_shard_bytes(self) -> int:
        return self.plan.target_shard_bytes


_COMMON_GEOMETRY = {
    "context_length": 2_048,
    "sequences_per_block": 16,
    "target_shard_bytes": 8 * 1024 * 1024,
}
_MODAL_B64_GEOMETRY = {
    "context_length": 2_048,
    "sequences_per_block": 64,
    # Four full legacy 8 MiB/16-sequence shards reblock into one full
    # 32 MiB/64-sequence shard without changing any stored sequence bytes.
    "target_shard_bytes": 32 * 1024 * 1024,
}

PROFILES: dict[str, DatasetProfile] = {
    # Historical first finite qualification.  It is retained only so its
    # manifest/plan can still be reproduced; producing this dataset again is
    # intentionally disabled after operational acceptance completed.
    "20m-10m": DatasetProfile(
        key="20m-10m",
        run_id=None,
        production_enabled=False,
        plan=QualificationProfile(
            name="20m-one-pass-v1",
            target_source_tokens=10_000_000,
            minimum_source_tokens=9_000_000,
            maximum_source_tokens=11_000_000,
            checkpoint_source_tokens=2_000_000,
            **_COMMON_GEOMETRY,
        ),
    ),
    "20m-100m": DatasetProfile(
        key="20m-100m",
        run_id="20m-100m-dataset-001",
        plan=QualificationProfile(
            name="20m-100m-data-scaling-v1",
            target_source_tokens=100_000_000,
            minimum_source_tokens=90_000_000,
            maximum_source_tokens=110_000_000,
            checkpoint_source_tokens=20_000_000,
            **_COMMON_GEOMETRY,
        ),
    ),
    "20m-500m": DatasetProfile(
        key="20m-500m",
        run_id="20m-500m-dataset-001",
        plan=QualificationProfile(
            name="20m-500m-data-scaling-v1",
            target_source_tokens=500_000_000,
            minimum_source_tokens=450_000_000,
            maximum_source_tokens=550_000_000,
            checkpoint_source_tokens=20_000_000,
            **_COMMON_GEOMETRY,
        ),
    ),
    "20m-2b": DatasetProfile(
        key="20m-2b",
        run_id="20m-2b-dataset-001",
        plan=QualificationProfile(
            name="20m-2b-data-scaling-v1",
            target_source_tokens=2_000_000_000,
            minimum_source_tokens=1_800_000_000,
            maximum_source_tokens=2_200_000_000,
            checkpoint_source_tokens=80_000_000,
            **_COMMON_GEOMETRY,
        ),
    ),
    "modal-2b-b64": DatasetProfile(
        key="modal-2b-b64",
        run_id="modal-2b-b64-dataset-001",
        plan=QualificationProfile(
            name="modal-2b-b64-v1",
            target_source_tokens=2_000_000_000,
            minimum_source_tokens=1_800_000_000,
            maximum_source_tokens=2_200_000_000,
            checkpoint_source_tokens=80_000_000,
            **_MODAL_B64_GEOMETRY,
        ),
    ),
}

ALIASES = {
    "10m": "20m-10m",
    "100m": "20m-100m",
    "500m": "20m-500m",
    "2b": "20m-2b",
    "modal-2b": "modal-2b-b64",
}

_LOCKED_PRODUCTION_FLAGS = frozenset(
    {
        "--run-id",
        "--target-tokens",
        "--minimum-tokens",
        "--maximum-tokens",
        "--checkpoint-source-tokens",
        "--context-length",
        "--sequences-per-block",
        "--target-shard-bytes",
        "--allow-local-only",
    }
)


def get_profile(key: str) -> DatasetProfile:
    """Resolve a canonical profile key or short token-budget alias."""

    canonical = ALIASES.get(key.lower(), key.lower())
    try:
        return PROFILES[canonical]
    except KeyError as error:
        supported = ", ".join(PROFILES)
        raise ValueError(f"unknown dataset profile {key!r}; choose one of: {supported}") from error


def production_arguments(profile: DatasetProfile | str, argv: Sequence[str]) -> list[str]:
    """Append immutable profile identity to safe producer tuning arguments."""

    resolved = get_profile(profile) if isinstance(profile, str) else profile
    if not resolved.production_enabled or resolved.run_id is None:
        raise SystemExit(
            f"profile {resolved.key} is historical and cannot be produced again; "
            "use the recorded artifacts/Git history for reproduction"
        )
    supplied = {
        argument.split("=", 1)[0]
        for argument in argv
        if argument.startswith("--")
    }
    conflicts = sorted(supplied & _LOCKED_PRODUCTION_FLAGS)
    if conflicts:
        raise SystemExit(
            f"dataset profile {resolved.key} fixes these arguments: "
            + ", ".join(conflicts)
        )
    return [
        *argv,
        "--run-id",
        resolved.run_id,
        "--target-tokens",
        str(resolved.target_source_tokens),
        "--minimum-tokens",
        str(resolved.minimum_source_tokens),
        "--maximum-tokens",
        str(resolved.maximum_source_tokens),
        "--checkpoint-source-tokens",
        str(resolved.checkpoint_source_tokens),
        "--context-length",
        str(resolved.context_length),
        "--sequences-per-block",
        str(resolved.sequences_per_block),
        "--target-shard-bytes",
        str(resolved.target_shard_bytes),
    ]


def _validate_run_id(manifest: Mapping[str, object], profile: DatasetProfile) -> None:
    if profile.run_id is None:
        return
    production = manifest.get("production")
    if not isinstance(production, Mapping):
        raise ValueError("qualification manifest has no production identity")
    actual = production.get("run_id")
    if actual != profile.run_id:
        raise ValueError(
            f"qualification production run_id mismatch: expected {profile.run_id!r}, got {actual!r}"
        )


def derive_plan(
    manifest: Mapping[str, object],
    *,
    profile: DatasetProfile | str,
    manifest_path: Path | None = None,
    drive_manifest_path: Path | None = None,
) -> dict[str, object]:
    """Derive a plan while binding the selected profile's dataset run ID."""

    resolved = get_profile(profile) if isinstance(profile, str) else profile
    _validate_run_id(manifest, resolved)
    return derive_qualification_plan(
        manifest,
        profile=resolved.plan,
        manifest_path=manifest_path,
        drive_manifest_path=drive_manifest_path,
    )


def profile_payload(profile: DatasetProfile) -> dict[str, object]:
    payload = asdict(profile.plan)
    return {
        "key": profile.key,
        "run_id": profile.run_id,
        "production_enabled": profile.production_enabled,
        **payload,
    }


def _profile_parser(command: str) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=f"python -m dataset.qualification {command}",
        add_help=True,
    )
    parser.add_argument(
        "--profile",
        required=True,
        choices=sorted(set(PROFILES) | set(ALIASES)),
    )
    return parser


def _run_build(argv: Sequence[str]) -> int:
    parser = _profile_parser("build")
    args, producer_args = parser.parse_known_args(argv)
    profile = get_profile(args.profile)
    return production_main(production_arguments(profile, producer_args))


def _run_report(argv: Sequence[str]) -> int:
    parser = _profile_parser("report")
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--drive-manifest", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    profile = get_profile(args.profile)
    manifest_path = args.dataset_dir / config.MANIFEST_FILENAME
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(payload, Mapping):
            raise ValueError("qualification manifest must contain a JSON object")
        plan = derive_plan(
            payload,
            profile=profile,
            manifest_path=manifest_path,
            drive_manifest_path=args.drive_manifest,
        )
        if args.output is not None:
            write_json_atomic(args.output, plan)
            print(
                json.dumps(
                    {
                        "qualification_report": "written",
                        "output": str(args.output),
                        "profile": profile.key,
                    },
                    sort_keys=True,
                )
            )
        else:
            print(json.dumps(plan, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - concise CLI failure boundary
        print(
            f"qualification report error: {type(error).__name__}: {error}",
            file=sys.stderr,
        )
        return 1


def main(argv: Sequence[str] | None = None) -> int:
    supplied = list(sys.argv[1:] if argv is None else argv)
    if not supplied or supplied[0] in {"-h", "--help"}:
        print(
            "usage: python -m dataset.qualification {profiles,build,report} ...\n\n"
            "profiles  list frozen finite-dataset profiles\n"
            "build     run the shared production builder with a frozen profile\n"
            "report    derive the exact one-pass trainer plan from a manifest"
        )
        return 0

    command, *rest = supplied
    if command == "profiles":
        if rest:
            raise SystemExit("profiles does not accept arguments")
        print(
            json.dumps(
                [profile_payload(PROFILES[key]) for key in PROFILES],
                indent=2,
                sort_keys=True,
            )
        )
        return 0
    if command == "build":
        return _run_build(rest)
    if command == "report":
        return _run_report(rest)
    raise SystemExit(f"unknown command {command!r}; choose profiles, build, or report")


if __name__ == "__main__":
    raise SystemExit(main())
