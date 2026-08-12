"""Profile-driven finite dataset production and trainer-plan CLI.

This is the single experiment-facing dataset surface. The schema-v2 producer
lives in :mod:`dataset.production`; this module freezes the identity and
geometry of approved finite scaling profiles and dispatches the shared producer
and report engines. Remote production is Hugging Face Storage Bucket backed;
legacy Drive manifests remain readable only for historical artifact compatibility.
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
    evict_remote_shards: bool = False
    incremental_frontier: bool = False
    nominal_training_tokens: int | None = None
    training_validation_blocks: int = 16

    def __post_init__(self) -> None:
        if self.incremental_frontier:
            if self.nominal_training_tokens is None or self.nominal_training_tokens <= 0:
                raise ValueError("incremental dataset profiles require nominal_training_tokens")
            if self.training_validation_blocks <= 0:
                raise ValueError("incremental dataset profiles require positive validation blocks")
        elif self.nominal_training_tokens is not None:
            raise ValueError("nominal_training_tokens is only valid for incremental profiles")

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
    "target_shard_bytes": 32 * 1024 * 1024,
}
_MODAL_B64_1G_GEOMETRY = {
    "context_length": 2_048,
    "sequences_per_block": 64,
    # 1 GiB holds 4,094 complete block-64/context-2048 optimizer blocks
    # (1,073,741,568 bytes), leaving only 256 bytes below the target.
    "target_shard_bytes": 1024**3,
}

PROFILES: dict[str, DatasetProfile] = {
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
    "modal-10b-b64": DatasetProfile(
        key="modal-10b-b64",
        run_id="modal-10b-b64-dataset-001",
        evict_remote_shards=True,
        incremental_frontier=True,
        nominal_training_tokens=10_000_000_000,
        training_validation_blocks=16,
        plan=QualificationProfile(
            name="modal-10b-b64-v1",
            target_source_tokens=10_000_000_000,
            minimum_source_tokens=9_000_000_000,
            maximum_source_tokens=11_000_000_000,
            checkpoint_source_tokens=500_000_000,
            **_MODAL_B64_1G_GEOMETRY,
        ),
    ),
}

ALIASES = {
    "10m": "20m-10m",
    "100m": "20m-100m",
    "500m": "20m-500m",
    "2b": "20m-2b",
    "modal-2b": "modal-2b-b64",
    "10b": "modal-10b-b64",
    "modal-10b": "modal-10b-b64",
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
        "--evict-remote-shards",
        "--incremental-frontier",
        "--nominal-training-tokens",
        "--training-validation-blocks",
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
    result = [
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
    if resolved.evict_remote_shards:
        result.append("--evict-remote-shards")
    if resolved.incremental_frontier:
        assert resolved.nominal_training_tokens is not None
        result.extend(
            [
                "--incremental-frontier",
                "--nominal-training-tokens",
                str(resolved.nominal_training_tokens),
                "--training-validation-blocks",
                str(resolved.training_validation_blocks),
            ]
        )
    return result


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
    """Derive a plan while binding the selected profile's dataset run ID.

    ``drive_manifest_path`` is retained solely for already-built historical
    datasets whose qualification identity includes that legacy manifest.
    """

    resolved = get_profile(profile) if isinstance(profile, str) else profile
    _validate_run_id(manifest, resolved)
    if resolved.incremental_frontier and manifest_path is not None:
        from dataset.incremental_frontier import RUN_CONTRACT_FILENAME

        contract_path = manifest_path.parent / RUN_CONTRACT_FILENAME
        if contract_path.is_file():
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
            if not isinstance(contract, Mapping) or contract.get("run_id") != resolved.run_id:
                raise ValueError("incremental qualification run contract is invalid")
            trainer = contract.get("trainer")
            if not isinstance(trainer, Mapping):
                raise ValueError("incremental qualification run contract has no trainer plan")
            planned_blocks = int(contract["planned_train_blocks"])
            raw_shards = manifest.get("shards")
            if not isinstance(raw_shards, list):
                raise ValueError("incremental qualification manifest has no shard inventory")
            train_last = max(
                (
                    int(row["last_block_id"])
                    for row in raw_shards
                    if isinstance(row, Mapping) and row.get("split") == "train"
                ),
                default=-1,
            )
            if train_last + 1 < planned_blocks:
                raise ValueError("incremental completed manifest does not cover the frozen train horizon")
            return {
                "version": 1,
                "qualification_profile": resolved.plan.name,
                "incremental_frontier": True,
                "contract_sha256": contract.get("contract_sha256"),
                "context_length": resolved.context_length,
                "sequences_per_block": resolved.sequences_per_block,
                "target_shard_bytes": resolved.target_shard_bytes,
                "train": {
                    "block_count": planned_blocks,
                    "target_tokens": int(trainer["planned_target_tokens"]),
                },
                "validation": {"block_count": int(trainer["validation_blocks"])},
                "trainer": dict(trainer),
                "identity": {
                    "manifest_path": str(manifest_path),
                    "run_contract_path": str(contract_path),
                },
            }
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
        "evict_remote_shards": profile.evict_remote_shards,
        "incremental_frontier": profile.incremental_frontier,
        "nominal_training_tokens": profile.nominal_training_tokens,
        "training_validation_blocks": profile.training_validation_blocks,
        "remote_backend": "hf_bucket",
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
    parser.add_argument(
        "--drive-manifest",
        type=Path,
        help="Legacy durability manifest for already-built historical datasets.",
    )
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
            "build     run the shared HF-backed production builder with a frozen profile\n"
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
