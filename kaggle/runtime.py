#!/usr/bin/env python3
"""Profile-driven runtime behind the canonical ``kaggle/launch.py`` CLI.

The finite-data training and publication mechanics were qualified in the 100M
implementation and later reused through per-profile overlay modules. This
module replaces those overlays with one explicit profile table and one adapter
for training/publication. The underlying shared engines remain unchanged so
checkpoint, dataset, W&B, and durability semantics do not drift before the 2B
run.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import re
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

KAGGLE_DIR = Path(__file__).resolve().parent
REPO = KAGGLE_DIR.parent
TRAINING_ENGINE = KAGGLE_DIR / "run_20m_100m_data_scaling.py"
PUBLISH_ENGINE = KAGGLE_DIR / "build_and_push_100m.py"
PUBLISH_REQUIREMENTS = KAGGLE_DIR / "requirements-100m-publish.txt"
PUBLISH_BOOTSTRAP_ENV = "SMALL_LLM_PUBLISH_BOOTSTRAPPED"
KAGGLE_TRANSPORT_ARCHIVE = re.compile(r"^[0-9]+\.archive$")
WANDB_INIT_TIMEOUT_SECONDS = "30"

if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))


class RuntimeFailure(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class ProfileSpec:
    model_parameters: int
    training_tokens: int
    model_label: str
    token_label: str
    token_key: str
    dataset_profile_key: str
    dataset_slug: str
    launch_commit: str
    wandb_run_id: str
    wandb_run_name: str
    wandb_token_tag: str
    default_weights: str
    default_dataset: str
    default_ops: str
    weights_env: str
    dataset_env: str
    ops_env: str
    handle_env: str
    run_microbatch_probe: bool
    skipped_probe_reason: str
    selected_microbatch: int = 4
    durability_every: int = 250

    @property
    def dataset_contract(self) -> Any:
        from dataset.qualification import get_profile

        return get_profile(self.dataset_profile_key)

    @property
    def dataset_profile(self) -> str:
        return str(self.dataset_contract.plan.name)

    @property
    def dataset_run_id(self) -> str:
        run_id = self.dataset_contract.run_id
        if run_id is None:
            raise RuntimeFailure(f"dataset profile {self.dataset_profile_key} has no production run ID")
        return str(run_id)

    @property
    def target_source_tokens(self) -> int:
        return int(self.dataset_contract.target_source_tokens)

    @property
    def minimum_source_tokens(self) -> int:
        return int(self.dataset_contract.minimum_source_tokens)

    @property
    def maximum_source_tokens(self) -> int:
        return int(self.dataset_contract.maximum_source_tokens)

    @property
    def checkpoint_source_tokens(self) -> int:
        return int(self.dataset_contract.checkpoint_source_tokens)

    @property
    def run_root_name(self) -> str:
        return f"small-llm-20m-{self.token_key}-data-scaling"

    @property
    def summary_name(self) -> str:
        return f"small_llm_20m_{self.token_key}_data_scaling_summary.json"

    def production_identity(self) -> dict[str, object]:
        return {
            "run_id": self.dataset_run_id,
            "target_source_tokens": self.target_source_tokens,
            "minimum_source_tokens": self.minimum_source_tokens,
            "maximum_source_tokens": self.maximum_source_tokens,
            "checkpoint_source_tokens": self.checkpoint_source_tokens,
            "target_reached": True,
            "remote_required": True,
        }


PROFILES: dict[tuple[int, int], ProfileSpec] = {
    (20_000_000, 100_000_000): ProfileSpec(
        model_parameters=20_000_000,
        training_tokens=100_000_000,
        model_label="20M",
        token_label="100M",
        token_key="100m",
        dataset_profile_key="20m-100m",
        dataset_slug="small-llm-20m-100m-dataset-001",
        launch_commit="8e3cd9cb149facc5fa28e8108a70304c1f8c1c15",
        wandb_run_id="20m-100m-data-004",
        wandb_run_name="20M model on 100M tokens",
        wandb_token_tag="100m-tokens",
        default_weights="/data/climbmix-mixture-calibration/climbmix_code_free_weights.json",
        default_dataset="/data/small-llm/20m-100m-dataset-001",
        default_ops="/data/small-llm/20m-100m-ops",
        weights_env="SMALL_LLM_100M_WEIGHTS_FILE",
        dataset_env="SMALL_LLM_100M_DATASET_DIR",
        ops_env="SMALL_LLM_100M_OPS_DIR",
        handle_env="SMALL_LLM_KAGGLE_DATASET_HANDLE",
        run_microbatch_probe=True,
        skipped_probe_reason="",
    ),
    (20_000_000, 500_000_000): ProfileSpec(
        model_parameters=20_000_000,
        training_tokens=500_000_000,
        model_label="20M",
        token_label="500M",
        token_key="500m",
        dataset_profile_key="20m-500m",
        dataset_slug="small-llm-20m-500m-dataset-001",
        launch_commit="c0214d00047c61a290d9a138a6bd94ed5701337c",
        wandb_run_id="20m-500m-data-001",
        wandb_run_name="20M model on 500M tokens",
        wandb_token_tag="500m-tokens",
        default_weights="/data/climbmix-mixture-calibration/climbmix_code_free_weights.json",
        default_dataset="/data/small-llm/20m-500m-dataset-001",
        default_ops="/data/small-llm/20m-500m-ops",
        weights_env="SMALL_LLM_500M_WEIGHTS_FILE",
        dataset_env="SMALL_LLM_500M_DATASET_DIR",
        ops_env="SMALL_LLM_500M_OPS_DIR",
        handle_env="SMALL_LLM_500M_KAGGLE_DATASET_HANDLE",
        run_microbatch_probe=False,
        skipped_probe_reason="microbatch_4_already_qualified_on_the_same_20m_model_and_T4_training_path",
    ),
    (20_000_000, 2_000_000_000): ProfileSpec(
        model_parameters=20_000_000,
        training_tokens=2_000_000_000,
        model_label="20M",
        token_label="2B",
        token_key="2b",
        dataset_profile_key="20m-2b",
        dataset_slug="small-llm-20m-2b-dataset-001",
        launch_commit="3c920a7b682382181d4dc7557e217e6509d0dabe",
        wandb_run_id="20m-2b-data-001",
        wandb_run_name="20M model on 2B tokens",
        wandb_token_tag="2b-tokens",
        default_weights="/data/climbmix-mixture-calibration/climbmix_code_free_weights.json",
        default_dataset="/data/small-llm/20m-2b-dataset-001",
        default_ops="/data/small-llm/20m-2b-ops",
        weights_env="SMALL_LLM_2B_WEIGHTS_FILE",
        dataset_env="SMALL_LLM_2B_DATASET_DIR",
        ops_env="SMALL_LLM_2B_OPS_DIR",
        handle_env="SMALL_LLM_2B_KAGGLE_DATASET_HANDLE",
        run_microbatch_probe=False,
        skipped_probe_reason="microbatch_4_and_mixed_fla_are_already_qualified_on_the_same_20m_model_and_T4_path",
    ),
}


def resolve_profile(model_parameters: int, training_tokens: int) -> ProfileSpec:
    try:
        return PROFILES[(model_parameters, training_tokens)]
    except KeyError as error:
        supported = ", ".join(
            f"{profile.model_label}/{profile.token_label}" for profile in PROFILES.values()
        )
        raise RuntimeFailure(f"unsupported profile; supported profiles: {supported}") from error


def _load(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeFailure(f"Cannot load runtime engine {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _profile_manifest_match(
    base: Any,
    profile: ProfileSpec,
    root: Path,
) -> tuple[bool, dict[str, Any]]:
    manifest_path = root / "manifest.json"
    drive_path = root / "drive_manifest.json"
    row: dict[str, Any] = {
        "root": str(root),
        "manifest": manifest_path.is_file(),
        "drive_manifest": drive_path.is_file(),
        "train": (root / "train").is_dir(),
        "validation": (root / "validation").is_dir(),
    }
    if not all(row[key] for key in ("manifest", "drive_manifest", "train", "validation")):
        return False, row
    manifest = base.read_object(manifest_path, "dataset manifest")
    production = manifest.get("production")
    contract = profile.dataset_contract
    top = {
        "schema_version": 2,
        "sequence_format": "context_plus_one",
        "context_length": contract.context_length,
        "stored_tokens_per_sequence": contract.context_length + 1,
        "sequences_per_block": contract.sequences_per_block,
        "target_shard_bytes": contract.target_shard_bytes,
    }
    matched = all(manifest.get(key) == value for key, value in top.items())
    matched = matched and isinstance(production, Mapping)
    if isinstance(production, Mapping):
        matched = matched and all(
            production.get(key) == value
            for key, value in profile.production_identity().items()
        )
    row["run_id"] = production.get("run_id") if isinstance(production, Mapping) else None
    if matched:
        row["manifest_sha256"] = base.common.sha256(manifest_path)
        row["drive_manifest_sha256"] = base.common.sha256(drive_path)
    return bool(matched), row


def _dataset_tree_identity(base: Any, root: Path) -> dict[str, object]:
    digest = hashlib.sha256()
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file()
        and not (
            path.parent == root
            and KAGGLE_TRANSPORT_ARCHIVE.fullmatch(path.name)
        )
    )
    total = 0
    for path in files:
        relative = path.relative_to(root).as_posix()
        size = path.stat().st_size
        file_hash = base.sha256(path)
        total += size
        digest.update(f"{relative}\0{size}\0{file_hash}\n".encode())
    return {
        "tree_sha256": digest.hexdigest(),
        "file_count": len(files),
        "total_bytes": total,
    }


def train(
    profile: ProfileSpec,
    *,
    dataset_dir: str | None = None,
    max_steps_this_session: int | None = None,
) -> int:
    os.environ["WANDB_INIT_TIMEOUT"] = WANDB_INIT_TIMEOUT_SECONDS
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    import run_20m_one_click as common
    from run_20m_100m_console import install_common_console, install_experiment_console

    install_common_console(common)
    base = _load(
        TRAINING_ENGINE,
        f"small_llm_training_runtime_{profile.token_key}",
    )
    original_wandb_preflight_command = base.wandb_preflight_command
    original_trainer_command = base.trainer_command
    original_qualify_microbatch = base.qualify_microbatch
    original_run = common.run
    builtin_print = print

    root = common.WORK / profile.run_root_name
    base.DEFAULT_COMMIT = profile.launch_commit
    base.DATASET_RUN_ID = profile.dataset_run_id
    base.PROFILE = profile.dataset_profile
    base.ROOT = root
    base.WORKTREE = root / "launch-worktree"
    base.EVIDENCE = root / (
        "evidence-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    )
    base.CHECKPOINTS = root / "checkpoints"
    base.SUMMARY = common.WORK / profile.summary_name
    base.WANDB_RUN_ID = profile.wandb_run_id
    base.LOCAL_EVERY = profile.durability_every
    base.EVAL_EVERY = profile.durability_every
    base.REMOTE_EVERY = profile.durability_every
    base.MAX_STEPS_PER_SESSION = sys.maxsize

    def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--dataset-dir", type=Path)
        parser.add_argument(
            "--max-steps-this-session",
            type=int,
            default=sys.maxsize,
        )
        parsed = parser.parse_args(argv)
        parsed.launch_commit = profile.launch_commit
        return parsed

    def profile_match(root_path: Path) -> tuple[bool, dict[str, Any]]:
        return _profile_manifest_match(base, profile, root_path)

    def find_dataset(explicit: Path | None) -> tuple[Path, list[dict[str, Any]]]:
        roots = (
            [explicit.resolve()]
            if explicit
            else sorted({path.parent for path in common.INPUT.rglob("manifest.json")})
        )
        inspected: list[dict[str, Any]] = []
        matches: list[Path] = []
        for root_path in roots:
            matched, row = profile_match(root_path)
            inspected.append(row)
            if matched:
                matches.append(root_path)
        if len(matches) != 1:
            raise base.LaunchFailure(
                f"Expected exactly one attached {profile.token_label} dataset; "
                f"found {len(matches)}.\n"
                + __import__("json").dumps(inspected, indent=2)
            )
        return matches[0], inspected

    def wandb_preflight_command(
        uv: str,
        evidence: Path,
        entity: str | None = None,
    ) -> tuple[list[str], Path, Path]:
        command, result_root, result = original_wandb_preflight_command(
            uv, evidence, entity
        )
        command[command.index("--run-name") + 1] = profile.wandb_run_name
        return command, result_root, result

    def trainer_command(*args: Any, **kwargs: Any) -> list[str]:
        command = original_trainer_command(*args, **kwargs)
        if "--wandb-run-name" in command:
            command[command.index("--wandb-run-name") + 1] = profile.wandb_run_name
        command = [
            profile.wandb_token_tag if item == "100m-tokens" else item
            for item in command
        ]
        return command

    def qualify_microbatch(*args: Any, **kwargs: Any) -> dict[str, Any]:
        if profile.run_microbatch_probe:
            return original_qualify_microbatch(*args, **kwargs)
        return {
            "status": "skipped_by_experiment_decision",
            "selected_microbatch": profile.selected_microbatch,
            "probe_steps_executed": 0,
            "reason": profile.skipped_probe_reason,
        }

    def profile_run(command: Sequence[str], *args: Any, **kwargs: Any) -> dict[str, Any]:
        rewritten = [
            profile.dataset_profile_key if item == "20m-100m" else item
            for item in command
        ]
        return original_run(rewritten, *args, **kwargs)

    def profile_print(*values: object, **kwargs: object) -> None:
        replaced = [
            value.replace(
                "100M-token run completed",
                f"{profile.token_label}-token run completed",
            )
            if isinstance(value, str)
            else value
            for value in values
        ]
        builtin_print(*replaced, **kwargs)

    base.arguments = arguments
    base.profile_match = profile_match
    base.find_dataset = find_dataset
    base.wandb_preflight_command = wandb_preflight_command
    base.trainer_command = trainer_command
    base.qualify_microbatch = qualify_microbatch
    base.print = profile_print
    common.run = profile_run

    if profile.run_microbatch_probe:
        install_experiment_console(base)

    argv: list[str] = []
    if dataset_dir:
        argv += ["--dataset-dir", dataset_dir]
    if max_steps_this_session is not None:
        argv += ["--max-steps-this-session", str(max_steps_this_session)]
    return int(base.main(argv))


def publication_bootstrap_command(
    argv: Sequence[str],
    *,
    uv: str = "uv",
) -> list[str]:
    return [
        uv,
        "run",
        "--python",
        "3.13",
        "--env-file",
        str(REPO / ".env"),
        "--with-requirements",
        str(PUBLISH_REQUIREMENTS),
        "python",
        str(KAGGLE_DIR / "launch.py"),
        *argv,
    ]


def ensure_publication_environment(argv: Sequence[str]) -> None:
    if os.environ.get(PUBLISH_BOOTSTRAP_ENV) == "1":
        return
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeFailure("uv is required for dataset publication")
    if not (REPO / ".env").is_file():
        raise RuntimeFailure(f"Missing {REPO / '.env'}")
    environment = os.environ.copy()
    environment[PUBLISH_BOOTSTRAP_ENV] = "1"
    os.execvpe(
        uv,
        publication_bootstrap_command(argv, uv=uv),
        environment,
    )


def publish(
    profile: ProfileSpec,
    *,
    weights_file: str | None = None,
    dataset_dir: str | None = None,
    ops_dir: str | None = None,
    kaggle_dataset_handle: str | None = None,
    force_upload: bool = False,
    remote_ready_timeout_seconds: int | None = None,
) -> int:
    base = _load(
        PUBLISH_ENGINE,
        f"small_llm_publish_runtime_{profile.token_key}",
    )
    base.PROFILE = profile.dataset_profile
    base.RUN_ID = profile.dataset_run_id
    base.SLUG = profile.dataset_slug
    base.DEFAULT_WEIGHTS = profile.default_weights
    base.DEFAULT_DATASET = profile.default_dataset
    base.DEFAULT_OPS = profile.default_ops

    def production_identity() -> dict[str, object]:
        return profile.production_identity()

    def resolve_handle(explicit: str | None, env: Mapping[str, str]) -> str:
        handle = explicit or env.get(profile.handle_env, "")
        if not handle and env.get("KAGGLE_USERNAME"):
            handle = f"{env['KAGGLE_USERNAME']}/{profile.dataset_slug}"
        if not base.HANDLE_RE.fullmatch(handle):
            raise base.SuiteFailure(
                f"Set KAGGLE_USERNAME or {profile.handle_env}=owner/dataset in .env"
            )
        return handle

    def producer_command(config: Any, resume: bool) -> list[str]:
        command = [
            sys.executable,
            "-m",
            "dataset.qualification",
            "build",
            "--profile",
            profile.dataset_profile_key,
            "--weights-file",
            str(config.weights),
            "--output-dir",
            str(config.dataset),
        ]
        return command + (["--resume"] if resume else [])

    def derive_plan(root: Path, prefix: str, config: Any) -> None:
        base.run(
            [
                sys.executable,
                "-m",
                "dataset.qualification",
                "report",
                "--profile",
                profile.dataset_profile_key,
                "--dataset-dir",
                str(root),
                "--drive-manifest",
                str(root / "drive_manifest.json"),
                "--output",
                str(root / "qualification_plan.json"),
            ],
            f"{prefix}-qualification-plan",
            config,
        )

    base.production_identity = production_identity
    base.resolve_handle = resolve_handle
    base.producer_command = producer_command
    base.derive_plan = derive_plan
    base.tree_identity = lambda root: _dataset_tree_identity(base, root)

    aliases = {
        "SMALL_LLM_100M_WEIGHTS_FILE": profile.weights_env,
        "SMALL_LLM_100M_DATASET_DIR": profile.dataset_env,
        "SMALL_LLM_100M_OPS_DIR": profile.ops_env,
    }
    for legacy, current in aliases.items():
        value = os.environ.get(current)
        if value:
            os.environ[legacy] = value
        elif legacy != current:
            os.environ.pop(legacy, None)

    argv: list[str] = []
    if weights_file:
        argv += ["--weights-file", weights_file]
    if dataset_dir:
        argv += ["--dataset-dir", dataset_dir]
    if ops_dir:
        argv += ["--ops-dir", ops_dir]
    if kaggle_dataset_handle:
        argv += ["--kaggle-dataset-handle", kaggle_dataset_handle]
    if force_upload:
        argv.append("--force-upload")
    if remote_ready_timeout_seconds is not None:
        argv += [
            "--remote-ready-timeout-seconds",
            str(remote_ready_timeout_seconds),
        ]
    return int(base.main(argv))
