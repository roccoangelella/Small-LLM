#!/usr/bin/env python3
"""Profile-driven runtime behind the canonical ``kaggle/launch_sft.py`` CLI."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
import re
import shutil
import subprocess
import sys
from typing import Sequence

KAGGLE_DIR = Path(__file__).resolve().parent
REPO = KAGGLE_DIR.parent
KAGGLE_WORK = Path("/kaggle/working")
KAGGLE_INPUT = Path("/kaggle/input")
PINNED_LAUNCH_COMMIT = "184adccc1c12437046594ac674bc8d61eb710125"


def _portable_work_root(
    configured: str | None,
    *,
    kaggle_work: Path = KAGGLE_WORK,
    repo: Path = REPO,
) -> Path:
    """Resolve a writable machine-agnostic work root.

    Explicit configuration wins. Kaggle keeps its conventional ephemeral working
    root when present. Other machines default beside the controlling repository so
    running the launcher from a normal clone never requires root-level /kaggle paths.
    """

    if configured:
        return Path(configured).expanduser().resolve()
    if kaggle_work.is_dir():
        return kaggle_work.resolve()
    return (repo.parent / "small-llm-work").resolve()


def _portable_input_root(
    configured: str | None,
    *,
    kaggle_input: Path = KAGGLE_INPUT,
) -> Path:
    """Resolve the optional implicit input root used by train/eval discovery."""

    if configured:
        return Path(configured).expanduser().resolve()
    return kaggle_input


WORK = _portable_work_root(os.environ.get("SMALL_LLM_WORK_DIR"))
INPUT = _portable_input_root(os.environ.get("SMALL_LLM_INPUT_DIR"))


class RuntimeFailure(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class SFTProfileSpec:
    model_parameters: int
    parent_training_tokens: int
    model_label: str
    token_label: str
    token_key: str
    parent_run_id: str
    sft_run_id: str
    wandb_run_id: str
    wandb_run_name: str
    dataset_slug: str
    known_parent_consumed_tokens: int | None
    launch_commit: str = PINNED_LAUNCH_COMMIT
    sft_fraction_numerator: int = 4
    sft_fraction_denominator: int = 100
    microbatch_size: int = 4
    cadence_steps: int = 250
    learning_rate: float = 3e-5

    @property
    def run_root(self) -> Path:
        return WORK / f"small-llm-20m-{self.token_key}-sft"

    @property
    def worktree(self) -> Path:
        return self.run_root / "launch-worktree"

    @property
    def checkpoint_dir(self) -> Path:
        return self.run_root / "checkpoints"

    @property
    def default_bundle(self) -> Path:
        return WORK / f"small-llm-20m-{self.token_key}-sft-bundle"

    @property
    def default_prepared(self) -> Path:
        return WORK / "small-llm-sft-smoltalk-pinned"

    @property
    def default_publication_ops(self) -> Path:
        return self.run_root / "bundle-publication"

    @property
    def requested_sft_targets(self) -> int | None:
        if self.known_parent_consumed_tokens is None:
            return None
        return self.known_parent_consumed_tokens * self.sft_fraction_numerator // self.sft_fraction_denominator


PROFILES: dict[tuple[int, int], SFTProfileSpec] = {
    (20_000_000, 500_000_000): SFTProfileSpec(
        model_parameters=20_000_000,
        parent_training_tokens=500_000_000,
        model_label="20M",
        token_label="500M",
        token_key="500m",
        parent_run_id="20m-500m-data-001",
        sft_run_id="20m-500m-sft-s0-001",
        wandb_run_id="20m-500m-sft-s0-001",
        wandb_run_name="20M / 500M parent / SFT S0",
        dataset_slug="small-llm-20m-500m-sft-s0-001",
        known_parent_consumed_tokens=500_156_416,
    ),
    (20_000_000, 2_000_000_000): SFTProfileSpec(
        model_parameters=20_000_000,
        parent_training_tokens=2_000_000_000,
        model_label="20M",
        token_label="2B",
        token_key="2b",
        parent_run_id="20m-2b-data-001",
        sft_run_id="20m-2b-sft-s0-001",
        wandb_run_id="20m-2b-sft-s0-001",
        wandb_run_name="20M / 2B parent / SFT S0",
        dataset_slug="small-llm-20m-2b-sft-s0-001",
        known_parent_consumed_tokens=None,
    ),
}


def resolve_profile(model_parameters: int, parent_training_tokens: int) -> SFTProfileSpec:
    try:
        return PROFILES[(model_parameters, parent_training_tokens)]
    except KeyError as error:
        supported = ", ".join(f"{profile.model_label}/{profile.token_label}" for profile in PROFILES.values())
        raise RuntimeFailure(f"unsupported SFT profile; supported profiles: {supported}") from error


def _run(command: Sequence[str], *, cwd: Path) -> int:
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(
        list(command),
        cwd=cwd,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
            "UV_LINK_MODE": "copy",
            "WANDB_INIT_TIMEOUT": "30",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
        },
    )
    if result.returncode:
        raise RuntimeFailure(f"command failed with exit code {result.returncode}: {' '.join(command)}")
    return 0


def _prepare_worktree(profile: SFTProfileSpec) -> Path:
    commit = profile.launch_commit
    if not re.fullmatch(r"[0-9a-f]{40}", commit):
        raise RuntimeFailure("SFT launch commit is not pinned")
    if subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    ).returncode:
        raise RuntimeFailure(f"pinned SFT launch commit is unavailable: {commit}")
    if subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=REPO,
        text=True,
    ).strip():
        raise RuntimeFailure("controlling Small-LLM clone has tracked modifications")

    profile.run_root.mkdir(parents=True, exist_ok=True)
    if profile.worktree.exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(profile.worktree)],
            cwd=REPO,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        shutil.rmtree(profile.worktree, ignore_errors=True)
    subprocess.run(["git", "worktree", "prune"], cwd=REPO, check=False)
    subprocess.run(["git", "worktree", "add", "--detach", str(profile.worktree), commit], cwd=REPO, check=True)
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=profile.worktree, text=True).strip()
    if actual != commit:
        raise RuntimeFailure("SFT launch worktree commit mismatch")
    return profile.worktree


def _uv_prefix(
    *,
    datasets: bool = False,
    wandb: bool = False,
    kagglehub: bool = False,
) -> list[str]:
    command = ["uv", "run", "--python", "3.13", "--extra", "model", "--extra", "post-training"]
    if datasets:
        command += ["--with", "datasets"]
    if wandb:
        command += ["--with", "wandb==0.26.1"]
    if kagglehub:
        command += ["--with", "kagglehub"]
    return command


def _find_bundle(explicit: str | None, profile: SFTProfileSpec | None = None) -> Path:
    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not (root / "bundle-manifest.json").is_file():
            raise RuntimeFailure(f"not an SFT bundle: {root}")
        return root
    if profile is not None:
        if profile.default_bundle.is_dir() and (profile.default_bundle / "bundle-manifest.json").is_file():
            return profile.default_bundle
        candidate = REPO / "tests" / "test_datasets" / profile.dataset_slug
        if candidate.is_dir() and (candidate / "bundle-manifest.json").is_file():
            return candidate.resolve()
    if INPUT.is_dir():
        matches = sorted({path.parent.resolve() for path in INPUT.rglob("bundle-manifest.json") if path.is_file()})
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise RuntimeFailure(f"expected exactly one attached SFT bundle; found {len(matches)}: {matches}")
    raise RuntimeFailure(
        "no implicit SFT input root is available; pass --dataset-dir or set SMALL_LLM_INPUT_DIR"
    )


def _resolve_replay_root(value: str) -> Path:
    root = Path(value).expanduser().resolve()
    if root.is_file():
        raise RuntimeFailure(
            "--replay-root must be the pretraining dataset directory containing manifest.json, "
            f"not a file: {root}"
        )
    if not root.is_dir():
        raise RuntimeFailure(f"pretraining replay dataset directory does not exist: {root}")
    manifest = root / "manifest.json"
    if not manifest.is_file() or manifest.is_symlink():
        raise RuntimeFailure(f"replay root has no safe immutable manifest.json: {root}")
    return root


def _exact_parent_tokens(
    profile: SFTProfileSpec,
    supplied: int | None,
) -> int:
    exact = supplied if supplied is not None else profile.known_parent_consumed_tokens
    if exact is None:
        raise RuntimeFailure(
            "this parent run has no completed exact token count yet; pass --parent-consumed-tokens from the verified final checkpoint"
        )
    return exact


def _expected_sft_targets(profile: SFTProfileSpec, parent_tokens: int) -> int:
    return parent_tokens * profile.sft_fraction_numerator // profile.sft_fraction_denominator


def _verify_existing_bundle_budget(output: Path, *, expected_targets: int) -> bool:
    manifest_path = output / "bundle-manifest.json"
    if not manifest_path.is_file():
        return False
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeFailure(f"existing SFT bundle manifest is invalid: {manifest_path}") from error
    if not isinstance(payload, dict):
        raise RuntimeFailure("existing SFT bundle manifest is not an object")
    if payload.get("train_target_tokens_requested") != expected_targets:
        raise RuntimeFailure(
            "existing SFT bundle was built for a different parent token budget: "
            f"expected {expected_targets}, found {payload.get('train_target_tokens_requested')}"
        )
    return True


def prepare(
    profile: SFTProfileSpec,
    *,
    replay_root: str,
    prepared_dir: str | None,
    output_dir: str | None,
    parent_consumed_tokens: int | None,
    revision: str | None,
) -> int:
    replay = _resolve_replay_root(replay_root)
    worktree = _prepare_worktree(profile)
    prepared = Path(prepared_dir).expanduser().resolve() if prepared_dir else profile.default_prepared
    output = Path(output_dir).expanduser().resolve() if output_dir else profile.default_bundle
    prepared_manifest_path = prepared / "prepared-manifest.json"
    revision_args = ["--revision", revision] if revision else []
    if prepared_manifest_path.is_file():
        if revision is not None:
            try:
                prepared_manifest = json.loads(prepared_manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError) as error:
                raise RuntimeFailure("existing prepared SFT source manifest is invalid") from error
            if not isinstance(prepared_manifest, dict) or prepared_manifest.get("revision") != revision:
                raise RuntimeFailure("existing prepared SFT source uses a different pinned revision")
    else:
        _run(
            _uv_prefix(datasets=True) + ["python", "-m", "post_training.sft.bundle", "prepare", "--output-dir", str(prepared), *revision_args],
            cwd=worktree,
        )

    exact_parent_tokens = _exact_parent_tokens(profile, parent_consumed_tokens)
    expected_targets = _expected_sft_targets(profile, exact_parent_tokens)
    if not _verify_existing_bundle_budget(output, expected_targets=expected_targets):
        if output.exists():
            raise RuntimeFailure(
                f"refusing to replace incomplete/non-bundle SFT output directory: {output}"
            )
        _run(
            _uv_prefix() + [
                "python", "-m", "post_training.sft.bundle", "build",
                "--prepared-dir", str(prepared),
                "--replay-root", str(replay),
                "--output-dir", str(output),
                "--parent-consumed-tokens", str(exact_parent_tokens),
                "--optimizer-target-tokens", "32768",
                "--instruction-share", "0.85",
                "--replay-share", "0.15",
                "--seed", "17",
            ],
            cwd=worktree,
        )
    return _run(
        _uv_prefix() + ["python", "-m", "post_training.sft.bundle", "verify", "--dataset-dir", str(output)],
        cwd=worktree,
    )


def _resolve_kaggle_handle(profile: SFTProfileSpec, explicit: str | None) -> str:
    handle = explicit or os.environ.get("SMALL_LLM_SFT_KAGGLE_DATASET_HANDLE")
    if not handle and os.environ.get("KAGGLE_USERNAME"):
        handle = f"{os.environ['KAGGLE_USERNAME']}/{profile.dataset_slug}"
    if not handle or handle.count("/") != 1:
        raise RuntimeFailure(
            "pass --kaggle-dataset-handle owner/dataset, set SMALL_LLM_SFT_KAGGLE_DATASET_HANDLE, or set KAGGLE_USERNAME"
        )
    return handle


def publish(
    profile: SFTProfileSpec,
    *,
    replay_root: str,
    prepared_dir: str | None,
    output_dir: str | None,
    parent_consumed_tokens: int | None,
    revision: str | None,
    kaggle_dataset_handle: str | None,
    ops_dir: str | None,
    force_upload: bool,
    remote_ready_timeout_seconds: int,
) -> int:
    prepare(
        profile,
        replay_root=replay_root,
        prepared_dir=prepared_dir,
        output_dir=output_dir,
        parent_consumed_tokens=parent_consumed_tokens,
        revision=revision,
    )
    worktree = _prepare_worktree(profile)
    bundle = Path(output_dir).expanduser().resolve() if output_dir else profile.default_bundle
    handle = _resolve_kaggle_handle(profile, kaggle_dataset_handle)
    ops = Path(ops_dir).expanduser().resolve() if ops_dir else profile.default_publication_ops
    command = _uv_prefix(kagglehub=True) + [
        "python",
        str(worktree / "kaggle" / "sft_publish.py"),
        "--dataset-dir",
        str(bundle.resolve()),
        "--handle",
        handle,
        "--ops-dir",
        str(ops.resolve()),
        "--remote-ready-timeout-seconds",
        str(remote_ready_timeout_seconds),
    ]
    if force_upload:
        command.append("--force-upload")
    return _run(command, cwd=worktree)


def _wandb_preflight(profile: SFTProfileSpec, *, worktree: Path, entity: str | None) -> None:
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeFailure("WANDB_API_KEY is required for online SFT training")
    root = profile.run_root / "wandb-preflight"
    result = root / "result.json"
    command = _uv_prefix(wandb=True) + [
        "python", str(worktree / "kaggle" / "wandb_preflight.py"),
        "--project", "Small-LLM",
        "--run-id", profile.wandb_run_id,
        "--run-name", profile.wandb_run_name,
        "--dir", str(root),
        "--result", str(result),
        "--init-timeout", "30",
    ]
    if entity:
        command += ["--entity", entity]
    _run(command, cwd=worktree)
    payload = json.loads(result.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or payload.get("status") != "passed":
        raise RuntimeFailure("SFT W&B preflight did not pass")


def train(
    profile: SFTProfileSpec,
    *,
    dataset_dir: str | None,
    parent_repo_id: str | None,
    checkpoint_repo_id: str | None,
    max_steps_this_session: int | None,
    wandb_entity: str | None,
) -> int:
    worktree = _prepare_worktree(profile)
    bundle = _find_bundle(dataset_dir)
    parent_repo = parent_repo_id or os.environ.get("SMALL_LLM_HF_REPO_ID")
    checkpoint_repo = checkpoint_repo_id or os.environ.get("SMALL_LLM_SFT_HF_REPO_ID", parent_repo)
    if not parent_repo:
        raise RuntimeFailure("pass --parent-repo-id or set SMALL_LLM_HF_REPO_ID")
    if not checkpoint_repo:
        raise RuntimeFailure("pass --checkpoint-repo-id or set SMALL_LLM_SFT_HF_REPO_ID")
    entity = wandb_entity or os.environ.get("WANDB_ENTITY")
    _wandb_preflight(profile, worktree=worktree, entity=entity)
    command = _uv_prefix(wandb=True) + [
        "python", "-m", "post_training.sft.train_cli",
        "--dataset-dir", str(bundle),
        "--checkpoint-dir", str(profile.checkpoint_dir),
        "--sft-run-id", profile.sft_run_id,
        "--parent-repo-id", parent_repo,
        "--parent-run-id", profile.parent_run_id,
        "--parent-pointer", "best",
        "--checkpoint-repo-id", checkpoint_repo,
        "--device", "cuda",
        "--precision", "fp16",
        "--microbatch-size", str(profile.microbatch_size),
        "--learning-rate", str(profile.learning_rate),
        "--checkpoint-every-steps", str(profile.cadence_steps),
        "--evaluation-every-steps", str(profile.cadence_steps),
        "--remote-publish-every-steps", str(profile.cadence_steps),
        "--remote-rolling-latest-only",
        "--wandb-mode", "online",
        "--wandb-project", "Small-LLM",
        "--wandb-run-id", profile.wandb_run_id,
        "--wandb-run-name", profile.wandb_run_name,
    ]
    if entity:
        command += ["--wandb-entity", entity]
    if max_steps_this_session is not None:
        command += ["--max-steps-this-session", str(max_steps_this_session)]
    return _run(command, cwd=worktree)


def evaluate(
    profile: SFTProfileSpec,
    *,
    dataset_dir: str | None = None,
    eval_dir: str | None = None,
    parent_repo_id: str | None = None,
    checkpoint_repo_id: str | None = None,
    parent_checkpoint_dir: str | None = None,
    sft_checkpoint_dir: str | None = None,
    output: str | None = None,
    suite: str = "full",
    device: str = "auto",
    precision: str = "auto",
    batch_size: int = 1,
    validation_blocks: int = 32,
    test_blocks: int = 32,
) -> int:
    bundle = _find_bundle(dataset_dir, profile)
    parent_repo = parent_repo_id or os.environ.get("SMALL_LLM_PARENT_HF_REPO_ID") or os.environ.get("SMALL_LLM_HF_REPO_ID")
    checkpoint_repo = checkpoint_repo_id or os.environ.get("SMALL_LLM_SFT_HF_REPO_ID", parent_repo)
    if not parent_checkpoint_dir and not parent_repo:
        raise RuntimeFailure("evaluation requires parent checkpoint repository ID or local directory")
    if not sft_checkpoint_dir and not checkpoint_repo:
        raise RuntimeFailure("evaluation requires SFT checkpoint repository ID or local directory")
    if eval_dir:
        selected_eval_dir = Path(eval_dir).expanduser().resolve()
    else:
        test_eval = REPO / "tests" / "test_datasets" / "eval_core_v1"
        selected_eval_dir = test_eval.resolve() if (test_eval / "manifest.json").is_file() else (WORK / "eval_core_v1")

    selected_output = (
        Path(output).expanduser().resolve()
        if output
        else profile.run_root / f"post-sft-{suite}-qualification.json"
    )
    cmd = [
        sys.executable, "-m", "post_training.sft.eval_suite",
        "--dataset-dir", str(bundle),
        "--eval-dir", str(selected_eval_dir),
        "--suite", suite,
        "--device", device,
        "--precision", precision,
        "--batch-size", str(batch_size),
        "--validation-blocks", str(validation_blocks),
        "--test-blocks", str(test_blocks),
        "--output", str(selected_output),
    ]
    if parent_checkpoint_dir:
        cmd += ["--parent-checkpoint-dir", str(Path(parent_checkpoint_dir).expanduser().resolve())]
    else:
        cmd += [
            "--parent-repo-id", parent_repo,
            "--parent-run-id", profile.parent_run_id,
            "--parent-pointer", "best",
        ]
    if sft_checkpoint_dir:
        cmd += ["--sft-checkpoint-dir", str(Path(sft_checkpoint_dir).expanduser().resolve())]
    else:
        cmd += [
            "--sft-repo-id", checkpoint_repo,
            "--sft-run-id", profile.sft_run_id,
            "--sft-pointer", "latest",
        ]
    return _run(_uv_prefix() + cmd, cwd=REPO)


__all__ = [
    "PROFILES",
    "RuntimeFailure",
    "SFTProfileSpec",
    "evaluate",
    "prepare",
    "publish",
    "resolve_profile",
    "train",
]
