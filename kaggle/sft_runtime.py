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
from typing import Sequence

KAGGLE_DIR = Path(__file__).resolve().parent
REPO = KAGGLE_DIR.parent
WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
PINNED_LAUNCH_COMMIT = "b5e706d4c940310dc2fe076874a9e0d67ae10266"


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
    def requested_sft_targets(self) -> int | None:
        if self.known_parent_consumed_tokens is None:
            return None
        return (
            self.known_parent_consumed_tokens
            * self.sft_fraction_numerator
            // self.sft_fraction_denominator
        )


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
        known_parent_consumed_tokens=None,
    ),
}


def resolve_profile(model_parameters: int, parent_training_tokens: int) -> SFTProfileSpec:
    try:
        return PROFILES[(model_parameters, parent_training_tokens)]
    except KeyError as error:
        supported = ", ".join(
            f"{profile.model_label}/{profile.token_label}" for profile in PROFILES.values()
        )
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
        raise RuntimeFailure(
            f"command failed with exit code {result.returncode}: {' '.join(command)}"
        )
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
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(profile.worktree), commit],
        cwd=REPO,
        check=True,
    )
    actual = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=profile.worktree, text=True).strip()
    if actual != commit:
        raise RuntimeFailure("SFT launch worktree commit mismatch")
    return profile.worktree


def _uv_prefix(*, datasets: bool = False, wandb: bool = False) -> list[str]:
    command = ["uv", "run", "--python", "3.13", "--extra", "model", "--extra", "post-training"]
    if datasets:
        command += ["--with", "datasets"]
    if wandb:
        command += ["--with", "wandb==0.26.1"]
    return command


def _find_bundle(explicit: str | None) -> Path:
    if explicit:
        root = Path(explicit).resolve()
        if not (root / "bundle-manifest.json").is_file():
            raise RuntimeFailure(f"not an SFT bundle: {root}")
        return root
    matches = sorted(
        {path.parent.resolve() for path in INPUT.rglob("bundle-manifest.json") if path.is_file()}
    )
    if len(matches) != 1:
        raise RuntimeFailure(
            f"expected exactly one attached SFT bundle; found {len(matches)}: {matches}"
        )
    return matches[0]


def prepare(
    profile: SFTProfileSpec,
    *,
    replay_root: str,
    prepared_dir: str | None,
    output_dir: str | None,
    parent_consumed_tokens: int | None,
    revision: str | None,
) -> int:
    worktree = _prepare_worktree(profile)
    prepared = Path(prepared_dir) if prepared_dir else profile.default_prepared
    output = Path(output_dir) if output_dir else profile.default_bundle
    revision_args = ["--revision", revision] if revision else []

    if not (prepared / "prepared-manifest.json").is_file():
        _run(
            _uv_prefix(datasets=True)
            + [
                "python",
                "-m",
                "post_training.sft.bundle",
                "prepare",
                "--output-dir",
                str(prepared),
                *revision_args,
            ],
            cwd=worktree,
        )

    exact_parent_tokens = (
        parent_consumed_tokens
        if parent_consumed_tokens is not None
        else profile.known_parent_consumed_tokens
    )
    if exact_parent_tokens is None:
        raise RuntimeFailure(
            "this parent run has no completed exact token count yet; "
            "pass --parent-consumed-tokens from the verified final checkpoint"
        )
    _run(
        _uv_prefix()
        + [
            "python",
            "-m",
            "post_training.sft.bundle",
            "build",
            "--prepared-dir",
            str(prepared),
            "--replay-root",
            str(Path(replay_root).resolve()),
            "--output-dir",
            str(output),
            "--parent-consumed-tokens",
            str(exact_parent_tokens),
            "--optimizer-target-tokens",
            "32768",
            "--instruction-share",
            "0.85",
            "--replay-share",
            "0.15",
            "--seed",
            "17",
        ],
        cwd=worktree,
    )
    return _run(
        _uv_prefix()
        + [
            "python",
            "-m",
            "post_training.sft.bundle",
            "verify",
            "--dataset-dir",
            str(output),
        ],
        cwd=worktree,
    )


def _wandb_preflight(profile: SFTProfileSpec, *, worktree: Path, entity: str | None) -> None:
    if not os.environ.get("WANDB_API_KEY"):
        raise RuntimeFailure("WANDB_API_KEY is required for online SFT training")
    root = profile.run_root / "wandb-preflight"
    result = root / "result.json"
    command = _uv_prefix(wandb=True) + [
        "python",
        str(worktree / "kaggle" / "wandb_preflight.py"),
        "--project",
        "Small-LLM",
        "--run-id",
        profile.wandb_run_id,
        "--run-name",
        profile.wandb_run_name,
        "--dir",
        str(root),
        "--result",
        str(result),
        "--init-timeout",
        "30",
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
        "python",
        "-m",
        "post_training.sft.train_cli",
        "--dataset-dir",
        str(bundle),
        "--checkpoint-dir",
        str(profile.checkpoint_dir),
        "--sft-run-id",
        profile.sft_run_id,
        "--parent-repo-id",
        parent_repo,
        "--parent-run-id",
        profile.parent_run_id,
        "--parent-pointer",
        "best",
        "--checkpoint-repo-id",
        checkpoint_repo,
        "--device",
        "cuda",
        "--precision",
        "fp16",
        "--microbatch-size",
        str(profile.microbatch_size),
        "--learning-rate",
        str(profile.learning_rate),
        "--checkpoint-every-steps",
        str(profile.cadence_steps),
        "--evaluation-every-steps",
        str(profile.cadence_steps),
        "--remote-publish-every-steps",
        str(profile.cadence_steps),
        "--wandb-mode",
        "online",
        "--wandb-project",
        "Small-LLM",
        "--wandb-run-id",
        profile.wandb_run_id,
        "--wandb-run-name",
        profile.wandb_run_name,
    ]
    if entity:
        command += ["--wandb-entity", entity]
    if max_steps_this_session is not None:
        command += ["--max-steps-this-session", str(max_steps_this_session)]
    return _run(command, cwd=worktree)


def evaluate(
    profile: SFTProfileSpec,
    *,
    dataset_dir: str | None,
    eval_dir: str | None,
    parent_repo_id: str | None,
    checkpoint_repo_id: str | None,
    output: str | None,
    suite: str,
) -> int:
    worktree = _prepare_worktree(profile)
    bundle = _find_bundle(dataset_dir)
    parent_repo = parent_repo_id or os.environ.get("SMALL_LLM_HF_REPO_ID")
    checkpoint_repo = checkpoint_repo_id or os.environ.get("SMALL_LLM_SFT_HF_REPO_ID", parent_repo)
    if not parent_repo or not checkpoint_repo:
        raise RuntimeFailure("evaluation requires parent and SFT checkpoint repository IDs")
    selected_eval_dir = Path(eval_dir) if eval_dir else WORK / "eval_core_v1"
    selected_output = Path(output) if output else profile.run_root / f"post-sft-{suite}-qualification.json"
    return _run(
        _uv_prefix()
        + [
            "python",
            "-m",
            "post_training.sft.eval_suite",
            "--dataset-dir",
            str(bundle),
            "--eval-dir",
            str(selected_eval_dir),
            "--suite",
            suite,
            "--device",
            "cuda",
            "--precision",
            "fp16",
            "--output",
            str(selected_output),
            "--parent-repo-id",
            parent_repo,
            "--parent-run-id",
            profile.parent_run_id,
            "--parent-pointer",
            "best",
            "--sft-repo-id",
            checkpoint_repo,
            "--sft-run-id",
            profile.sft_run_id,
            "--sft-pointer",
            "latest",
        ],
        cwd=worktree,
    )


__all__ = [
    "PROFILES",
    "RuntimeFailure",
    "SFTProfileSpec",
    "evaluate",
    "prepare",
    "resolve_profile",
    "train",
]
