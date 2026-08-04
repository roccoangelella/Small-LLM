#!/usr/bin/env python3
"""Run the Small-LLM Kaggle qualification from an already-cloned repository.

Kaggle usage:
    %cd /kaggle/working/Small-LLM
    !python kaggle/run_20m_from_clone.py

This entrypoint keeps the controlling clone intact and creates an isolated,
detached Git worktree at the frozen launch commit for all evidence-producing
commands. It delegates the qualification stages to run_20m_one_click.py.
"""

from __future__ import annotations

import importlib.util
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

SOURCE_REPO = Path(__file__).resolve().parents[1]
LAUNCHER_PATH = Path(__file__).with_name("run_20m_one_click.py")
CONTROLLER_ROOT = Path("/kaggle/working/small-llm-qualification-controller")
LATEST_SUMMARY = Path("/kaggle/working/small_llm_qualification_summary.json")


def load_launcher():
    spec = importlib.util.spec_from_file_location(
        "small_llm_kaggle_launcher", LAUNCHER_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load repository launcher: {LAUNCHER_PATH}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepare_worktree(
    _unused_token: str,
    commit: str,
    repo: Path,
    evidence: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    try:
        top = Path(
            subprocess.check_output(
                ["git", "rev-parse", "--show-toplevel"],
                cwd=SOURCE_REPO,
                text=True,
                stderr=subprocess.STDOUT,
            ).strip()
        ).resolve()
    except (OSError, subprocess.CalledProcessError) as error:
        raise RuntimeError(
            "Run this file from a cloned Small-LLM repository."
        ) from error

    if top != SOURCE_REPO.resolve():
        raise RuntimeError(f"Repository root mismatch: expected {SOURCE_REPO}, got {top}")
    tracked_dirty = subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=no"],
        cwd=SOURCE_REPO,
        text=True,
    ).strip()
    if tracked_dirty:
        raise RuntimeError("The controlling clone has tracked modifications")

    controller_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=SOURCE_REPO, text=True
    ).strip()
    present = subprocess.run(
        ["git", "cat-file", "-e", f"{commit}^{{commit}}"],
        cwd=SOURCE_REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )
    if present.returncode:
        raise RuntimeError(
            f"Frozen launch commit {commit} is absent. Run git fetch origin first."
        )

    if repo.exists():
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(repo)],
            cwd=SOURCE_REPO,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )
        if repo.exists():
            shutil.rmtree(repo)
    subprocess.run(
        ["git", "worktree", "prune"],
        cwd=SOURCE_REPO,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    )

    stage = launcher.run(
        ["git", "worktree", "add", "--detach", str(repo), commit],
        name="git-launch-worktree",
        evidence=evidence,
        cwd=SOURCE_REPO,
    )
    actual = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True
    ).strip()
    dirty = subprocess.check_output(
        ["git", "status", "--porcelain"], cwd=repo, text=True
    ).strip()
    if actual != commit or dirty:
        raise RuntimeError(
            f"Exact launch worktree failed: actual={actual}, dirty={bool(dirty)}"
        )
    return {
        "requested": commit,
        "actual": actual,
        "clean": True,
        "detached": True,
        "controller_repo": str(SOURCE_REPO),
        "controller_head": controller_head,
        "launch_worktree": str(repo),
    }, [stage]


def repository_secret(name: str, required: bool = True) -> str | None:
    if name == "GITHUB_TOKEN":
        return "already-cloned-repository"
    return original_secret(name, required=required)


launcher = load_launcher()
original_secret = launcher.secret
CONTROLLER_ROOT.mkdir(parents=True, exist_ok=True)
launcher.WORK = CONTROLLER_ROOT
launcher.LATEST = LATEST_SUMMARY
launcher.clone_repo = prepare_worktree
launcher.secret = repository_secret


if __name__ == "__main__":
    raise SystemExit(launcher.main(sys.argv[1:]))
