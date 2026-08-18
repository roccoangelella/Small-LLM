#!/usr/bin/env python3
"""Resolve the S0 retention bundle and build the committed R-SFT pilot on Kaggle.

The first R0 corpus itself is versioned in git.  Kaggle only needs to resolve the
already-published S0 bundle used for the frozen 10% instruction-retention lane,
then materialize the matched atomic/textual native SFT bundles in /kaggle/working.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
from typing import Mapping

import sft_runtime as base

DEFAULT_S0_KAGGLE_HANDLE = "roccoangelella/small-llm-100m-2b-sft-s0-001"
DEFAULT_MATCHED_BUNDLE_NAME = "rsft-r0-pilot-630-bundles"
REASONING_RELATIVE_PATH = Path("artifacts/rsft-r0-pilot-630/generation/reasoning.jsonl")
TOKEN_SPEC_RELATIVE_PATH = Path("post_training/R-SFT/reasoning-tokens.json")
PILOT_OPTIMIZER_TARGET_TOKENS = 2_048


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, TypeError, ValueError) as error:
        raise base.RuntimeFailure(f"{label} is missing or invalid: {path}") from error
    if not isinstance(payload, Mapping):
        raise base.RuntimeFailure(f"{label} must be a JSON object: {path}")
    return dict(payload)


def _looks_like_s0_bundle(root: Path) -> bool:
    manifest_path = root / "bundle-manifest.json"
    if not manifest_path.is_file() or manifest_path.is_symlink():
        return False
    try:
        payload = _read_json(manifest_path, label="S0 bundle manifest")
    except base.RuntimeFailure:
        return False
    if payload.get("schema") != "small-llm-sft-bundle" or "rsft" in payload:
        return False
    prepared = payload.get("prepared_source")
    return (
        isinstance(prepared, Mapping)
        and prepared.get("dataset_name") == "HuggingFaceTB/smol-smoltalk"
        and payload.get("instruction_share") == 0.85
        and payload.get("replay_share") == 0.15
    )


def _find_s0_bundles(root: Path) -> tuple[Path, ...]:
    if not root.is_dir():
        return ()
    candidates: set[Path] = set()
    if _looks_like_s0_bundle(root):
        candidates.add(root.resolve())
    for manifest in root.rglob("bundle-manifest.json"):
        candidate = manifest.parent
        if _looks_like_s0_bundle(candidate):
            candidates.add(candidate.resolve())
    return tuple(sorted(candidates))


def _require_single_s0_bundle(root: Path, *, label: str) -> Path:
    matches = _find_s0_bundles(root)
    if len(matches) != 1:
        raise base.RuntimeFailure(
            f"expected exactly one completed 100M/2B S0 bundle under {label}; "
            f"found {len(matches)}: {list(matches)}"
        )
    return matches[0]


def _download_s0_bundle(*, worktree: Path, handle: str) -> Path:
    script = (
        "import kagglehub,sys; "
        "path=kagglehub.dataset_download(sys.argv[1]); "
        "print('__SMALL_LLM_S0_PATH__=' + str(path))"
    )
    command = [
        *base._uv_prefix(kagglehub=True),
        "python",
        "-c",
        script,
        handle,
    ]
    print("$ " + " ".join(command), flush=True)
    result = subprocess.run(
        command,
        cwd=worktree,
        env={**os.environ, "PYTHONUNBUFFERED": "1", "UV_LINK_MODE": "copy"},
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.stdout:
        print(result.stdout, end="" if result.stdout.endswith("\n") else "\n", flush=True)
    if result.returncode:
        raise base.RuntimeFailure(
            f"failed to download private S0 Kaggle dataset {handle!r}; "
            "attach it to the notebook or pass --s0-bundle explicitly"
        )
    prefix = "__SMALL_LLM_S0_PATH__="
    resolved = None
    for line in result.stdout.splitlines():
        if line.startswith(prefix):
            resolved = Path(line[len(prefix) :].strip()).expanduser().resolve()
    if resolved is None:
        raise base.RuntimeFailure("kagglehub did not report the downloaded S0 dataset path")
    return _require_single_s0_bundle(resolved, label=f"downloaded Kaggle dataset {handle!r}")


def resolve_s0_bundle(explicit: str | None, *, worktree: Path) -> Path:
    """Resolve the exact completed S0 bundle used for the retention lane.

    Order: explicit path -> already attached /kaggle/input bundle -> private
    Kaggle dataset download using the frozen S0 dataset handle.
    """

    if explicit:
        root = Path(explicit).expanduser().resolve()
        if not _looks_like_s0_bundle(root):
            raise base.RuntimeFailure(f"--s0-bundle is not the completed S0 bundle: {root}")
        return root

    attached = _find_s0_bundles(base.INPUT)
    if len(attached) == 1:
        return attached[0]
    if len(attached) > 1:
        raise base.RuntimeFailure(
            f"multiple S0-shaped bundles are attached under {base.INPUT}; pass --s0-bundle explicitly: "
            f"{list(attached)}"
        )

    handle = os.environ.get("SMALL_LLM_S0_KAGGLE_DATASET_HANDLE", DEFAULT_S0_KAGGLE_HANDLE)
    return _download_s0_bundle(worktree=worktree, handle=handle)


def default_matched_bundle_root() -> Path:
    return (base.WORK / DEFAULT_MATCHED_BUNDLE_NAME).resolve()


def preparation_plan(*, worktree: Path, s0_bundle: str | None = None) -> dict[str, object]:
    root = default_matched_bundle_root()
    return {
        "reasoning_jsonl": str((worktree / REASONING_RELATIVE_PATH).resolve()),
        "reasoning_token_spec": str((worktree / TOKEN_SPEC_RELATIVE_PATH).resolve()),
        "s0_bundle": str(Path(s0_bundle).expanduser().resolve()) if s0_bundle else "auto:attached-or-private-kaggle",
        "s0_kaggle_handle": os.environ.get(
            "SMALL_LLM_S0_KAGGLE_DATASET_HANDLE", DEFAULT_S0_KAGGLE_HANDLE
        ),
        "matched_bundle_root": str(root),
        "atomic_bundle": str(root / "atomic"),
        "textual_bundle": str(root / "textual"),
        "optimizer_target_tokens": PILOT_OPTIMIZER_TARGET_TOKENS,
        "passes": 1,
    }


def prepare_pilot_bundles(
    *,
    worktree: Path,
    s0_bundle: str | None = None,
    output_root: Path | None = None,
) -> Path:
    """Build or verify the matched pilot bundles from the committed corpus."""

    reasoning = (worktree / REASONING_RELATIVE_PATH).resolve()
    token_spec = (worktree / TOKEN_SPEC_RELATIVE_PATH).resolve()
    produce = (worktree / "post_training" / "R-SFT" / "produce.py").resolve()
    if not reasoning.is_file() or reasoning.is_symlink():
        raise base.RuntimeFailure(f"pinned worktree has no safe committed reasoning corpus: {reasoning}")
    if not token_spec.is_file() or token_spec.is_symlink():
        raise base.RuntimeFailure(f"pinned worktree has no safe reasoning-token spec: {token_spec}")

    output = (output_root or default_matched_bundle_root()).expanduser().resolve()
    pilot_manifest = output / "pilot-manifest.json"
    verify_command = [
        *base._uv_prefix(),
        "python",
        str(produce),
        "verify",
        "--dataset-dir",
        str(output),
    ]
    if pilot_manifest.is_file():
        base._run(verify_command, cwd=worktree)
        return output
    if output.exists():
        raise base.RuntimeFailure(
            f"refusing to replace incomplete R-SFT pilot output: {output}; remove it explicitly first"
        )

    source_bundle = resolve_s0_bundle(s0_bundle, worktree=worktree)
    build_command = [
        *base._uv_prefix(),
        "python",
        str(produce),
        "build",
        "--reasoning-jsonl",
        str(reasoning),
        "--s0-bundle",
        str(source_bundle),
        "--token-spec",
        str(token_spec),
        "--output-dir",
        str(output),
        "--optimizer-target-tokens",
        str(PILOT_OPTIMIZER_TARGET_TOKENS),
    ]
    base._run(build_command, cwd=worktree)
    base._run(verify_command, cwd=worktree)
    return output


__all__ = [
    "DEFAULT_MATCHED_BUNDLE_NAME",
    "DEFAULT_S0_KAGGLE_HANDLE",
    "PILOT_OPTIMIZER_TARGET_TOKENS",
    "REASONING_RELATIVE_PATH",
    "TOKEN_SPEC_RELATIVE_PATH",
    "default_matched_bundle_root",
    "preparation_plan",
    "prepare_pilot_bundles",
    "resolve_s0_bundle",
]
