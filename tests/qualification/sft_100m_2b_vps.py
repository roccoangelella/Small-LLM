"""Run the complete 100M/2B SFT qualification directly on a VPS.

This is the provider-neutral replacement for the former Kaggle-facing eval launch.
Large evaluation datasets live outside git under ``tests/test_datasets``.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Sequence

from dataset.eval_core import verify_eval_core
from post_training.sft.bundle import verify_bundle
from post_training.sft import eval_suite


REPO = Path(__file__).resolve().parents[2]
TEST_DATA_ROOT = REPO / "tests" / "test_datasets"
DEFAULT_SFT_BUNDLE = TEST_DATA_ROOT / "100m-2b-sft-s0-001"
DEFAULT_EVAL_CORE = TEST_DATA_ROOT / "eval_core_v1"
PARENT_RUN_ID = "100m-2b-data-001"
SFT_RUN_ID = "100m-2b-sft-s0-001"


def _positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _load_dotenv(path: Path = REPO / ".env") -> None:
    """Load simple KEY=VALUE entries without adding a runtime dependency.

    Existing process variables always win. Shell interpolation is deliberately not
    implemented; secrets and repository IDs should be literal values in ``.env``.
    """

    if not path.is_file():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, separator, value = line.partition("=")
        key = key.strip()
        if not separator or not key or not key.replace("_", "").isalnum():
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        os.environ.setdefault(key, value)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("fast", "full"), default="full")
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_SFT_BUNDLE)
    parser.add_argument("--eval-dir", type=Path, default=DEFAULT_EVAL_CORE)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repo-id", help="use one Hugging Face repository for parent and SFT")
    parser.add_argument("--parent-repo-id")
    parser.add_argument("--sft-repo-id")
    parser.add_argument("--parent-checkpoint-dir", type=Path)
    parser.add_argument("--sft-checkpoint-dir", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    parser.add_argument("--batch-size", type=_positive_int, default=1)
    parser.add_argument("--validation-blocks", type=_positive_int, default=32)
    parser.add_argument("--test-blocks", type=_positive_int, default=32)
    parser.add_argument("--bootstrap-samples", type=int, default=200)
    return parser


def _require_directory(path: Path, *, label: str, marker: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_dir() or not (resolved / marker).is_file():
        raise RuntimeError(
            f"{label} is missing at {resolved}; expected {marker}. "
            "See tests/test_datasets/README.md for the local VPS layout."
        )
    return resolved


def _repo_ids(args: argparse.Namespace) -> tuple[str | None, str | None]:
    # The completed 100M parent and its SFT checkpoints currently share one HF
    # qualification repository, so the SFT-specific variable is the safest common
    # default. Explicit CLI values still win.
    shared = (
        args.repo_id
        or os.environ.get("SMALL_LLM_SFT_HF_REPO_ID")
        or os.environ.get("SMALL_LLM_HF_REPO_ID")
    )
    parent = args.parent_repo_id or os.environ.get("SMALL_LLM_PARENT_HF_REPO_ID") or shared
    sft = args.sft_repo_id or os.environ.get("SMALL_LLM_SFT_HF_REPO_ID") or shared
    return parent, sft


def main(argv: Sequence[str] | None = None) -> int:
    _load_dotenv()
    args = build_parser().parse_args(argv)

    bundle = _require_directory(
        args.dataset_dir,
        label="100M/2B SFT bundle",
        marker="bundle-manifest.json",
    )
    eval_dir = _require_directory(
        args.eval_dir,
        label="eval_core_v1",
        marker="manifest.json",
    )

    print(f"Verifying local SFT bundle at {bundle}", flush=True)
    verify_bundle(bundle)
    print(f"Verifying local eval_core_v1 at {eval_dir}", flush=True)
    verify_eval_core(eval_dir)

    output = (
        args.output.expanduser().resolve()
        if args.output is not None
        else (REPO / "artifacts" / f"100m-2b-sft-{args.suite}-qualification.json").resolve()
    )
    parent_repo, sft_repo = _repo_ids(args)

    forwarded = [
        "--dataset-dir", str(bundle),
        "--eval-dir", str(eval_dir),
        "--suite", args.suite,
        "--device", args.device,
        "--precision", args.precision,
        "--batch-size", str(args.batch_size),
        "--validation-blocks", str(args.validation_blocks),
        "--test-blocks", str(args.test_blocks),
        "--bootstrap-samples", str(args.bootstrap_samples),
        "--output", str(output),
    ]

    if args.parent_checkpoint_dir is not None:
        forwarded += ["--parent-checkpoint-dir", str(args.parent_checkpoint_dir.expanduser().resolve())]
    else:
        if not parent_repo:
            raise RuntimeError(
                "set SMALL_LLM_SFT_HF_REPO_ID (or pass --repo-id/--parent-repo-id) "
                "or provide --parent-checkpoint-dir"
            )
        forwarded += [
            "--parent-repo-id", parent_repo,
            "--parent-run-id", PARENT_RUN_ID,
            "--parent-pointer", "best",
        ]

    if args.sft_checkpoint_dir is not None:
        forwarded += ["--sft-checkpoint-dir", str(args.sft_checkpoint_dir.expanduser().resolve())]
    else:
        if not sft_repo:
            raise RuntimeError(
                "set SMALL_LLM_SFT_HF_REPO_ID (or pass --repo-id/--sft-repo-id) "
                "or provide --sft-checkpoint-dir"
            )
        forwarded += [
            "--sft-repo-id", sft_repo,
            "--sft-run-id", SFT_RUN_ID,
            "--sft-pointer", "latest",
        ]

    print(
        f"Running {args.suite} 100M/2B SFT qualification on the VPS; output={output}",
        flush=True,
    )
    return eval_suite.main(forwarded)


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = [
    "DEFAULT_EVAL_CORE",
    "DEFAULT_SFT_BUNDLE",
    "PARENT_RUN_ID",
    "SFT_RUN_ID",
    "TEST_DATA_ROOT",
    "build_parser",
    "main",
]
