#!/usr/bin/env python3
"""Single human entry point for Small-LLM dataset publication and training."""
from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import sys
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import runtime

_QUANTITY = re.compile(r"^(\d+(?:\.\d+)?)([KMBT]?)$", re.IGNORECASE)
_MULTIPLIERS = {
    "": Decimal(1),
    "K": Decimal(1_000),
    "M": Decimal(1_000_000),
    "B": Decimal(1_000_000_000),
    "T": Decimal(1_000_000_000_000),
}


def parse_quantity(value: str) -> int:
    compact = value.strip().replace("_", "").replace(",", "").replace(" ", "")
    match = _QUANTITY.fullmatch(compact)
    if match is None:
        raise argparse.ArgumentTypeError(
            f"invalid size {value!r}; use forms such as 20M, 500M, or 2B"
        )
    try:
        amount = Decimal(match.group(1)) * _MULTIPLIERS[match.group(2).upper()]
    except InvalidOperation as error:
        raise argparse.ArgumentTypeError(f"invalid size {value!r}") from error
    if amount <= 0 or amount != amount.to_integral_value():
        raise argparse.ArgumentTypeError(
            f"size must resolve to a positive whole number: {value!r}"
        )
    return int(amount)


def format_quantity(value: int) -> str:
    for suffix, scale in (
        ("T", 1_000_000_000_000),
        ("B", 1_000_000_000),
        ("M", 1_000_000),
        ("K", 1_000),
    ):
        if value >= scale and value % scale == 0:
            return f"{value // scale}{suffix}"
    return str(value)


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _add_profile_selector(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, type=parse_quantity, metavar="SIZE")
    parser.add_argument("--tokens", required=True, type=parse_quantity, metavar="SIZE")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--resume", action="store_true", help=argparse.SUPPRESS)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="One fail-closed entry point for dataset publication and Kaggle training.",
        epilog="Resume is automatic: rerun the identical command after interruption.",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    train = subparsers.add_parser("train", help="launch or exactly resume training")
    _add_profile_selector(train)
    train.add_argument("--dataset-dir")
    train.add_argument("--max-steps-this-session", type=positive_int)

    publish = subparsers.add_parser(
        "publish", help="build, verify, and privately publish the finite dataset"
    )
    _add_profile_selector(publish)
    publish.add_argument("--weights-file")
    publish.add_argument("--dataset-dir")
    publish.add_argument("--ops-dir")
    publish.add_argument("--kaggle-dataset-handle")
    publish.add_argument("--force-upload", action="store_true")
    publish.add_argument("--remote-ready-timeout-seconds", type=positive_int)

    qualify = subparsers.add_parser(
        "qualify-dual-t4",
        help="compare exact-batch two-T4 DDP with the single-T4 20M/2B path",
    )
    _add_profile_selector(qualify)
    qualify.add_argument("--dataset-dir")
    qualify.add_argument("--warmup-blocks", type=nonnegative_int)
    qualify.add_argument("--measure-blocks", type=positive_int)
    qualify.add_argument("--minimum-speedup", type=positive_float)
    qualify.add_argument("--output")

    subparsers.add_parser("profiles", help="list registered profiles")
    return parser


def resolve_profile(args: argparse.Namespace) -> runtime.ProfileSpec:
    try:
        return runtime.resolve_profile(args.model, args.tokens)
    except runtime.RuntimeFailure:
        supported = ", ".join(
            f"{profile.model_label}/{profile.token_label}"
            for profile in runtime.PROFILES.values()
        )
        raise ValueError(
            f"unsupported model/token profile "
            f"{format_quantity(args.model)}/{format_quantity(args.tokens)}; "
            f"supported profiles: {supported}"
        )


def _dry_run_payload(
    action: str,
    profile: runtime.ProfileSpec,
    args: argparse.Namespace,
) -> dict[str, object]:
    forwarded: dict[str, object] = {}
    for name in (
        "dataset_dir",
        "max_steps_this_session",
        "weights_file",
        "ops_dir",
        "kaggle_dataset_handle",
        "remote_ready_timeout_seconds",
        "warmup_blocks",
        "measure_blocks",
        "minimum_speedup",
        "output",
    ):
        if hasattr(args, name):
            value = getattr(args, name)
            if value is not None:
                forwarded[name] = value
    if getattr(args, "force_upload", False):
        forwarded["force_upload"] = True
    return {
        "action": action,
        "model": profile.model_label,
        "tokens": profile.token_label,
        "runtime": (
            "kaggle/qualify_dual_t4_watchdog.py"
            if action == "qualify-dual-t4"
            else "kaggle/runtime.py"
        ),
        "profile": profile.dataset_profile,
        "launch_commit": profile.launch_commit,
        "dataset_run_id": profile.dataset_run_id,
        "wandb_run_id": profile.wandb_run_id,
        "arguments": forwarded,
        "resume": "not_applicable" if action == "qualify-dual-t4" else "automatic_verified",
    }


def _print_profiles() -> None:
    print("Supported profiles:")
    for profile in runtime.PROFILES.values():
        print(
            f"  model={profile.model_label:<4} tokens={profile.token_label:<4} "
            f"profile={profile.dataset_profile}"
        )


def _dual_t4_arguments(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = []
    for name in ("dataset_dir", "warmup_blocks", "measure_blocks", "minimum_speedup", "output"):
        value = getattr(args, name, None)
        if value is not None:
            forwarded += ["--" + name.replace("_", "-"), str(value)]
    return forwarded


def _run_dual_t4_qualification(args: argparse.Namespace) -> int:
    uv = shutil.which("uv")
    if not uv:
        raise RuntimeError("uv is required for dual-T4 qualification")
    command = [
        uv,
        "run",
        "--python",
        "3.13",
        "--extra",
        "model",
        "python",
        str(REPO / "kaggle" / "qualify_dual_t4_watchdog.py"),
        *_dual_t4_arguments(args),
    ]
    return int(subprocess.call(command, cwd=REPO))


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.action == "profiles":
        _print_profiles()
        return 0
    if args.resume:
        parser.error(
            "--resume is intentionally unnecessary. Resume is fail-closed and automatic; "
            "rerun the exact same command after an interruption."
        )
    try:
        profile = resolve_profile(args)
    except ValueError as error:
        parser.error(str(error))

    if args.action == "qualify-dual-t4" and (
        profile.model_parameters != 20_000_000 or profile.training_tokens != 2_000_000_000
    ):
        parser.error("qualify-dual-t4 is currently qualified only for --model 20M --tokens 2B")

    if args.dry_run:
        print(
            json.dumps(
                _dry_run_payload(args.action, profile, args),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    if args.action == "publish":
        runtime.ensure_publication_environment(
            sys.argv[1:] if argv is None else list(argv)
        )

    if args.action == "qualify-dual-t4":
        print(
            f"[launch] action={args.action} model={profile.model_label} "
            f"tokens={profile.token_label} qualification=disposable",
            flush=True,
        )
        return _run_dual_t4_qualification(args)

    print(
        f"[launch] action={args.action} model={profile.model_label} "
        f"tokens={profile.token_label} resume=automatic_verified",
        flush=True,
    )

    if args.action == "train":
        return runtime.train(
            profile,
            dataset_dir=args.dataset_dir,
            max_steps_this_session=args.max_steps_this_session,
        )

    return runtime.publish(
        profile,
        weights_file=args.weights_file,
        dataset_dir=args.dataset_dir,
        ops_dir=args.ops_dir,
        kaggle_dataset_handle=args.kaggle_dataset_handle,
        force_upload=args.force_upload,
        remote_ready_timeout_seconds=args.remote_ready_timeout_seconds,
    )


if __name__ == "__main__":
    raise SystemExit(main())
