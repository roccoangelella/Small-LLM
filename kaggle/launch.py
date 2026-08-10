#!/usr/bin/env python3
"""Unified entry point for Small-LLM dataset publication and Kaggle training.

Examples:
    python kaggle/launch.py publish --model 20M --tokens 2B
    python kaggle/launch.py train --model 20M --tokens 2B
    python kaggle/launch.py train --model 20M --tokens 2B --max-steps-this-session 250
    python kaggle/launch.py profiles

Publication and training are both resumable by rerunning the exact same command.
The profile-specific implementations remain fail-closed and own checkpoint,
dataset, W&B, and immutable launch-commit validation.
"""

from __future__ import annotations

import argparse
import importlib
import json
import re
import sys
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from pathlib import Path
from typing import Sequence

KAGGLE_DIR = Path(__file__).resolve().parent
if str(KAGGLE_DIR) not in sys.path:
    sys.path.insert(0, str(KAGGLE_DIR))


@dataclass(frozen=True, slots=True)
class Profile:
    model_parameters: int
    training_tokens: int
    train_module: str
    publish_module: str

    @property
    def model_label(self) -> str:
        return format_quantity(self.model_parameters)

    @property
    def token_label(self) -> str:
        return format_quantity(self.training_tokens)


PROFILES: dict[tuple[int, int], Profile] = {
    (20_000_000, 100_000_000): Profile(
        model_parameters=20_000_000,
        training_tokens=100_000_000,
        train_module="run_20m_100m",
        publish_module="build_and_push_100m_entry",
    ),
    (20_000_000, 500_000_000): Profile(
        model_parameters=20_000_000,
        training_tokens=500_000_000,
        train_module="run_20m_500m",
        publish_module="build_and_push_500m",
    ),
    (20_000_000, 2_000_000_000): Profile(
        model_parameters=20_000_000,
        training_tokens=2_000_000_000,
        train_module="run_20m_2b",
        publish_module="build_and_push_2b",
    ),
}

_QUANTITY = re.compile(r"^(\d+(?:\.\d+)?)([KMBT]?)$", re.IGNORECASE)
_MULTIPLIERS = {
    "": Decimal(1),
    "K": Decimal(1_000),
    "M": Decimal(1_000_000),
    "B": Decimal(1_000_000_000),
    "T": Decimal(1_000_000_000_000),
}


def parse_quantity(value: str) -> int:
    """Parse quantities such as 20M, 500m, 2B, or 2000M."""
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


def resolve_profile(model_parameters: int, training_tokens: int) -> Profile:
    try:
        return PROFILES[(model_parameters, training_tokens)]
    except KeyError as error:
        supported = ", ".join(
            f"{profile.model_label}/{profile.token_label}"
            for profile in PROFILES.values()
        )
        raise ValueError(
            f"unsupported model/token profile "
            f"{format_quantity(model_parameters)}/{format_quantity(training_tokens)}; "
            f"supported profiles: {supported}"
        ) from error


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _add_profile_selector(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--model",
        required=True,
        type=parse_quantity,
        metavar="SIZE",
        help="nominal model size, e.g. 20M",
    )
    parser.add_argument(
        "--tokens",
        required=True,
        type=parse_quantity,
        metavar="SIZE",
        help="training-token profile, e.g. 100M, 500M, or 2B",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="resolve and print the selected backend without executing it",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help=argparse.SUPPRESS,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One fail-closed entry point for finite-dataset publication and "
            "Kaggle training."
        ),
        epilog=(
            "Resume is automatic: after an interruption, rerun the identical "
            "publish or train command."
        ),
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    train = subparsers.add_parser(
        "train",
        help="launch or exactly resume a registered training profile",
    )
    _add_profile_selector(train)
    train.add_argument(
        "--dataset-dir",
        help="explicit attached dataset directory; normally auto-detected on Kaggle",
    )
    train.add_argument(
        "--max-steps-this-session",
        type=positive_int,
        help="diagnostic session cap; omit for normal finite-plan training",
    )

    publish = subparsers.add_parser(
        "publish",
        help="build, verify, and privately publish a registered finite dataset",
    )
    _add_profile_selector(publish)
    publish.add_argument("--weights-file", help="mixture weights JSON path")
    publish.add_argument("--dataset-dir", help="local producer output directory")
    publish.add_argument("--ops-dir", help="operations/evidence directory")
    publish.add_argument(
        "--kaggle-dataset-handle",
        help="explicit owner/dataset handle; otherwise profile environment defaults apply",
    )
    publish.add_argument(
        "--force-upload",
        action="store_true",
        help="intentionally publish a new Kaggle dataset version",
    )
    publish.add_argument(
        "--remote-ready-timeout-seconds",
        type=positive_int,
        help="Kaggle publication readiness timeout",
    )

    subparsers.add_parser("profiles", help="list registered model/token profiles")
    return parser


def _training_args(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = []
    if args.dataset_dir:
        forwarded += ["--dataset-dir", args.dataset_dir]
    if args.max_steps_this_session is not None:
        forwarded += ["--max-steps-this-session", str(args.max_steps_this_session)]
    return forwarded


def _publish_args(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = []
    for flag, attribute in (
        ("--weights-file", "weights_file"),
        ("--dataset-dir", "dataset_dir"),
        ("--ops-dir", "ops_dir"),
        ("--kaggle-dataset-handle", "kaggle_dataset_handle"),
    ):
        value = getattr(args, attribute)
        if value:
            forwarded += [flag, value]
    if args.force_upload:
        forwarded.append("--force-upload")
    if args.remote_ready_timeout_seconds is not None:
        forwarded += [
            "--remote-ready-timeout-seconds",
            str(args.remote_ready_timeout_seconds),
        ]
    return forwarded


def _dry_run_payload(
    action: str,
    profile: Profile,
    module_name: str,
    backend_argv: Sequence[str],
) -> dict[str, object]:
    return {
        "action": action,
        "model": profile.model_label,
        "tokens": profile.token_label,
        "backend_module": module_name,
        "backend_argv": list(backend_argv),
        "resume": "automatic_verified",
    }


def _dispatch(module_name: str, backend_argv: Sequence[str]) -> int:
    module = importlib.import_module(module_name)
    entry = getattr(module, "main", None)
    if not callable(entry):
        raise RuntimeError(f"{module_name} has no callable main()")
    result = entry(list(backend_argv))
    return 0 if result is None else int(result)


def _print_profiles() -> None:
    print("Supported profiles:")
    for profile in PROFILES.values():
        print(
            f"  model={profile.model_label:<4} tokens={profile.token_label:<4} "
            f"train={profile.train_module} publish={profile.publish_module}"
        )


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
        profile = resolve_profile(args.model, args.tokens)
    except ValueError as error:
        parser.error(str(error))

    if args.action == "train":
        module_name = profile.train_module
        backend_argv = _training_args(args)
    else:
        module_name = profile.publish_module
        backend_argv = _publish_args(args)

    if args.dry_run:
        print(
            json.dumps(
                _dry_run_payload(args.action, profile, module_name, backend_argv),
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(
        f"[launch] action={args.action} model={profile.model_label} "
        f"tokens={profile.token_label} resume=automatic_verified",
        flush=True,
    )
    return _dispatch(module_name, backend_argv)


if __name__ == "__main__":
    raise SystemExit(main())
