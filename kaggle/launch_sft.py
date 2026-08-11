#!/usr/bin/env python3
"""Single human entry point for Small-LLM supervised fine-tuning."""

from __future__ import annotations

import argparse
from dataclasses import replace
from decimal import Decimal, InvalidOperation
import json
import os
import re
from typing import Sequence

import sft_runtime

_QUANTITY = re.compile(r"^(\d+(?:\.\d+)?)([KMBT]?)$", re.IGNORECASE)
_MULTIPLIERS = {
    "": Decimal(1),
    "K": Decimal(1_000),
    "M": Decimal(1_000_000),
    "B": Decimal(1_000_000_000),
    "T": Decimal(1_000_000_000_000),
}

# Reachable main ancestor containing the complete operational SFT implementation.
_SFT_IMPLEMENTATION_COMMIT = "a47e3a875cfaf04ce06b8ff203bf96a18170bb2a"
_PARENT_CHECKPOINT_RUN_IDS = {
    (20_000_000, 500_000_000): "20m-500m-dataset-001",
    (20_000_000, 2_000_000_000): "20m-2b-dataset-001",
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


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _add_profile(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, type=parse_quantity, metavar="SIZE")
    parser.add_argument(
        "--tokens",
        "--pretrain-tokens",
        dest="tokens",
        required=True,
        type=parse_quantity,
        metavar="SIZE",
        help="nominal parent pretraining token profile, e.g. 500M or 2B",
    )
    parser.add_argument("--dry-run", action="store_true")


def _add_bundle_build_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--replay-root", required=True)
    parser.add_argument("--prepared-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--parent-consumed-tokens", type=positive_int)
    parser.add_argument("--revision")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "One profile-driven entry point for SFT data preparation/publication, "
            "training, verified resume, and comprehensive qualification."
        ),
        epilog="SFT resume is automatic: rerun the identical train command after interruption.",
    )
    subparsers = parser.add_subparsers(dest="action", required=True)

    prepare = subparsers.add_parser(
        "prepare",
        help="prepare pinned instruction data and build/verify the immutable 4%-scaled SFT bundle",
    )
    _add_profile(prepare)
    _add_bundle_build_arguments(prepare)

    publish = subparsers.add_parser(
        "publish",
        help="build, verify, privately publish, and round-trip the immutable SFT bundle",
    )
    _add_profile(publish)
    _add_bundle_build_arguments(publish)
    publish.add_argument("--kaggle-dataset-handle")
    publish.add_argument("--ops-dir")
    publish.add_argument("--force-upload", action="store_true")
    publish.add_argument("--remote-ready-timeout-seconds", type=positive_int, default=900)

    train = subparsers.add_parser("train", help="launch or exactly resume SFT")
    _add_profile(train)
    train.add_argument("--dataset-dir")
    train.add_argument("--parent-repo-id")
    train.add_argument("--checkpoint-repo-id")
    train.add_argument("--max-steps-this-session", type=positive_int)
    train.add_argument("--wandb-entity")

    evaluate = subparsers.add_parser(
        "eval",
        help="run the comprehensive parent-versus-SFT qualification scorecard",
    )
    _add_profile(evaluate)
    evaluate.add_argument("--dataset-dir")
    evaluate.add_argument("--eval-dir")
    evaluate.add_argument("--parent-repo-id")
    evaluate.add_argument("--checkpoint-repo-id")
    evaluate.add_argument("--output")
    evaluate.add_argument("--suite", choices=("fast", "full"), default="full")

    subparsers.add_parser("profiles", help="list registered SFT profiles")
    return parser


def _profile(args: argparse.Namespace) -> sft_runtime.SFTProfileSpec:
    try:
        profile = sft_runtime.resolve_profile(args.model, args.tokens)
    except sft_runtime.RuntimeFailure as error:
        raise ValueError(str(error)) from error
    key = (profile.model_parameters, profile.parent_training_tokens)
    try:
        parent_run_id = _PARENT_CHECKPOINT_RUN_IDS[key]
    except KeyError as error:
        raise ValueError(f"no canonical parent checkpoint namespace for {key}") from error
    return replace(
        profile,
        parent_run_id=parent_run_id,
        launch_commit=_SFT_IMPLEMENTATION_COMMIT,
    )


def _print_profiles() -> None:
    print("Supported SFT profiles:")
    for raw in sft_runtime.PROFILES.values():
        key = (raw.model_parameters, raw.parent_training_tokens)
        profile = replace(
            raw,
            parent_run_id=_PARENT_CHECKPOINT_RUN_IDS[key],
            launch_commit=_SFT_IMPLEMENTATION_COMMIT,
        )
        target = (
            str(profile.requested_sft_targets)
            if profile.requested_sft_targets is not None
            else "derived from verified final parent"
        )
        print(
            f"  model={profile.model_label:<4} parent_tokens={profile.token_label:<4} "
            f"parent_run={profile.parent_run_id} sft_run={profile.sft_run_id} targets={target}"
        )


def _dry_run(args: argparse.Namespace, profile: sft_runtime.SFTProfileSpec) -> dict[str, object]:
    forwarded = {
        key: value
        for key, value in vars(args).items()
        if key not in {"action", "model", "tokens", "dry_run"} and value is not None
    }
    return {
        "action": args.action,
        "model": profile.model_label,
        "parent_pretraining_tokens": profile.token_label,
        "parent_run_id": profile.parent_run_id,
        "sft_run_id": profile.sft_run_id,
        "sft_fraction": 0.04,
        "known_exact_parent_consumed_tokens": profile.known_parent_consumed_tokens,
        "requested_sft_targets": profile.requested_sft_targets,
        "microbatch_size": profile.microbatch_size,
        "cadence_steps": profile.cadence_steps,
        "learning_rate": profile.learning_rate,
        "launch_commit": profile.launch_commit,
        "resume": "automatic_verified",
        "arguments": forwarded,
    }


def _preflight_publish(parser: argparse.ArgumentParser, args: argparse.Namespace) -> None:
    has_handle = bool(
        args.kaggle_dataset_handle
        or os.environ.get("SMALL_LLM_SFT_KAGGLE_DATASET_HANDLE")
        or os.environ.get("KAGGLE_USERNAME")
    )
    if not has_handle:
        parser.error(
            "publish requires --kaggle-dataset-handle owner/dataset, "
            "SMALL_LLM_SFT_KAGGLE_DATASET_HANDLE, or KAGGLE_USERNAME"
        )
    if not os.environ.get("KAGGLE_API_TOKEN"):
        parser.error("publish requires KAGGLE_API_TOKEN in the process environment")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "profiles":
        _print_profiles()
        return 0
    try:
        profile = _profile(args)
    except ValueError as error:
        parser.error(str(error))

    if args.dry_run:
        print(json.dumps(_dry_run(args, profile), indent=2, sort_keys=True))
        return 0

    if args.action == "publish":
        _preflight_publish(parser, args)

    print(
        f"[launch] action={args.action} model={profile.model_label} "
        f"tokens={profile.token_label} resume=automatic_verified",
        flush=True,
    )

    if args.action == "prepare":
        return sft_runtime.prepare(
            profile,
            replay_root=args.replay_root,
            prepared_dir=args.prepared_dir,
            output_dir=args.output_dir,
            parent_consumed_tokens=args.parent_consumed_tokens,
            revision=args.revision,
        )
    if args.action == "publish":
        return sft_runtime.publish(
            profile,
            replay_root=args.replay_root,
            prepared_dir=args.prepared_dir,
            output_dir=args.output_dir,
            parent_consumed_tokens=args.parent_consumed_tokens,
            revision=args.revision,
            kaggle_dataset_handle=args.kaggle_dataset_handle,
            ops_dir=args.ops_dir,
            force_upload=args.force_upload,
            remote_ready_timeout_seconds=args.remote_ready_timeout_seconds,
        )
    if args.action == "train":
        return sft_runtime.train(
            profile,
            dataset_dir=args.dataset_dir,
            parent_repo_id=args.parent_repo_id,
            checkpoint_repo_id=args.checkpoint_repo_id,
            max_steps_this_session=args.max_steps_this_session,
            wandb_entity=args.wandb_entity,
        )
    return sft_runtime.evaluate(
        profile,
        dataset_dir=args.dataset_dir,
        eval_dir=args.eval_dir,
        parent_repo_id=args.parent_repo_id,
        checkpoint_repo_id=args.checkpoint_repo_id,
        output=args.output,
        suite=args.suite,
    )


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["build_parser", "main", "parse_quantity"]
