"""Canonical SFT CLI implementation."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
from typing import Sequence

import launch as pretraining_launch
import sft_runtime
import sft_scaled_runtime
from sft_100m import PROFILE as PROFILE_100M_2B

LEGACY_IMPLEMENTATION_COMMIT = "806411edc1a93a32ce913e4e73b15452619f5579"
_PARENT_RUNS = {
    (20_000_000, 500_000_000): "20m-500m-dataset-001",
    (20_000_000, 2_000_000_000): "20m-2b-dataset-001",
}

parse_quantity = pretraining_launch.parse_quantity


def positive_int(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, type=parse_quantity, metavar="SIZE")
    parser.add_argument(
        "--tokens",
        "--pretrain-tokens",
        dest="tokens",
        required=True,
        type=parse_quantity,
        metavar="SIZE",
    )
    parser.add_argument("--dry-run", action="store_true")


def _bundle_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--replay-root", required=True)
    parser.add_argument("--prepared-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--parent-consumed-tokens", type=positive_int)
    parser.add_argument("--revision")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Profile-driven Small-LLM supervised fine-tuning launcher."
    )
    subs = parser.add_subparsers(dest="action", required=True)

    prepare = subs.add_parser("prepare")
    _profile_args(prepare)
    _bundle_args(prepare)

    publish = subs.add_parser("publish")
    _profile_args(publish)
    _bundle_args(publish)
    publish.add_argument("--kaggle-dataset-handle")
    publish.add_argument("--ops-dir")
    publish.add_argument("--force-upload", action="store_true")
    publish.add_argument("--remote-ready-timeout-seconds", type=positive_int, default=900)

    train = subs.add_parser("train")
    _profile_args(train)
    train.add_argument("--dataset-dir")
    train.add_argument("--parent-repo-id")
    train.add_argument("--checkpoint-repo-id")
    train.add_argument("--max-steps-this-session", type=positive_int)
    train.add_argument("--wandb-entity")

    evaluate = subs.add_parser("eval")
    _profile_args(evaluate)
    evaluate.add_argument("--dataset-dir")
    evaluate.add_argument("--eval-dir")
    evaluate.add_argument("--parent-repo-id")
    evaluate.add_argument("--checkpoint-repo-id")
    evaluate.add_argument("--output")
    evaluate.add_argument("--suite", choices=("fast", "full"), default="full")

    subs.add_parser("profiles")
    return parser


def resolve_profile(model: int, tokens: int) -> sft_runtime.SFTProfileSpec:
    key = (model, tokens)
    if key == (100_000_000, 2_000_000_000):
        return PROFILE_100M_2B
    raw = sft_runtime.resolve_profile(model, tokens)
    try:
        parent_run = _PARENT_RUNS[key]
    except KeyError as error:
        raise sft_runtime.RuntimeFailure(
            f"no canonical parent checkpoint namespace for {key}"
        ) from error
    return replace(
        raw,
        parent_run_id=parent_run,
        launch_commit=LEGACY_IMPLEMENTATION_COMMIT,
    )


def runtime_for(profile: sft_runtime.SFTProfileSpec):
    if (
        profile.model_parameters == 100_000_000
        and profile.parent_training_tokens == 2_000_000_000
    ):
        return sft_scaled_runtime
    return sft_runtime


def dry_run_payload(
    args: argparse.Namespace, profile: sft_runtime.SFTProfileSpec
) -> dict[str, object]:
    forwarded = {
        key: value
        for key, value in vars(args).items()
        if key not in {"action", "model", "tokens", "dry_run"} and value is not None
    }
    dual_t4 = profile is PROFILE_100M_2B
    return {
        "action": args.action,
        "model": profile.model_label,
        "parent_pretraining_tokens": profile.token_label,
        "parent_run_id": profile.parent_run_id,
        "sft_run_id": profile.sft_run_id,
        "sft_fraction": profile.sft_fraction_numerator / profile.sft_fraction_denominator,
        "known_exact_parent_consumed_tokens": profile.known_parent_consumed_tokens,
        "requested_sft_targets": profile.requested_sft_targets,
        "microbatch_size": profile.microbatch_size,
        "cadence_steps": profile.cadence_steps,
        "learning_rate": profile.learning_rate,
        "kaggle_training_topology": "2xT4-DDP" if dual_t4 else "single-cuda",
        "launch_commit": profile.launch_commit,
        "resume": "automatic_verified",
        "arguments": forwarded,
    }


__all__ = [
    "build_parser",
    "dry_run_payload",
    "parse_quantity",
    "resolve_profile",
    "runtime_for",
]
