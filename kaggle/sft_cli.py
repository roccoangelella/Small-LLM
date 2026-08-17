"""Canonical SFT CLI implementation."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import os
from pathlib import Path
from typing import Sequence

import launch as pretraining_launch
import sft_runtime
import sft_scaled_runtime
from sft_100m import PROFILE as PROFILE_100M_2B

REPO = Path(__file__).resolve().parents[1]
LEGACY_IMPLEMENTATION_COMMIT = "806411edc1a93a32ce913e4e73b15452619f5579"
_PARENT_RUNS = {
    (20_000_000, 500_000_000): "20m-500m-dataset-001",
    (20_000_000, 2_000_000_000): "20m-2b-dataset-001",
}

parse_quantity = pretraining_launch.parse_quantity


def _load_dotenv(path: Path = REPO / ".env") -> None:
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
    evaluate.add_argument("--parent-checkpoint-dir")
    evaluate.add_argument("--sft-checkpoint-dir")
    evaluate.add_argument("--output")
    evaluate.add_argument("--suite", choices=("fast", "full"), default="full")
    evaluate.add_argument("--device", default="auto")
    evaluate.add_argument(
        "--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto"
    )
    evaluate.add_argument("--batch-size", type=positive_int, default=1)
    evaluate.add_argument("--validation-blocks", type=positive_int, default=32)
    evaluate.add_argument("--test-blocks", type=positive_int, default=32)

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
    if profile is PROFILE_100M_2B:
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
        "kaggle_training_topology": "2xT4-DDP" if profile is PROFILE_100M_2B else "single-cuda",
        "launch_commit": profile.launch_commit,
        "resume": "automatic_verified",
        "arguments": forwarded,
    }


def _profiles() -> int:
    print("Supported SFT profiles:")
    profiles = [
        resolve_profile(20_000_000, 500_000_000),
        resolve_profile(20_000_000, 2_000_000_000),
        PROFILE_100M_2B,
    ]
    for profile in profiles:
        fraction = profile.sft_fraction_numerator / profile.sft_fraction_denominator
        print(
            f"  model={profile.model_label:<4} parent_tokens={profile.token_label:<4} "
            f"parent_run={profile.parent_run_id} sft_run={profile.sft_run_id} "
            f"fraction={fraction:.0%} targets={profile.requested_sft_targets}"
        )
    return 0


def _discover_eval_dir(explicit: str | None) -> str | None:
    if explicit:
        return str(Path(explicit).expanduser().resolve())
    root = Path(os.environ.get("SMALL_LLM_INPUT_DIR", "/kaggle/input"))
    if root.is_dir():
        matches: set[Path] = set()
        for manifest in root.rglob("manifest.json"):
            try:
                payload = json.loads(manifest.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                continue
            if isinstance(payload, dict) and payload.get("name") == "eval_core_v1":
                matches.add(manifest.parent.resolve())
        if len(matches) > 1:
            raise sft_runtime.RuntimeFailure(
                f"expected at most one attached eval_core_v1 corpus; found {sorted(matches)}"
            )
        if matches:
            return str(next(iter(matches)))
    candidate = REPO / "tests" / "test_datasets" / "eval_core_v1"
    if candidate.is_dir() and (candidate / "manifest.json").is_file():
        return str(candidate.resolve())
    return None


def main(argv: Sequence[str] | None = None) -> int:
    _load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.action == "profiles":
        return _profiles()
    try:
        profile = resolve_profile(args.model, args.tokens)
    except sft_runtime.RuntimeFailure as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(dry_run_payload(args, profile), indent=2, sort_keys=True))
        return 0

    print(
        f"[launch] action={args.action} model={profile.model_label} "
        f"tokens={profile.token_label} resume=automatic_verified",
        flush=True,
    )
    runtime = runtime_for(profile)
    if args.action == "prepare":
        return runtime.prepare(
            profile,
            replay_root=args.replay_root,
            prepared_dir=args.prepared_dir,
            output_dir=args.output_dir,
            parent_consumed_tokens=args.parent_consumed_tokens,
            revision=args.revision,
        )
    if args.action == "publish":
        return runtime.publish(
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
        return runtime.train(
            profile,
            dataset_dir=args.dataset_dir,
            parent_repo_id=args.parent_repo_id,
            checkpoint_repo_id=args.checkpoint_repo_id,
            max_steps_this_session=args.max_steps_this_session,
            wandb_entity=args.wandb_entity,
        )
    try:
        eval_dir = _discover_eval_dir(args.eval_dir)
    except sft_runtime.RuntimeFailure as error:
        parser.error(str(error))
    return runtime.evaluate(
        profile,
        dataset_dir=args.dataset_dir,
        eval_dir=eval_dir,
        parent_repo_id=args.parent_repo_id,
        checkpoint_repo_id=args.checkpoint_repo_id,
        parent_checkpoint_dir=args.parent_checkpoint_dir,
        sft_checkpoint_dir=args.sft_checkpoint_dir,
        output=args.output,
        suite=args.suite,
        device=args.device,
        precision=args.precision,
        batch_size=args.batch_size,
        validation_blocks=args.validation_blocks,
        test_blocks=args.test_blocks,
    )


__all__ = [
    "build_parser",
    "dry_run_payload",
    "main",
    "parse_quantity",
    "resolve_profile",
    "runtime_for",
]
