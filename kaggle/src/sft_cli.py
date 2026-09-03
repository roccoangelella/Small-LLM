"""Canonical SFT CLI implementation."""
from __future__ import annotations

import argparse
from dataclasses import replace
from fractions import Fraction
import json
import os
from pathlib import Path
from typing import Sequence

import launch as pretraining_launch
import sft_runtime
import sft_scaled_runtime
from sft_100m import PROFILE as PROFILE_100M_2B
from sft_100m_10b import PROFILE as PROFILE_100M_10B

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


def parse_sft_fraction(value: str) -> Fraction:
    raw = value.strip()
    if not raw:
        raise argparse.ArgumentTypeError("SFT fraction cannot be empty")
    try:
        fraction = Fraction(raw[:-1]) / 100 if raw.endswith("%") else Fraction(raw)
    except (ValueError, ZeroDivisionError) as error:
        raise argparse.ArgumentTypeError(
            "SFT fraction must be a ratio such as 1/5, a decimal such as 0.20, or a percentage such as 20%"
        ) from error
    if fraction <= 0 or fraction >= 1:
        raise argparse.ArgumentTypeError("SFT fraction must be strictly between 0 and 1")
    return fraction


def _fraction_label(fraction: Fraction) -> str:
    percent = fraction * 100
    if percent.denominator == 1:
        return f"{percent.numerator}pct"
    return f"{fraction.numerator}of{fraction.denominator}"


def _fraction_display(fraction: Fraction) -> str:
    percent = float(fraction * 100)
    return f"{percent:g}%"


def _variant_id(value: str, label: str) -> str:
    if value.endswith("-001"):
        return f"{value[:-4]}-{label}-001"
    return f"{value}-{label}"


def _recipe_ready(profile: sft_runtime.SFTProfileSpec) -> bool:
    return bool(getattr(profile, "recipe_ready", True))


def _parent_pointer(profile: sft_runtime.SFTProfileSpec) -> str:
    return str(getattr(profile, "parent_pointer", "best"))


def _parent_transport(profile: sft_runtime.SFTProfileSpec) -> str:
    return str(getattr(profile, "parent_transport", "model_repo"))


def with_sft_fraction(
    profile: sft_runtime.SFTProfileSpec,
    fraction: Fraction | None,
) -> sft_runtime.SFTProfileSpec:
    if fraction is None:
        return profile
    current = Fraction(profile.sft_fraction_numerator, profile.sft_fraction_denominator)
    if fraction == current:
        return profile
    dataset_label = _fraction_label(fraction)
    canonical_peak3000_10pct = (
        profile.model_parameters == 100_000_000
        and profile.parent_training_tokens == 2_000_000_000
        and fraction == Fraction(1, 10)
    )
    run_label = f"{dataset_label}-peak3000" if canonical_peak3000_10pct else dataset_label
    run_name_suffix = " / peak-through-3000" if canonical_peak3000_10pct else ""
    return replace(
        profile,
        sft_run_id=_variant_id(profile.sft_run_id, run_label),
        wandb_run_id=_variant_id(profile.wandb_run_id, run_label),
        wandb_run_name=(
            f"{profile.model_label} / {profile.token_label} parent / SFT S0 / "
            f"{_fraction_display(fraction)}{run_name_suffix}"
        ),
        # Reuse the already-published immutable 10% corpus. The scientific
        # trajectory gets a new run identity, not a new dataset identity.
        dataset_slug=_variant_id(profile.dataset_slug, dataset_label),
        sft_fraction_numerator=fraction.numerator,
        sft_fraction_denominator=fraction.denominator,
    )


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
    parser.add_argument(
        "--sft-fraction",
        type=parse_sft_fraction,
        help=(
            "override the profile SFT target budget as a parent-token fraction; "
            "accepts values such as 20%%, 0.20, or 1/5"
        ),
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
    if key == (100_000_000, 10_000_000_000):
        return PROFILE_100M_10B
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
        and profile.parent_training_tokens in {2_000_000_000, 10_000_000_000}
    ):
        return sft_scaled_runtime
    return sft_runtime


def dry_run_payload(
    args: argparse.Namespace, profile: sft_runtime.SFTProfileSpec
) -> dict[str, object]:
    forwarded = {
        key: value
        for key, value in vars(args).items()
        if key not in {"action", "model", "tokens", "dry_run", "sft_fraction"} and value is not None
    }
    recipe_ready = _recipe_ready(profile)
    return {
        "action": args.action,
        "model": profile.model_label,
        "parent_pretraining_tokens": profile.token_label,
        "parent_run_id": profile.parent_run_id,
        "parent_pointer": _parent_pointer(profile),
        "parent_transport": _parent_transport(profile),
        "sft_run_id": profile.sft_run_id,
        "dataset_slug": profile.dataset_slug,
        "recipe_status": "ready" if recipe_ready else "pending",
        "sft_fraction": (
            profile.sft_fraction_numerator / profile.sft_fraction_denominator
            if recipe_ready
            else None
        ),
        "known_exact_parent_consumed_tokens": profile.known_parent_consumed_tokens,
        "requested_sft_targets": profile.requested_sft_targets,
        "microbatch_size": profile.microbatch_size,
        "cadence_steps": profile.cadence_steps,
        "learning_rate": profile.learning_rate if recipe_ready else None,
        "kaggle_training_topology": "2xT4-DDP" if runtime_for(profile) is sft_scaled_runtime else "single-cuda",
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
        PROFILE_100M_10B,
    ]
    for profile in profiles:
        if _recipe_ready(profile):
            fraction = profile.sft_fraction_numerator / profile.sft_fraction_denominator
            fraction_text = f"{fraction:.0%}"
            targets_text = str(profile.requested_sft_targets)
        else:
            fraction_text = "pending"
            targets_text = "pending"
        print(
            f"  model={profile.model_label:<4} parent_tokens={profile.token_label:<4} "
            f"parent_run={profile.parent_run_id} sft_run={profile.sft_run_id} "
            f"fraction={fraction_text} targets={targets_text}"
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
        profile = with_sft_fraction(
            resolve_profile(args.model, args.tokens),
            args.sft_fraction,
        )
    except sft_runtime.RuntimeFailure as error:
        parser.error(str(error))
    if args.dry_run:
        print(json.dumps(dry_run_payload(args, profile), indent=2, sort_keys=True))
        return 0
    if not _recipe_ready(profile):
        parser.error(
            "100M/10B SFT parent wiring is registered, but ADR 0138 leaves the scientific "
            "recipe pending; prepare/publish/train/eval remain fail-closed until a later "
            "decision pins the target budget, data recipe, and LR schedule"
        )

    print(
        f"[launch] action={args.action} model={profile.model_label} "
        f"tokens={profile.token_label} sft_fraction="
        f"{profile.sft_fraction_numerator}/{profile.sft_fraction_denominator} "
        "resume=automatic_verified",
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
    "parse_sft_fraction",
    "resolve_profile",
    "runtime_for",
    "with_sft_fraction",
]
