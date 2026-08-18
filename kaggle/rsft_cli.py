"""Canonical Kaggle CLI for reasoning supervised fine-tuning."""
from __future__ import annotations

import argparse
from typing import Sequence

import rsft_runtime
import sft_cli

parse_quantity = sft_cli.parse_quantity
positive_int = sft_cli.positive_int


def positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train Small-LLM R-SFT on Kaggle's qualified 2xTesla-T4 DDP path."
    )
    subs = parser.add_subparsers(dest="action", required=True)
    train = subs.add_parser("train")
    train.add_argument("--model", required=True, type=parse_quantity, metavar="SIZE")
    train.add_argument("--tokens", required=True, type=parse_quantity, metavar="SIZE")
    train.add_argument(
        "--delimiter-format",
        choices=("atomic", "textual"),
        required=True,
        help="matched R-SFT delimiter arm; this is the only required experiment choice",
    )
    train.add_argument(
        "--dataset-dir",
        help="optional prebuilt arm bundle; omit to build from the committed 630-example corpus",
    )
    train.add_argument(
        "--s0-bundle",
        help="optional completed S0 bundle override for the 10%% retention lane",
    )
    train.add_argument(
        "--run-id",
        help="optional stable run identity; defaults to the canonical pilot ID for the selected arm",
    )
    train.add_argument(
        "--token-spec",
        help="optional reasoning-token spec override; defaults to the frozen repository spec",
    )
    train.add_argument("--parent-repo-id")
    train.add_argument("--checkpoint-repo-id")
    train.add_argument("--wandb-entity")
    train.add_argument("--max-steps-this-session", type=positive_int)
    train.add_argument(
        "--learning-rate",
        type=positive_float,
        default=rsft_runtime.DEFAULT_LEARNING_RATE,
    )
    train.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    sft_cli._load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)
    run_id = args.run_id or rsft_runtime.default_run_id(args.delimiter_format)
    try:
        profile = rsft_runtime.resolve_profile(
            args.model,
            args.tokens,
            run_id=run_id,
            delimiter_format=args.delimiter_format,
            learning_rate=float(args.learning_rate),
        )
        print(
            f"[launch-rsft] action=train model={profile.model_label} tokens={profile.token_label} "
            f"parent={profile.parent_run_id} run={profile.sft_run_id} "
            f"delimiter={args.delimiter_format} topology=2xT4-DDP "
            f"data={'explicit' if args.dataset_dir else 'auto-committed-pilot'} "
            "bundle_exact_one_pass=true",
            flush=True,
        )
        return rsft_runtime.train(
            profile,
            dataset_dir=args.dataset_dir,
            delimiter_format=args.delimiter_format,
            token_spec=args.token_spec,
            s0_bundle=args.s0_bundle,
            parent_repo_id=args.parent_repo_id,
            checkpoint_repo_id=args.checkpoint_repo_id,
            max_steps_this_session=args.max_steps_this_session,
            wandb_entity=args.wandb_entity,
            dry_run=bool(args.dry_run),
        )
    except rsft_runtime.base.RuntimeFailure as error:
        parser.error(str(error))
    return 2


__all__ = ["build_parser", "main", "parse_quantity", "positive_float"]
