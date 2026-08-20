"""Canonical Kaggle CLI for reasoning supervised fine-tuning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Sequence

import rsft_eval_runtime
import rsft_runtime
import sft_cli

parse_quantity = sft_cli.parse_quantity
positive_int = sft_cli.positive_int


def positive_float(value: str) -> float:
    result = float(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return result


def _profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, type=parse_quantity, metavar="SIZE")
    parser.add_argument("--tokens", required=True, type=parse_quantity, metavar="SIZE")
    parser.add_argument("--parent-repo-id")
    parser.add_argument("--checkpoint-repo-id")
    parser.add_argument("--wandb-entity")
    parser.add_argument("--max-steps-this-session", type=positive_int)
    parser.add_argument(
        "--learning-rate",
        type=positive_float,
        default=rsft_runtime.DEFAULT_LEARNING_RATE,
    )
    parser.add_argument(
        "--num-epochs",
        type=positive_int,
        default=1,
        help="number of exact passes over the frozen R-SFT train blocks (experimental when >1)",
    )
    parser.add_argument("--dry-run", action="store_true")


def _eval_profile_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", required=True, type=parse_quantity, metavar="SIZE")
    parser.add_argument("--tokens", required=True, type=parse_quantity, metavar="SIZE")
    parser.add_argument("--s0-bundle", help="completed S0 bundle; auto-discovered on Kaggle when omitted")
    parser.add_argument("--dataset-dir", help="production atomic R-SFT bundle; rebuilt and verified when omitted")
    parser.add_argument("--eval-dir", help="eval_core_v1 directory; auto-discovered on Kaggle when omitted")
    parser.add_argument("--parent-repo-id", help="S0 checkpoint repository override")
    parser.add_argument("--checkpoint-repo-id", help="R-SFT checkpoint repository override")
    parser.add_argument("--parent-checkpoint-dir", help="local S0 checkpoint override")
    parser.add_argument("--rsft-checkpoint-dir", help="local R-SFT checkpoint override")
    parser.add_argument("--output")
    parser.add_argument("--suite", choices=("fast", "full"), default="full")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=("auto", "fp32", "fp16", "bf16"), default="auto")
    parser.add_argument("--batch-size", type=positive_int, default=1)
    parser.add_argument("--validation-blocks", type=positive_int, default=32)
    parser.add_argument("--test-blocks", type=positive_int, default=32)
    parser.add_argument(
        "--reasoning-samples",
        type=positive_int,
        default=8,
        help="responses per novel reasoning problem for sampled pass@1 estimation",
    )
    parser.add_argument(
        "--reasoning-max-new-tokens",
        type=positive_int,
        default=256,
        help="maximum generated tokens for each reasoning-aware probe",
    )
    parser.add_argument("--dry-run", action="store_true")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train or qualify Small-LLM R-SFT on Kaggle."
    )
    subs = parser.add_subparsers(dest="action", required=True)

    train = subs.add_parser(
        "train",
        help="canonical production R-SFT: atomic special-token interface only",
    )
    _profile_args(train)
    train.add_argument(
        "--dataset-dir",
        help="optional prebuilt atomic-production-v1 bundle; omit to build from the committed 12,306-row Superior instruction checkpoint corpus",
    )
    train.add_argument(
        "--s0-bundle",
        help="optional completed S0 bundle override for the production 10%% retention lane",
    )
    train.add_argument(
        "--run-id",
        default=rsft_runtime.PRODUCTION_RUN_ID,
        help="stable production run identity",
    )
    train.add_argument(
        "--token-spec",
        help="optional explicit token spec; must exactly match the frozen production spec",
    )

    ablation = subs.add_parser(
        "ablation",
        help="historical 630-example delimiter experiment and explicit repeat probes",
    )
    _profile_args(ablation)
    ablation.add_argument(
        "--delimiter-format",
        choices=("atomic", "textual"),
        required=True,
        help="historical matched delimiter arm",
    )
    ablation.add_argument(
        "--dataset-dir",
        help="optional prebuilt pilot arm bundle; omit to use the committed 630-example corpus",
    )
    ablation.add_argument(
        "--s0-bundle",
        help="optional completed S0 bundle override for the 10%% retention lane",
    )
    ablation.add_argument(
        "--run-id",
        help="optional stable run identity; repeat probes get an epoch-specific ID automatically",
    )
    ablation.add_argument(
        "--token-spec",
        help="optional pilot token-spec override",
    )

    evaluate = subs.add_parser(
        "eval",
        help="S0-versus-production-R-SFT qualification with reasoning-aware probes",
    )
    _eval_profile_args(evaluate)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    sft_cli._load_dotenv()
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.action == "eval":
        try:
            profile = rsft_runtime.resolve_profile(
                args.model,
                args.tokens,
                run_id=rsft_runtime.PRODUCTION_RUN_ID,
                delimiter_format="atomic",
            )
            eval_dir = sft_cli._discover_eval_dir(args.eval_dir)
            if args.dry_run:
                payload = rsft_eval_runtime.evaluation_plan(
                    profile,
                    s0_bundle=args.s0_bundle,
                    dataset_dir=args.dataset_dir,
                    eval_dir=eval_dir,
                    parent_repo_id=args.parent_repo_id,
                    checkpoint_repo_id=args.checkpoint_repo_id,
                    parent_checkpoint_dir=args.parent_checkpoint_dir,
                    rsft_checkpoint_dir=args.rsft_checkpoint_dir,
                    output=args.output,
                    suite=args.suite,
                    device=args.device,
                    precision=args.precision,
                    batch_size=args.batch_size,
                    validation_blocks=args.validation_blocks,
                    test_blocks=args.test_blocks,
                    reasoning_samples=args.reasoning_samples,
                    reasoning_max_new_tokens=args.reasoning_max_new_tokens,
                )
                print(json.dumps(payload, indent=2, sort_keys=True))
                return 0
            print(
                f"[launch-rsft] action=eval model={profile.model_label} tokens={profile.token_label} "
                f"parent={profile.parent_run_id} run={profile.sft_run_id} suite={args.suite} "
                f"reasoning_samples={args.reasoning_samples}",
                flush=True,
            )
            return rsft_eval_runtime.evaluate(
                profile,
                s0_bundle=args.s0_bundle,
                dataset_dir=args.dataset_dir,
                eval_dir=eval_dir,
                parent_repo_id=args.parent_repo_id,
                checkpoint_repo_id=args.checkpoint_repo_id,
                parent_checkpoint_dir=args.parent_checkpoint_dir,
                rsft_checkpoint_dir=args.rsft_checkpoint_dir,
                output=args.output,
                suite=args.suite,
                device=args.device,
                precision=args.precision,
                batch_size=args.batch_size,
                validation_blocks=args.validation_blocks,
                test_blocks=args.test_blocks,
                reasoning_samples=args.reasoning_samples,
                reasoning_max_new_tokens=args.reasoning_max_new_tokens,
            )
        except rsft_runtime.base.RuntimeFailure as error:
            parser.error(str(error))
        return 2

    production = args.action == "train"
    num_epochs = int(args.num_epochs)
    delimiter_format = "atomic" if production else args.delimiter_format
    if production:
        run_id = args.run_id
    else:
        run_id = (
            args.run_id
            if args.run_id
            else rsft_runtime.default_experiment_run_id(delimiter_format, num_epochs=num_epochs)
        )
    try:
        profile = rsft_runtime.resolve_profile(
            args.model,
            args.tokens,
            run_id=run_id,
            delimiter_format=delimiter_format,
            learning_rate=float(args.learning_rate),
            num_epochs=num_epochs,
        )
        contract = "atomic-production-v1" if production else (
            "pilot-ablation-v1" if num_epochs == 1 else "pilot-repeat-v1"
        )
        print(
            f"[launch-rsft] action={args.action} model={profile.model_label} tokens={profile.token_label} "
            f"parent={profile.parent_run_id} run={profile.sft_run_id} "
            f"delimiter={delimiter_format} epochs={num_epochs} topology=2xT4-DDP "
            f"contract={contract} bundle_exact_passes={num_epochs}",
            flush=True,
        )
        return rsft_runtime.train(
            profile,
            dataset_dir=args.dataset_dir,
            delimiter_format=delimiter_format,
            token_spec=args.token_spec,
            s0_bundle=args.s0_bundle,
            parent_repo_id=args.parent_repo_id,
            checkpoint_repo_id=args.checkpoint_repo_id,
            max_steps_this_session=args.max_steps_this_session,
            wandb_entity=args.wandb_entity,
            num_epochs=num_epochs,
            production=production,
            dry_run=bool(args.dry_run),
        )
    except rsft_runtime.base.RuntimeFailure as error:
        parser.error(str(error))
    return 2


__all__ = ["build_parser", "main", "parse_quantity", "positive_float"]
