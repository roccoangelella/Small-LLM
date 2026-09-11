"""Explicit MoE process entrypoint; shares the canonical training loop."""
from __future__ import annotations

import math

from trainer.cli import main as train
from trainer.cli_args import parse_args as parse_trainer_args
from trainer.cli_args import parser as trainer_parser

from .setup import setup, validation_reader


def _enable_accepted_model_size(parser) -> None:
    """Extend only the MoE CLI without widening the dense trainer's model choices."""

    model_size_action = next(
        action for action in parser._actions if getattr(action, "dest", None) == "model_size"
    )
    model_size_action.choices = ("smoke", "substantive", "accepted")


def parse_args(argv=None):
    parser = trainer_parser()
    _enable_accepted_model_size(parser)
    parser.add_argument(
        "--load-balancing",
        choices=("none", "loss_free_sign", "quantile"),
        default=None,
        help=(
            "Optional experimental controller override. The accepted production model "
            "intrinsically uses quantile balancing and cannot be downgraded here."
        ),
    )
    parser.add_argument("--balancing-step-size", type=float, default=0.0)
    args = parse_trainer_args(argv, argument_parser=parser)
    if not math.isfinite(args.balancing_step_size) or args.balancing_step_size < 0:
        parser.error("--balancing-step-size must be finite and non-negative")
    if args.balancing_step_size and args.load_balancing != "loss_free_sign":
        parser.error("--balancing-step-size requires --load-balancing loss_free_sign")
    if args.model_size == "accepted":
        if args.load_balancing not in {None, "quantile"}:
            parser.error(
                "--model-size accepted is frozen to Quantile Balancing; "
                "omit --load-balancing or pass quantile"
            )
        if args.balancing_step_size:
            parser.error("accepted Quantile Balancing does not use --balancing-step-size")
    elif args.load_balancing == "quantile":
        parser.error("--load-balancing quantile requires --model-size accepted")
    return args


def main(argv=None):
    return train(argv, setup_fn=setup, validation_reader_fn=validation_reader,
                 parse_args_fn=parse_args)


if __name__ == "__main__":
    raise SystemExit(main())
