"""Explicit MoE process entrypoint; shares the canonical training loop."""
from __future__ import annotations

import math

from trainer.cli import main as train
from trainer.cli_args import parse_args as parse_trainer_args
from trainer.cli_args import parser as trainer_parser

from .setup import setup, validation_reader


def parse_args(argv=None):
    parser = trainer_parser()
    parser.add_argument("--load-balancing", choices=("none", "loss_free_sign"), default="none")
    parser.add_argument("--balancing-step-size", type=float, default=0.0)
    args = parse_trainer_args(argv, argument_parser=parser)
    if not math.isfinite(args.balancing_step_size) or args.balancing_step_size < 0:
        parser.error("--balancing-step-size must be finite and non-negative")
    if args.load_balancing == "none" and args.balancing_step_size:
        parser.error("M0 requires --balancing-step-size 0")
    return args


def main(argv=None):
    return train(argv, setup_fn=setup, validation_reader_fn=validation_reader,
                 parse_args_fn=parse_args)


if __name__ == "__main__":
    raise SystemExit(main())
