"""Build an immutable SFT bundle with an explicit parent-token fraction."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from .bundle import build_bundle, sft_budget_from_parent


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepared-dir", type=Path, required=True)
    parser.add_argument("--replay-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--parent-consumed-tokens", type=_positive_int, required=True)
    parser.add_argument("--fraction-numerator", type=_positive_int, required=True)
    parser.add_argument("--fraction-denominator", type=_positive_int, required=True)
    parser.add_argument("--optimizer-target-tokens", type=_positive_int, default=32768)
    parser.add_argument("--instruction-share", type=float, default=0.85)
    parser.add_argument("--replay-share", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=17)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    target = sft_budget_from_parent(
        args.parent_consumed_tokens,
        numerator=args.fraction_numerator,
        denominator=args.fraction_denominator,
    )
    bundle = build_bundle(
        args.prepared_dir,
        replay_root=args.replay_root,
        output_dir=args.output_dir,
        train_target_tokens=target,
        optimizer_target_tokens=args.optimizer_target_tokens,
        instruction_share=args.instruction_share,
        replay_share=args.replay_share,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "parent_consumed_tokens": args.parent_consumed_tokens,
                "sft_fraction": args.fraction_numerator / args.fraction_denominator,
                "requested_sft_targets": target,
                "bundle": bundle,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
