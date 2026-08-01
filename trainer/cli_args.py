"""Arguments and safety gates for the bounded trainer CLI."""
from __future__ import annotations
import argparse
from pathlib import Path

def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bounded smoke training on immutable schema-v2 shards.")
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument("--dataset-manifest", type=Path,
        help="Optional schema-v2 manifest or restored checkpoint drive_manifest.json.")
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--resume")
    p.add_argument("--sequences-per-block", type=int)
    p.add_argument("--model-size", choices=("smoke", "substantive"), default="smoke")
    p.add_argument("--architecture", choices=("gdn2_hybrid", "swa_hybrid", "all_mha"),
                   default="gdn2_hybrid")
    p.add_argument("--allow-unqualified-gdn2", action="store_true",
        help="Permit diagnostic GDN-2 execution despite the recorded T4 parity failure.")
    p.add_argument("--initialization", choices=("normal", "xavier"), default="normal")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    p.add_argument("--microbatch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--schedule", choices=("constant", "wsd"), default="constant")
    p.add_argument("--warmup-tokens", type=int, default=0)
    p.add_argument("--stable-tokens", type=int, default=0)
    p.add_argument("--decay-tokens", type=int, default=0)
    p.add_argument("--minimum-lr-ratio", type=float, default=0.1)
    p.add_argument("--checkpoint-every-steps", type=int, default=0)
    p.add_argument("--evaluation-every-steps", type=int, default=0)
    p.add_argument("--validation-blocks", type=int, default=0)
    p.add_argument("--seed", type=int, default=17)
    return p

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    args = parser().parse_args(argv)
    if args.steps <= 0:
        raise SystemExit("--steps must be positive")
    if args.validation_blocks < 0:
        raise SystemExit("--validation-blocks cannot be negative")
    if args.architecture == "gdn2_hybrid" and not args.allow_unqualified_gdn2:
        raise SystemExit("GDN-2 is blocked for trusted pretraining by the recorded T4 parity defect; "
            "use --architecture swa_hybrid for trainer qualification, or add "
            "--allow-unqualified-gdn2 only to reproduce or diagnose that defect")
    return args
