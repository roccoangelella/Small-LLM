"""Arguments and safety gates for the bounded trainer CLI."""

from __future__ import annotations

import argparse
from pathlib import Path


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Bounded smoke training on immutable schema-v2 shards."
    )
    p.add_argument("--dataset-dir", type=Path, required=True)
    p.add_argument(
        "--dataset-manifest",
        type=Path,
        help="Optional schema-v2 manifest or restored checkpoint drive_manifest.json.",
    )
    p.add_argument("--checkpoint-dir", type=Path, required=True)
    p.add_argument("--steps", type=int, required=True)
    p.add_argument("--resume")
    p.add_argument("--sequences-per-block", type=int)
    p.add_argument("--model-size", choices=("smoke", "substantive"), default="smoke")
    p.add_argument(
        "--architecture",
        choices=("gdn2_hybrid", "swa_hybrid", "all_mha"),
        default="gdn2_hybrid",
    )
    p.add_argument(
        "--gdn-chunk-size",
        type=int,
        help=(
            "GDN-2 training chunk. Trusted T4 FP16 runs default to the "
            "qualified size 32; other GDN-2 modes retain the model default 64."
        ),
    )
    p.add_argument(
        "--allow-unqualified-gdn2-chunk",
        action="store_true",
        help=(
            "Permit a non-32 GDN-2 chunk under FP16 for diagnostics only. "
            "Chunk 64 is not qualified for trusted T4 FP16 training."
        ),
    )
    p.add_argument("--initialization", choices=("normal", "xavier"), default="normal")
    p.add_argument("--device", default="cuda")
    p.add_argument("--precision", choices=("fp32", "fp16", "bf16"), default="fp16")
    p.add_argument(
        "--optimizer",
        choices=("hybrid_muon_adamw", "adamw"),
        default="hybrid_muon_adamw",
    )
    p.add_argument("--microbatch-size", type=int, default=1)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--muon-momentum", type=float, default=0.95)
    p.add_argument("--muon-lr-multiplier", type=float, default=1.0)
    p.add_argument("--muon-update-rms", type=float, default=0.18)
    p.add_argument("--muon-weight-decay", type=float, default=0.1)
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
    if args.gdn_chunk_size is not None and args.gdn_chunk_size <= 0:
        raise SystemExit("--gdn-chunk-size must be positive")

    if args.architecture == "gdn2_hybrid":
        if args.gdn_chunk_size is None:
            args.gdn_chunk_size = 32 if args.precision == "fp16" else 64
        if (
            args.precision == "fp16"
            and args.gdn_chunk_size != 32
            and not args.allow_unqualified_gdn2_chunk
        ):
            raise SystemExit(
                "trusted GDN-2 FP16 training requires the T4-qualified chunk size 32; "
                "use --allow-unqualified-gdn2-chunk only for a diagnostic run"
            )
    else:
        if args.gdn_chunk_size is not None:
            raise SystemExit("--gdn-chunk-size applies only to gdn2_hybrid")
        if args.allow_unqualified_gdn2_chunk:
            raise SystemExit(
                "--allow-unqualified-gdn2-chunk applies only to gdn2_hybrid"
            )
    return args
