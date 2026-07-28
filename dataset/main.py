"""Command-line entry point for the token-only Nemotron-ClimbMix corpus.

One obvious production command exists:

    uv run python -m dataset.main build
    uv run python -m dataset.main build --resume
    uv run python -m dataset.main status
    uv run python -m dataset.main verify

Bounded smoke/test overrides are allowed but do not change the frozen policy:

    uv run python -m dataset.main build \\
        --target-tokens 10000000 \\
        --max-work-items 20 \\
        --output-dir /tmp/climbmix-smoke
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from dataset import config


def configure_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _effective_config(args: argparse.Namespace) -> config.EffectiveConfig:
    get = lambda name, default: getattr(args, name, default)  # noqa: E731
    max_work_items = get("max_work_items", None)
    crash_after = get("simulate_crash_after_written_bytes", None)
    target = int(get("target_tokens", config.TARGET_ACCEPTED_SOURCE_TOKENS))
    # A bounded smoke/test run legitimately targets far fewer than the 80B
    # production minimum; the run's minimum is derived so such a corpus can
    # still be marked complete and verified.  The production default (90B)
    # keeps the full 80B floor.
    minimum = min(config.MINIMUM_ACCEPTED_SOURCE_TOKENS, target)
    overrides = {
        "output_dir": Path(get("output_dir", config.DEFAULT_OUTPUT_DIR)),
        "target_accepted_source_tokens": target,
        "minimum_accepted_source_tokens": minimum,
        "maximum_accepted_source_tokens": int(get("maximum_tokens", config.MAXIMUM_ACCEPTED_SOURCE_TOKENS)),
        "region_bytes": int(get("region_bytes", config.REGION_BYTES)),
        "writer_buffer_bytes": int(get("writer_buffer_bytes", config.WRITER_BUFFER_BYTES)),
        "checkpoint_bytes_threshold": int(get("checkpoint_bytes_threshold", config.CHECKPOINT_BYTES_THRESHOLD)),
        "max_work_items": int(max_work_items) if max_work_items is not None else None,
        "strict": bool(get("strict", False)),
        "allow_unsafe_low_disk": bool(get("allow_unsafe_low_disk", False)),
        "reset": bool(get("reset", False)),
        "full_scan": bool(get("full_scan", False)),
        "resume": bool(get("resume", False)),
        "crash_after_written_bytes": int(crash_after) if crash_after is not None else None,
    }
    _validate_overrides(overrides)
    return config.EffectiveConfig(**overrides)


def _validate_overrides(overrides: dict[str, object]) -> None:
    target = int(overrides["target_accepted_source_tokens"])
    if target <= 0:
        raise SystemExit("--target-tokens must be a positive integer")
    if int(overrides["region_bytes"]) <= 0:
        raise SystemExit("--region-bytes must be positive")
    if int(overrides["writer_buffer_bytes"]) <= 0:
        raise SystemExit("--writer-buffer-bytes must be positive")
    if int(overrides["checkpoint_bytes_threshold"]) <= 0:
        raise SystemExit("--checkpoint-bytes-threshold must be positive")
    max_work_items = overrides["max_work_items"]
    if max_work_items is not None and int(max_work_items) <= 0:
        raise SystemExit("--max-work-items must be positive")
    crash_after = overrides["crash_after_written_bytes"]
    if crash_after is not None and int(crash_after) <= 0:
        raise SystemExit("--simulate-crash-after-written-bytes must be positive")
    minimum = int(overrides["minimum_accepted_source_tokens"])
    maximum = int(overrides["maximum_accepted_source_tokens"])
    if target > maximum:
        raise SystemExit(
            f"--target-tokens {target} exceeds the maximum {maximum}"
        )
    if minimum > target:
        raise SystemExit(
            f"effective minimum {minimum} exceeds --target-tokens {target}"
        )
    if bool(overrides["resume"]) and bool(overrides["reset"]):
        raise SystemExit("--resume and --reset cannot be used together")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Build or verify a token-only, cluster-filtered Nemotron-ClimbMix corpus."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    build_parser = subcommands.add_parser(
        "build", help="Stream byte ranges, append little-endian uint16 tokens to train/val .bin."
    )
    build_parser.add_argument("--resume", action="store_true",
                             help="Resume from the crash-safe checkpoint in the output directory.")
    build_parser.add_argument("--reset", action="store_true",
                             help="Delete an existing uncheckpointed corpus before a fresh build.")
    build_parser.add_argument("--strict", action="store_true",
                             help="Abort immediately on any structurally invalid source record.")
    build_parser.add_argument("--allow-unsafe-low-disk", action="store_true",
                             help="Skip the free-disk-space and large-file preflight checks.")
    build_parser.add_argument("--output-dir", default=str(config.DEFAULT_OUTPUT_DIR),
                             help="Output directory (default: dataset/output).")
    build_parser.add_argument("--target-tokens", type=int,
                             default=config.TARGET_ACCEPTED_SOURCE_TOKENS,
                             help="Target accepted source tokens (default 90 B).")
    build_parser.add_argument("--maximum-tokens", type=int,
                             default=config.MAXIMUM_ACCEPTED_SOURCE_TOKENS,
                             help="Hard cap on accepted source tokens (default 100 B).")
    build_parser.add_argument("--max-work-items", type=int, default=None,
                             help="Bounded test: stop after this many work items.")
    build_parser.add_argument("--region-bytes", type=int, default=config.REGION_BYTES,
                             help="Logical byte region size for the work plan (default 256 MiB).")
    build_parser.add_argument("--writer-buffer-bytes", type=int, default=config.WRITER_BUFFER_BYTES,
                             help="In-memory write buffer per writer (default 256 MiB).")
    build_parser.add_argument("--checkpoint-bytes-threshold", type=int,
                             default=config.CHECKPOINT_BYTES_THRESHOLD,
                             help="Written bytes between durable checkpoints (default 1 GiB).")
    build_parser.add_argument(
        "--simulate-crash-after-written-bytes",
        type=int,
        default=None,
        help=(
            "Bounded smoke-test hook: stop after writing this many bytes, "
            "before committing the next checkpoint."
        ),
    )

    verify_parser = subcommands.add_parser(
        "verify", help="Validate manifest, hashes, sizes, and sampled token ranges."
    )
    verify_parser.add_argument("--output-dir", default=str(config.DEFAULT_OUTPUT_DIR),
                               help="Output directory to verify.")
    verify_parser.add_argument("--full-scan", action="store_true",
                               help="Scan every token instead of sampling (small corpora only).")

    status_parser = subcommands.add_parser(
        "status", help="Show durable state and the latest live progress snapshot."
    )
    status_parser.add_argument("--output-dir", default=str(config.DEFAULT_OUTPUT_DIR),
                               help="Output directory (default: dataset/output).")

    stream_cache_parser = subcommands.add_parser(
        "stream-cache",
        help="Validate stream-cache cluster weights JSON and print sequence geometry configuration.",
    )
    stream_cache_parser.add_argument(
        "--weights-file",
        type=str,
        required=True,
        help="Path to JSON file containing cluster weights mapping for accepted clusters 1-10 and 12-20.",
    )
    stream_cache_parser.add_argument(
        "--context-length",
        type=int,
        default=2048,
        help="Context length tokens (default: 2048).",
    )
    stream_cache_parser.add_argument(
        "--sequences-per-block",
        type=int,
        default=512,
        help="Sequences per block (default: 512).",
    )
    stream_cache_parser.add_argument(
        "--target-shard-bytes",
        type=int,
        default=1073741824,
        help="Target bytes per shard file (default: 1 GiB).",
    )
    stream_cache_parser.add_argument(
        "--reader-workers",
        type=int,
        default=4,
        help="Parallel reader threads (default: 4).",
    )
    stream_cache_parser.add_argument(
        "--max-in-flight-work-items",
        type=int,
        default=16,
        help="Maximum in-flight work items for parallel reader (default: 16).",
    )
    stream_cache_parser.add_argument(
        "--per-cluster-queue-limit",
        type=int,
        default=100,
        help="Per-cluster queue limit (default: 100).",
    )
    stream_cache_parser.add_argument(
        "--prepared-block-queue-limit",
        type=int,
        default=200,
        help="Prepared block queue limit (default: 200).",
    )
    stream_cache_parser.add_argument(
        "--prefetch-head-start",
        type=int,
        default=10,
        help="Prefetch head start (default: 10).",
    )
    stream_cache_parser.add_argument("--minimum-prefetched-source-tokens", type=int, default=1_000_000,
                                     help="Source-token head start before normal scheduling.")
    stream_cache_parser.add_argument("--minimum-populated-cluster-queues", type=int, default=2,
                                     help="Preferred populated queues before scheduling; never a hard all-cluster gate.")
    stream_cache_parser.add_argument("--maximum-rolling-mixture-error", type=float, default=1.0,
                                     help="Documented maximum normalized rolling mixture error before waiting briefly.")
    stream_cache_parser.add_argument("--maximum-waiting-documents", type=int, default=32,
                                     help="Bounded wait before scheduling from currently available queues.")
    stream_cache_parser.add_argument("--reader-batch-source-tokens", type=int, default=1_000_000,
                                     help="Maximum accepted source tokens retained per source-reader batch.")
    stream_cache_parser.add_argument("--reader-batch-documents", type=int, default=1000,
                                     help="Maximum documents retained per source-reader batch.")
    stream_cache_parser.add_argument("--reader-batch-max-bytes", type=int, default=16 * 1024 * 1024,
                                     help="Estimated maximum parsed bytes retained per source-reader batch.")
    stream_cache_parser.add_argument(
        "--scheduler-tie-break-seed",
        type=str,
        default=config.SELECTION_SEED,
        help="Tie break seed for scheduler (default: dataset.config.SELECTION_SEED).",
    )
    stream_cache_parser.add_argument(
        "--show-stream-config",
        action="store_true",
        help="Print normalized weights and sequence geometry JSON.",
    )

    args = parser.parse_args(argv)
    configure_logging()

    if args.command == "stream-cache":
        return _handle_stream_cache(args)

    effective = _effective_config(args)

    if args.command == "build":
        from dataset.src.build import build
        from dataset.src.exceptions import is_intentional_crash

        try:
            result = build(effective)
        except Exception as error:  # noqa: BLE001
            if is_intentional_crash(error):
                print(json.dumps({"intentional_crash": True, "message": str(error)}))
                return 0
            raise
        print(json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True, default=str))
        return 0

    if args.command == "verify":
        from dataset.src.verify import verify
        report = verify(Path(args.output_dir), full_scan=bool(args.full_scan))
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report.passed else 1

    if args.command == "status":
        from dataset.src.progress_report import format_status, status_report

        print(format_status(status_report(Path(args.output_dir))))
        return 0

    return 1


def _handle_stream_cache(args: argparse.Namespace) -> int:
    weights_path = Path(args.weights_file)
    if not weights_path.is_file():
        sys.stderr.write(f"Error: weights file not found: {weights_path}\n")
        return 1

    try:
        with weights_path.open("r", encoding="utf-8") as f:
            raw_weights = json.load(f)
    except Exception as error:
        sys.stderr.write(f"Error reading JSON from weights file {weights_path}: {error}\n")
        return 1

    if not isinstance(raw_weights, dict):
        sys.stderr.write(
            f"Error: weights file JSON must contain a top-level mapping/object, got {type(raw_weights).__name__}\n"
        )
        return 1

    from dataset.src.streaming import StreamCacheConfig, normalize_cluster_weights

    try:
        stream_cfg = StreamCacheConfig(
            context_length=args.context_length,
            sequences_per_block=args.sequences_per_block,
            target_shard_bytes=args.target_shard_bytes,
            reader_workers=args.reader_workers,
            max_in_flight_work_items=args.max_in_flight_work_items,
            per_cluster_queue_limit=args.per_cluster_queue_limit,
            prepared_block_queue_limit=args.prepared_block_queue_limit,
            prefetch_head_start=args.prefetch_head_start,
            weights=raw_weights,
            scheduler_tie_break_seed=args.scheduler_tie_break_seed,
            minimum_prefetched_source_tokens=args.minimum_prefetched_source_tokens,
            minimum_populated_cluster_queues=args.minimum_populated_cluster_queues,
            maximum_rolling_mixture_error=args.maximum_rolling_mixture_error,
            maximum_waiting_documents=args.maximum_waiting_documents,
            reader_batch_source_tokens=args.reader_batch_source_tokens,
            reader_batch_documents=args.reader_batch_documents,
            reader_batch_max_bytes=args.reader_batch_max_bytes,
        )
    except Exception as error:
        sys.stderr.write(f"StreamCacheConfig validation error: {error}\n")
        return 1

    supplied_weights, weight_units = normalize_cluster_weights(raw_weights)

    out_data = {
        "command": "stream-cache",
        "status": "validated",
        "notice": (
            "Legacy 'build' command is retained for prebuild monolithic binary format. "
            "No network calls or production streaming runs were started by this validation command."
        ),
        "stream_cache_config": {
            "context_length": stream_cfg.context_length,
            "stored_sequence_tokens": stream_cfg.stored_sequence_tokens,
            "sequences_per_block": stream_cfg.sequences_per_block,
            "block_bytes": stream_cfg.block_bytes,
            "target_shard_bytes": stream_cfg.target_shard_bytes,
            "weights": {
                "supplied": supplied_weights,
                "normalized_integer_units": {str(k): v for k, v in weight_units.items()},
            },
            "final_partial_sequence_policy": stream_cfg.final_partial_sequence_policy,
            "scheduler_tie_break_seed": stream_cfg.scheduler_tie_break_seed,
            "mixture_policy": {
                "minimum_prefetched_source_tokens": stream_cfg.minimum_prefetched_source_tokens,
                "minimum_populated_cluster_queues": stream_cfg.minimum_populated_cluster_queues,
                "maximum_rolling_mixture_error": stream_cfg.maximum_rolling_mixture_error,
                "maximum_waiting_documents": stream_cfg.maximum_waiting_documents,
            },
            "reader_batch_bounds": {
                "source_tokens": stream_cfg.reader_batch_source_tokens,
                "documents": stream_cfg.reader_batch_documents,
                "estimated_bytes": stream_cfg.reader_batch_max_bytes,
            },
        },
    }
    print(json.dumps(out_data, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
