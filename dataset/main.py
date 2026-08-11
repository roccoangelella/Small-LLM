"""Low-level schema-v2 verification and stream-cache development CLI.

Finite experiment datasets are produced through ``dataset.qualification``.
This module intentionally keeps only shared verification and the low-level
stream-cache surface used by implementation tests.
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Verify a schema-v2 dataset or exercise the low-level stream-cache builder."
    )
    subcommands = parser.add_subparsers(dest="command", required=True)

    verify_parser = subcommands.add_parser(
        "verify", help="Validate manifest, hashes, sizes, and sampled token ranges."
    )
    verify_parser.add_argument("--output-dir", default=str(config.DEFAULT_OUTPUT_DIR),
                               help="Output directory to verify.")
    verify_parser.add_argument("--full-scan", action="store_true",
                               help="Scan every token instead of sampling (small corpora only).")

    stream_cache_parser = subcommands.add_parser(
        "stream-cache",
        help="Build/resume a schema-v2 stream cache, or validate its configuration.",
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
    stream_cache_parser.add_argument(
        "--build",
        action="store_true",
        help="Resolve the pinned source and build the stream cache.",
    )
    stream_cache_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume a stream-cache build from its durable source-reader cursor.",
    )
    stream_cache_parser.add_argument(
        "--output-dir",
        default=str(config.DEFAULT_OUTPUT_DIR),
        help="Stream-cache output directory (default: dataset/output).",
    )
    stream_cache_parser.add_argument(
        "--checkpoint-every-documents",
        type=int,
        default=1,
        help="Durably checkpoint the cache and source-reader cursor after this many documents (default: 1).",
    )

    args = parser.parse_args(argv)
    configure_logging()

    if args.command == "stream-cache":
        return _handle_stream_cache(args)

    if args.command == "verify":
        from dataset.src.verify import verify

        report = verify(Path(args.output_dir), full_scan=bool(args.full_scan))
        print(json.dumps(report.as_dict(), ensure_ascii=False, indent=2, sort_keys=True))
        return 0 if report.passed else 1

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

    if args.resume and not args.build:
        sys.stderr.write("Error: --resume requires --build\n")
        return 1
    if args.checkpoint_every_documents <= 0:
        sys.stderr.write("Error: --checkpoint-every-documents must be positive\n")
        return 1

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
    if not args.build:
        print(json.dumps(out_data, indent=2, sort_keys=True))
        return 0

    from dataset.src.bytesource import list_source_files, make_http_reader
    from dataset.src.streaming import build_stream_cache
    from dataset.src.workplan import build_work_plan, load_work_plan, save_work_plan

    output_dir = Path(args.output_dir).resolve()
    plan_path = output_dir / config.WORK_PLAN_FILENAME
    if args.resume:
        plan = load_work_plan(plan_path)
    else:
        if plan_path.exists() or (output_dir / config.PROGRESS_FILENAME).exists():
            raise FileExistsError(
                f"stream-cache state already exists in {output_dir}; use --resume or a new output directory"
            )
        source_files = list_source_files(config.DATASET_REPOSITORY, config.DATASET_REVISION)
        plan = build_work_plan(
            source_files,
            region_bytes=config.REGION_BYTES,
            seed=config.SELECTION_SEED,
            repository=config.DATASET_REPOSITORY,
            revision=config.DATASET_REVISION,
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        save_work_plan(plan_path, plan)
    manifest = build_stream_cache(
        output_dir,
        stream_cfg,
        plan,
        lambda source: make_http_reader(source, config.DATASET_REPOSITORY, config.DATASET_REVISION),
        resume=bool(args.resume),
        checkpoint_every_documents=args.checkpoint_every_documents,
    )
    out_data.update({
        "status": "complete",
        "notice": "Stream-cache build completed from the pinned source revision.",
        "output_dir": str(output_dir),
        "work_plan_hash": plan.hash,
        "manifest": manifest,
    })
    print(json.dumps(out_data, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    sys.exit(main())
