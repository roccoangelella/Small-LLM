"""Command-line entry point for the production dataset cache."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dataset import config
from dataset.src.bytesource import list_source_files, make_http_reader
from dataset.src.hf_bucket_shards import HuggingFaceBucketShardStore
from dataset.src.remote import RemoteShardStore
from dataset.src.storage import write_json_atomic
from dataset.src.streaming import StreamCacheConfig
from dataset.src.workplan import build_work_plan, load_work_plan, save_work_plan

from .builder import build_production_cache
from .policy import DEFAULT_CHECKPOINT_SOURCE_TOKENS, ProductionPolicy
from .safety import preflight_disk, preflight_remote_shard_disk


def _load_weights(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"could not read weights file {path}: {error}") from error
    if not isinstance(payload, dict):
        raise RuntimeError("weights JSON must be a top-level object")
    return payload


def _default_hf_dataset_bucket() -> str | None:
    explicit = os.environ.get("SMALL_LLM_HF_DATASET_BUCKET_ID", "").strip()
    if explicit:
        return explicit
    repo_id = os.environ.get("SMALL_LLM_HF_REPO_ID", "").strip()
    return f"{repo_id}-datasets" if repo_id else None


def _builder_resume_mode(output_dir: Path) -> bool:
    progress_path = output_dir / config.PROGRESS_FILENAME
    if progress_path.is_file():
        return True
    manifest_path = output_dir / config.MANIFEST_FILENAME
    if manifest_path.exists():
        raise RuntimeError(
            f"resume requested without {config.PROGRESS_FILENAME}, but a manifest exists in {output_dir}"
        )
    allowed = {config.WORK_PLAN_FILENAME}
    present = {path.name for path in output_dir.iterdir()} if output_dir.is_dir() else set()
    unexpected = sorted(present - allowed)
    if unexpected:
        raise RuntimeError(
            f"resume requested without {config.PROGRESS_FILENAME}, and {output_dir} contains "
            f"unexpected pre-checkpoint artifacts: {unexpected}"
        )
    logging.info(
        "Resume requested before the first durable dataset checkpoint; reusing %s and starting the cache builder from empty state",
        config.WORK_PLAN_FILENAME,
    )
    return False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Build or resume the production schema-v2 Nemotron cache."
    )
    parser.add_argument("--weights-file", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-local-only", action="store_true")
    parser.add_argument("--allow-unsafe-low-disk", action="store_true")
    parser.add_argument(
        "--evict-remote-shards",
        action="store_true",
        help="Evict finalized local shards only after verified HF-bucket durability and progress commit.",
    )
    parser.add_argument(
        "--hf-bucket-id",
        default=_default_hf_dataset_bucket(),
        help=(
            "Private HF Storage Bucket; defaults to SMALL_LLM_HF_DATASET_BUCKET_ID "
            "or <SMALL_LLM_HF_REPO_ID>-datasets."
        ),
    )
    parser.add_argument("--hf-token-env", default="HF_TOKEN")
    parser.add_argument("--target-tokens", type=int, default=config.TARGET_ACCEPTED_SOURCE_TOKENS)
    parser.add_argument("--minimum-tokens", type=int, default=config.MINIMUM_ACCEPTED_SOURCE_TOKENS)
    parser.add_argument("--maximum-tokens", type=int, default=config.MAXIMUM_ACCEPTED_SOURCE_TOKENS)
    parser.add_argument(
        "--checkpoint-source-tokens",
        type=int,
        default=DEFAULT_CHECKPOINT_SOURCE_TOKENS,
    )
    parser.add_argument("--context-length", type=int, default=2048)
    parser.add_argument("--sequences-per-block", type=int, default=512)
    parser.add_argument("--target-shard-bytes", type=int, default=1024**3)
    parser.add_argument("--reader-workers", type=int, default=4)
    parser.add_argument("--max-in-flight-work-items", type=int, default=16)
    parser.add_argument("--per-cluster-queue-limit", type=int, default=100)
    parser.add_argument("--prepared-block-queue-limit", type=int, default=200)
    parser.add_argument("--prefetch-head-start", type=int, default=10)
    parser.add_argument("--minimum-prefetched-source-tokens", type=int, default=1_000_000)
    parser.add_argument("--minimum-populated-cluster-queues", type=int, default=2)
    parser.add_argument("--maximum-rolling-mixture-error", type=float, default=1.0)
    parser.add_argument("--maximum-waiting-documents", type=int, default=32)
    parser.add_argument("--reader-batch-source-tokens", type=int, default=1_000_000)
    parser.add_argument("--reader-batch-documents", type=int, default=1000)
    parser.add_argument("--reader-batch-max-bytes", type=int, default=16 * 1024 * 1024)
    parser.add_argument(
        "--simulate-crash-after-documents",
        type=int,
        default=None,
        help=argparse.SUPPRESS,
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = build_parser().parse_args(argv)
    try:
        if args.allow_local_only and args.evict_remote_shards:
            raise RuntimeError("--evict-remote-shards requires HF bucket durability")

        weights = _load_weights(args.weights_file)
        policy = ProductionPolicy(
            run_id=args.run_id,
            target_source_tokens=args.target_tokens,
            minimum_source_tokens=args.minimum_tokens,
            maximum_source_tokens=args.maximum_tokens,
            checkpoint_source_tokens=args.checkpoint_source_tokens,
            remote_required=not args.allow_local_only,
        )
        stream = StreamCacheConfig(
            context_length=args.context_length,
            sequences_per_block=args.sequences_per_block,
            target_shard_bytes=args.target_shard_bytes,
            reader_workers=args.reader_workers,
            max_in_flight_work_items=args.max_in_flight_work_items,
            per_cluster_queue_limit=args.per_cluster_queue_limit,
            prepared_block_queue_limit=args.prepared_block_queue_limit,
            prefetch_head_start=args.prefetch_head_start,
            weights=weights,
            scheduler_tie_break_seed=config.SELECTION_SEED,
            minimum_prefetched_source_tokens=args.minimum_prefetched_source_tokens,
            minimum_populated_cluster_queues=args.minimum_populated_cluster_queues,
            maximum_rolling_mixture_error=args.maximum_rolling_mixture_error,
            maximum_waiting_documents=args.maximum_waiting_documents,
            reader_batch_source_tokens=args.reader_batch_source_tokens,
            reader_batch_documents=args.reader_batch_documents,
            reader_batch_max_bytes=args.reader_batch_max_bytes,
        )
        output_dir = args.output_dir.resolve()
        if args.evict_remote_shards:
            preflight_remote_shard_disk(
                output_dir,
                stream.target_shard_bytes,
                allow_unsafe=args.allow_unsafe_low_disk,
            )
        else:
            preflight_disk(
                output_dir,
                policy.maximum_source_tokens,
                allow_unsafe=args.allow_unsafe_low_disk,
            )

        plan_path = output_dir / config.WORK_PLAN_FILENAME
        builder_resume = False
        if args.resume:
            plan = load_work_plan(plan_path)
            builder_resume = _builder_resume_mode(output_dir)
        else:
            if plan_path.exists() or (output_dir / config.PROGRESS_FILENAME).exists():
                raise FileExistsError(
                    f"production state already exists in {output_dir}; use --resume or a new directory"
                )
            source_files = list_source_files(
                config.DATASET_REPOSITORY,
                config.DATASET_REVISION,
            )
            plan = build_work_plan(
                source_files,
                region_bytes=config.REGION_BYTES,
                seed=config.SELECTION_SEED,
                repository=config.DATASET_REPOSITORY,
                revision=config.DATASET_REVISION,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            save_work_plan(plan_path, plan)

        remote_store: RemoteShardStore | None = None
        hf_store: HuggingFaceBucketShardStore | None = None
        if not args.allow_local_only:
            if not args.hf_bucket_id:
                raise RuntimeError(
                    "HF dataset durability requires --hf-bucket-id, "
                    "SMALL_LLM_HF_DATASET_BUCKET_ID, or SMALL_LLM_HF_REPO_ID"
                )
            token = os.environ.get(args.hf_token_env)
            if not token:
                raise RuntimeError(
                    f"{args.hf_token_env} is required for HF dataset bucket access"
                )
            hf_store = HuggingFaceBucketShardStore(
                args.hf_bucket_id,
                token=token,
                private=True,
                create_bucket=True,
            )
            remote_store = hf_store

        manifest = build_production_cache(
            output_dir,
            stream,
            policy,
            plan,
            lambda source: make_http_reader(
                source,
                config.DATASET_REPOSITORY,
                config.DATASET_REVISION,
            ),
            remote_store=remote_store,
            resume=builder_resume,
            simulate_crash_after_documents=args.simulate_crash_after_documents,
            evict_remote_shards=args.evict_remote_shards,
        )
        manifest["sequences_per_block"] = stream.sequences_per_block
        manifest["target_shard_bytes"] = stream.target_shard_bytes
        manifest["remote_transport"] = {
            "backend": "hf_bucket" if not args.allow_local_only else "local_only",
            "evict_local_finalized_shards": bool(args.evict_remote_shards),
            "hf_bucket_id": args.hf_bucket_id if hf_store is not None else None,
        }
        manifest_path = output_dir / config.MANIFEST_FILENAME
        write_json_atomic(manifest_path, manifest)
        if hf_store is not None:
            ready = hf_store.publish_dataset_manifest(
                run_id=policy.run_id,
                manifest_path=manifest_path,
            )
            logging.info(
                "published HF dataset readiness: run_id=%s bucket=%s target_reached=%s",
                policy.run_id,
                hf_store.bucket_id,
                ready.get("target_reached"),
            )
        print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
        return 0
    except Exception as error:  # noqa: BLE001 - concise CLI failure boundary
        sys.stderr.write(f"production dataset error: {type(error).__name__}: {error}\n")
        return 1
