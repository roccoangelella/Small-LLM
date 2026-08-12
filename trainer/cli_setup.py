"""Construct CLI model, data source, trainer, and checkpoint coordinator."""

from __future__ import annotations

import json
import os
from collections.abc import Mapping

from dataset.src.joint_checkpoint import CheckpointCoordinator
from model.config import ModelConfig
from model.initialization import initialize_model
from model.model import SmallLLM

from .config import TrainerConfig
from .engine import TrainerEngine, TrainingSession, seed_everything
from .identity import checkpoint_identity, saved_checkpoint_identity
from .shards import SchemaV2ShardReader


def _rolling_cache(args: object) -> object | None:
    # Explicit trainer flags remain the generic provider-neutral interface.
    # Modal may inject the same values through environment variables so its
    # existing command builder does not acquire dataset-provider logic.
    explicit_bucket = getattr(args, "dataset_shard_bucket", None)
    bucket_id = explicit_bucket or os.environ.get("SMALL_LLM_DATASET_SHARD_BUCKET")
    if not bucket_id:
        return None
    # Modal microbatch probes use only a handful of blocks from the CPU-staged
    # bootstrap window. Do not start a one-GiB successor download from a short-
    # lived probe process; the real online subprocess enables rolling prefetch.
    if (
        not explicit_bucket
        and os.environ.get("SMALL_LLM_MODAL_ROLLING_DATASET") == "1"
        and getattr(args, "wandb_mode", "disabled") == "disabled"
    ):
        return None

    run_id = getattr(args, "dataset_shard_run_id", None) or os.environ.get(
        "SMALL_LLM_DATASET_SHARD_RUN_ID"
    )
    manifest_path = getattr(args, "dataset_manifest", None)
    if not run_id or manifest_path is None:
        raise RuntimeError("rolling dataset shards require a run ID and manifest")
    token_env = getattr(args, "dataset_shard_token_env", "HF_TOKEN")
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(f"{token_env} is required for rolling dataset shard reads")
    prefetch = getattr(args, "dataset_shard_prefetch", 1)
    env_prefetch = os.environ.get("SMALL_LLM_DATASET_SHARD_PREFETCH")
    if env_prefetch is not None:
        try:
            prefetch = int(env_prefetch)
        except ValueError as error:
            raise RuntimeError("SMALL_LLM_DATASET_SHARD_PREFETCH must be an integer") from error
    if isinstance(prefetch, bool) or prefetch < 1:
        raise RuntimeError("rolling dataset shard prefetch must be at least one")

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("rolling dataset manifest must contain a JSON object")
    production = payload.get("production")
    if not isinstance(production, Mapping) or production.get("run_id") != run_id:
        raise RuntimeError("rolling dataset manifest run ID mismatch")

    from dataset.src.hf_bucket_shards import HuggingFaceBucketShardStore

    store = HuggingFaceBucketShardStore(bucket_id, token=token, create_bucket=False)
    incremental = payload.get("incremental_frontier")
    if isinstance(incremental, Mapping):
        from dataset.incremental_frontier import IncrementalRollingShardCache, RUN_CONTRACT_FILENAME

        contract_path = manifest_path.parent / RUN_CONTRACT_FILENAME
        try:
            contract = json.loads(contract_path.read_text(encoding="utf-8"))
        except (OSError, ValueError, TypeError) as error:
            raise RuntimeError("incremental rolling dataset has no readable run contract") from error
        if not isinstance(contract, Mapping):
            raise RuntimeError("incremental rolling dataset run contract is not an object")
        if contract.get("run_id") != run_id:
            raise RuntimeError("incremental rolling dataset run contract ID mismatch")
        if incremental.get("contract_sha256") != contract.get("contract_sha256"):
            raise RuntimeError("incremental rolling dataset manifest/contract identity mismatch")
        return IncrementalRollingShardCache(
            root=args.dataset_dir,
            run_id=run_id,
            contract=contract,
            store=store,
            prefetch_shards=prefetch,
        )

    from dataset.rolling_cache import RollingShardCache

    return RollingShardCache(
        root=args.dataset_dir,
        run_id=run_id,
        manifest=payload,
        store=store,
        prefetch_shards=prefetch,
        evict_consumed=True,
    )


def setup(args: object):
    seed_everything(args.seed)
    factory = ModelConfig.smoke if args.model_size == "smoke" else ModelConfig.substantive
    model_overrides: dict[str, object] = {"architecture": args.architecture}
    if args.gdn_chunk_size is not None:
        model_overrides["gdn_chunk_size"] = args.gdn_chunk_size
    model_config = factory(**model_overrides)
    model = SmallLLM(model_config)
    initialize_model(model, args.initialization)
    trainer_config = TrainerConfig(
        optimizer=args.optimizer,
        microbatch_size=args.microbatch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        muon_momentum=args.muon_momentum,
        muon_lr_multiplier=args.muon_lr_multiplier,
        muon_update_rms=args.muon_update_rms,
        muon_weight_decay=args.muon_weight_decay,
        max_grad_norm=args.max_grad_norm,
        precision=args.precision,
        schedule=args.schedule,
        warmup_tokens=args.warmup_tokens,
        stable_tokens=args.stable_tokens,
        decay_tokens=args.decay_tokens,
        minimum_lr_ratio=args.minimum_lr_ratio,
        checkpoint_every_steps=args.checkpoint_every_steps,
        evaluation_every_steps=args.evaluation_every_steps,
        seed=args.seed,
    )
    cache_manager = _rolling_cache(args)
    source = SchemaV2ShardReader(
        args.dataset_dir,
        split="train",
        sequences_per_block=args.sequences_per_block,
        semantic_vocab_size=model_config.semantic_vocab_size,
        manifest_path=args.dataset_manifest,
        context_length=model_config.max_seq_len if args.dataset_manifest is not None else None,
        cache_manager=cache_manager,
    )
    engine = TrainerEngine(model, trainer_config, device=args.device)
    session = TrainingSession(engine, source)
    checkpoint_root = args.checkpoint_dir / args.resume if args.resume else None
    if checkpoint_root is not None and (checkpoint_root / "checkpoint.json").is_file():
        identities = saved_checkpoint_identity(checkpoint_root)
    else:
        identities = checkpoint_identity(
            args.dataset_dir,
            model_config=model_config,
            trainer_config=trainer_config,
            manifest_path=args.dataset_manifest,
            context_length=model_config.max_seq_len,
            sequences_per_block=args.sequences_per_block,
        )
    coordinator = CheckpointCoordinator(
        args.checkpoint_dir,
        configuration_hash=identities[0],
        source_hash=identities[1],
        schema_hash=identities[2],
    )
    if args.resume:
        session.load_checkpoint(coordinator, args.resume)
    return model_config, trainer_config, engine, session, coordinator


def validation_reader(args: object, model_config: object) -> SchemaV2ShardReader:
    return SchemaV2ShardReader(
        args.dataset_dir,
        split="validation",
        sequences_per_block=args.sequences_per_block,
        semantic_vocab_size=model_config.semantic_vocab_size,
        manifest_path=args.dataset_manifest,
        context_length=model_config.max_seq_len if args.dataset_manifest is not None else None,
    )
