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
    bucket_id = getattr(args, "dataset_shard_bucket", None)
    if not bucket_id:
        return None
    run_id = getattr(args, "dataset_shard_run_id", None)
    manifest_path = getattr(args, "dataset_manifest", None)
    if not run_id or manifest_path is None:
        raise RuntimeError("rolling dataset shards require a run ID and manifest")
    token_env = getattr(args, "dataset_shard_token_env", "HF_TOKEN")
    token = os.environ.get(token_env)
    if not token:
        raise RuntimeError(f"{token_env} is required for rolling dataset shard reads")
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise RuntimeError("rolling dataset manifest must contain a JSON object")
    production = payload.get("production")
    if not isinstance(production, Mapping) or production.get("run_id") != run_id:
        raise RuntimeError("rolling dataset manifest run ID mismatch")

    from dataset.rolling_cache import RollingShardCache
    from dataset.src.hf_bucket_shards import HuggingFaceBucketShardStore

    store = HuggingFaceBucketShardStore(bucket_id, token=token, create_bucket=False)
    return RollingShardCache(
        root=args.dataset_dir,
        run_id=run_id,
        manifest=payload,
        store=store,
        prefetch_shards=getattr(args, "dataset_shard_prefetch", 1),
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
