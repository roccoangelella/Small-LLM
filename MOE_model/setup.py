"""Build the dedicated MoE model while reusing the validated data/checkpoint stack."""

from __future__ import annotations

from dataclasses import replace

from dataset.src.joint_checkpoint import CheckpointCoordinator
from trainer.cli_setup import _rolling_cache
from trainer.config import TrainerConfig
from trainer.engine import TrainingSession, seed_everything
from trainer.identity import checkpoint_identity, saved_checkpoint_identity
from trainer.shards import SchemaV2ShardReader

from .compile_lane import DEFAULT_COMPILE_MODE, apply_compile_lane
from .config import MoEModelConfig
from .engine import MoETrainerEngine
from .initialization import initialize_moe_model
from .model import MoESmallLLM


def _model_config_from_args(args: object) -> MoEModelConfig:
    """Resolve an explicit MoE identity without silently changing accepted production settings."""

    dense_overrides: dict[str, object] = {"architecture": args.architecture}
    if args.gdn_chunk_size is not None:
        dense_overrides["gdn_chunk_size"] = args.gdn_chunk_size

    if args.model_size == "smoke":
        model_config = MoEModelConfig.smoke(**dense_overrides)
    elif args.model_size == "substantive":
        model_config = MoEModelConfig.substantive(**dense_overrides)
    elif args.model_size == "accepted":
        model_config = MoEModelConfig.accepted(**dense_overrides)
    else:
        raise ValueError(f"unsupported MoE model size {args.model_size!r}")

    requested_balancing = getattr(args, "load_balancing", None)
    balancing_step_size = float(getattr(args, "balancing_step_size", 0.0))

    if args.model_size == "accepted":
        # Quantile Balancing is checkpoint-visible architecture, not a launch default.
        # A generic CLI option may confirm it but must never silently replace it.
        if requested_balancing not in {None, "quantile"}:
            raise ValueError(
                "the accepted production MoE is frozen to Quantile Balancing; "
                "omit --load-balancing or pass --load-balancing quantile"
            )
        if balancing_step_size != 0.0:
            raise ValueError(
                "the accepted Quantile Balancing controller does not use --balancing-step-size"
            )
        return model_config

    if requested_balancing == "quantile":
        raise ValueError("Quantile Balancing is available only on the version-3 accepted model")
    if requested_balancing is not None:
        model_config = replace(
            model_config,
            load_balancing=requested_balancing,
            balancing_step_size=balancing_step_size,
        )
    elif balancing_step_size != 0.0:
        raise ValueError("--balancing-step-size requires --load-balancing loss_free_sign")
    return model_config


def setup(args: object):
    seed_everything(args.seed)
    model_config = _model_config_from_args(args)

    model = MoESmallLLM(model_config)
    initialize_moe_model(model, args.initialization)
    # Execution lane, applied before the engine builds the optimizer. Compiling a
    # block in place never rebinds its parameters, so the optimizer below captures
    # the same tensors and the checkpoint contract is untouched.
    apply_compile_lane(model, getattr(args, "compile_mode", DEFAULT_COMPILE_MODE))
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
        schedule_anchor_tokens=args.schedule_anchor_tokens,
        cooldown_start_tokens=args.cooldown_start_tokens,
        settle_tokens=args.settle_tokens,
        settle_lr_ratio=args.settle_lr_ratio,
        base_power=args.base_power,
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
        context_length=(
            model_config.max_seq_len if args.dataset_manifest is not None else None
        ),
        cache_manager=cache_manager,
    )
    engine = MoETrainerEngine(model, trainer_config, device=args.device)
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


def validation_reader(args: object, model_config: MoEModelConfig) -> SchemaV2ShardReader:
    return SchemaV2ShardReader(
        args.dataset_dir,
        split="validation",
        sequences_per_block=args.sequences_per_block,
        semantic_vocab_size=model_config.semantic_vocab_size,
        manifest_path=args.dataset_manifest,
        context_length=(
            model_config.max_seq_len if args.dataset_manifest is not None else None
        ),
    )


__all__ = ["setup", "validation_reader", "_model_config_from_args"]
