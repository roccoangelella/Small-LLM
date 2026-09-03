#!/usr/bin/env python3
"""Qualify exact-batch two-T4 DDP against the current single-T4 training path.

This is intentionally a disposable qualification harness. It never writes into
production checkpoint namespaces and never talks to W&B or remote storage.
Both executions start from the same seed and consume the same real schema-v2
training blocks. The DDP path keeps the 16-sequence optimizer block intact:
each of two ranks owns eight rows, executes two microbatches of four, and DDP
synchronizes only on the second local backward pass.
"""
from __future__ import annotations

import argparse
import contextlib
import json
import math
import os
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

RUN_ID = "20m-2b-dataset-001"
SEQUENCES_PER_BLOCK = 16
MICROBATCH = 4
CONTEXT_LENGTH = 2048
WORLD_SIZE = 2
SEED = 17
DEFAULT_WARMUP_BLOCKS = 1
DEFAULT_MEASURE_BLOCKS = 4
DEFAULT_MIN_SPEEDUP = 1.60
DEFAULT_MAX_LOSS_DELTA = 8e-4
DEFAULT_MAX_GRADIENT_RELATIVE_DELTA = 8e-3
DEFAULT_MAX_PARAMETER_RELATIVE_L2 = 1e-3
DEFAULT_MAX_PARAMETER_ABS = 8e-4
DEFAULT_MAX_OPTIMIZER_RELATIVE_L2 = 2e-3
DEFAULT_MAX_OPTIMIZER_ABS = 1e-3


class QualificationFailure(RuntimeError):
    pass


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be non-negative")
    return parsed


def positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive and finite")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path)
    parser.add_argument("--warmup-blocks", type=nonnegative_int, default=DEFAULT_WARMUP_BLOCKS)
    parser.add_argument("--measure-blocks", type=positive_int, default=DEFAULT_MEASURE_BLOCKS)
    parser.add_argument("--minimum-speedup", type=positive_float, default=DEFAULT_MIN_SPEEDUP)
    parser.add_argument("--output", type=Path, default=Path("/kaggle/working/dual-t4-qualification.json"))
    parser.add_argument("--max-loss-delta", type=positive_float, default=DEFAULT_MAX_LOSS_DELTA)
    parser.add_argument(
        "--max-gradient-relative-delta",
        type=positive_float,
        default=DEFAULT_MAX_GRADIENT_RELATIVE_DELTA,
    )
    parser.add_argument(
        "--max-parameter-relative-l2",
        type=positive_float,
        default=DEFAULT_MAX_PARAMETER_RELATIVE_L2,
    )
    parser.add_argument("--max-parameter-abs", type=positive_float, default=DEFAULT_MAX_PARAMETER_ABS)
    parser.add_argument(
        "--max-optimizer-relative-l2",
        type=positive_float,
        default=DEFAULT_MAX_OPTIMIZER_RELATIVE_L2,
    )
    parser.add_argument("--max-optimizer-abs", type=positive_float, default=DEFAULT_MAX_OPTIMIZER_ABS)
    parser.add_argument("--worker", choices=("single", "ddp"), help=argparse.SUPPRESS)
    parser.add_argument("--worker-output", type=Path, help=argparse.SUPPRESS)
    return parser


def _read_json(path: Path) -> Mapping[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise QualificationFailure(f"cannot read JSON {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise QualificationFailure(f"expected JSON object in {path}")
    return value


def _dataset_matches(root: Path) -> bool:
    manifest = root / "manifest.json"
    if not manifest.is_file() or not (root / "train").is_dir():
        return False
    try:
        payload = _read_json(manifest)
    except QualificationFailure:
        return False
    production = payload.get("production")
    return bool(
        payload.get("schema_version") == 2
        and payload.get("sequence_format") == "context_plus_one"
        and payload.get("context_length") == CONTEXT_LENGTH
        and payload.get("sequences_per_block") == SEQUENCES_PER_BLOCK
        and isinstance(production, Mapping)
        and production.get("run_id") == RUN_ID
    )


def resolve_dataset(explicit: Path | None) -> Path:
    if explicit is not None:
        root = explicit.resolve()
        if not _dataset_matches(root):
            raise QualificationFailure(f"dataset does not match {RUN_ID}: {root}")
        return root
    kaggle_input = Path("/kaggle/input")
    if not kaggle_input.is_dir():
        raise QualificationFailure("--dataset-dir is required outside Kaggle")
    roots = sorted({path.parent for path in kaggle_input.rglob("manifest.json")})
    matches = [root for root in roots if _dataset_matches(root)]
    if len(matches) != 1:
        raise QualificationFailure(
            f"expected exactly one attached {RUN_ID} dataset, found {len(matches)}: "
            + ", ".join(str(path) for path in matches)
        )
    return matches[0]


def _hardware_snapshot() -> dict[str, Any]:
    import torch

    count = torch.cuda.device_count()
    devices = []
    for index in range(count):
        props = torch.cuda.get_device_properties(index)
        devices.append(
            {
                "index": index,
                "name": props.name,
                "total_memory_bytes": int(props.total_memory),
                "major": int(props.major),
                "minor": int(props.minor),
            }
        )
    return {"cuda_device_count": count, "devices": devices}


def require_two_t4s() -> dict[str, Any]:
    snapshot = _hardware_snapshot()
    if snapshot["cuda_device_count"] < WORLD_SIZE:
        raise QualificationFailure("dual-T4 qualification requires at least two visible CUDA devices")
    first_two = snapshot["devices"][:WORLD_SIZE]
    if any("T4" not in str(device["name"]).upper() for device in first_two):
        raise QualificationFailure(
            "dual-T4 qualification requires the first two CUDA devices to be T4s: "
            + json.dumps(first_two)
        )
    return snapshot


def _training_config():
    from trainer.config import TrainerConfig

    return TrainerConfig(
        optimizer="hybrid_muon_adamw",
        microbatch_size=MICROBATCH,
        learning_rate=3e-4,
        weight_decay=0.1,
        muon_momentum=0.95,
        muon_lr_multiplier=1.0,
        muon_update_rms=0.18,
        muon_weight_decay=0.1,
        max_grad_norm=1.0,
        precision="fp16",
        schedule="constant",
        seed=SEED,
    )


def _build_model(device: int):
    import torch
    from model.config import ModelConfig
    from model.initialization import initialize_model
    from model.model import SmallLLM
    from trainer.engine import _build_training_optimizer, seed_everything

    seed_everything(SEED)
    config = ModelConfig.smoke(architecture="gdn2_hybrid", gdn_chunk_size=32)
    model = SmallLLM(config)
    initialize_model(model, "normal")
    torch.cuda.set_device(device)
    model = model.to(torch.device("cuda", device))
    trainer_config = _training_config()
    optimizer = _build_training_optimizer(model, trainer_config)
    scaler = torch.amp.GradScaler("cuda", enabled=True)
    return model, optimizer, scaler, trainer_config, config


def _load_blocks(dataset: Path, count: int) -> list[Any]:
    from model.config import ModelConfig
    from trainer.shards import SchemaV2ShardReader

    model_config = ModelConfig.smoke(architecture="gdn2_hybrid", gdn_chunk_size=32)
    reader = SchemaV2ShardReader(
        dataset,
        split="train",
        sequences_per_block=SEQUENCES_PER_BLOCK,
        semantic_vocab_size=model_config.semantic_vocab_size,
        manifest_path=dataset / "manifest.json",
        context_length=CONTEXT_LENGTH,
    )
    blocks = list(reader.iter_from_start(maximum_blocks=count))
    if len(blocks) != count:
        raise QualificationFailure(f"requested {count} blocks but dataset provided {len(blocks)}")
    for block in blocks:
        if block.sequence_count != SEQUENCES_PER_BLOCK:
            raise QualificationFailure(
                f"block {block.block_id} has {block.sequence_count} sequences, expected {SEQUENCES_PER_BLOCK}"
            )
    return blocks


def _global_loss_sum(local_loss_sum: float, distributed: bool) -> float:
    if not distributed:
        return local_loss_sum
    import torch
    import torch.distributed as dist

    value = torch.tensor(local_loss_sum, dtype=torch.float64, device="cuda")
    dist.all_reduce(value, op=dist.ReduceOp.SUM)
    return float(value.item())


def _run_one_update(
    model: Any,
    optimizer: Any,
    scaler: Any,
    config: Any,
    block: Any,
    *,
    distributed: bool,
    rank: int,
    world_size: int,
) -> dict[str, Any]:
    import torch
    import torch.distributed as dist
    from torch.nn import functional as F
    from trainer.step import _microbatch_to_device, _ordered_batch_tensors

    if distributed and world_size != WORLD_SIZE:
        raise QualificationFailure(f"expected world_size={WORLD_SIZE}, got {world_size}")

    input_ids, labels = _ordered_batch_tensors(block)
    if distributed:
        rows_per_rank = block.sequence_count // world_size
        if rows_per_rank * world_size != block.sequence_count:
            raise QualificationFailure("optimizer block cannot be split evenly across DDP ranks")
        rank_start = rank * rows_per_rank
        rank_stop = rank_start + rows_per_rank
    else:
        rank_start, rank_stop = 0, block.sequence_count

    retries = 0
    initial_scale = float(scaler.get_scale())
    while True:
        optimizer.zero_grad(set_to_none=True)
        local_loss = torch.zeros((), dtype=torch.float32, device=torch.device("cuda", rank if distributed else 0))
        starts = list(range(rank_start, rank_stop, config.microbatch_size))
        for local_index, start in enumerate(starts):
            stop = min(rank_stop, start + config.microbatch_size)
            micro_inputs, micro_labels = _microbatch_to_device(
                input_ids,
                labels,
                start=start,
                stop=stop,
                device=torch.device("cuda", rank if distributed else 0),
            )
            sync_context = contextlib.nullcontext()
            if distributed and local_index + 1 < len(starts):
                sync_context = model.no_sync()
            with sync_context:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(micro_inputs)
                    loss_sum = F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        micro_labels.reshape(-1),
                        reduction="sum",
                    )
                if not torch.isfinite(loss_sum):
                    raise QualificationFailure(f"non-finite forward loss on block {block.block_id}")
                local_loss += loss_sum.detach().float()
                normalization = float(block.target_token_count)
                multiplier = float(world_size) if distributed else 1.0
                scaler.scale(multiplier * loss_sum / normalization).backward()

        scaler.unscale_(optimizer)
        parameters = model.module.parameters() if distributed else model.parameters()
        gradient_norm = torch.nn.utils.clip_grad_norm_(parameters, config.max_grad_norm)
        finite_gradient = bool(torch.isfinite(gradient_norm))
        scale_before = float(scaler.get_scale())
        scaler.step(optimizer)
        scaler.update()
        scale_after = float(scaler.get_scale())
        overflow = not finite_gradient or scale_after < scale_before
        if distributed:
            overflow_tensor = torch.tensor(int(overflow), device=torch.device("cuda", rank))
            dist.all_reduce(overflow_tensor, op=dist.ReduceOp.MAX)
            overflow = bool(overflow_tensor.item())
        if overflow:
            retries += 1
            if retries > 20:
                raise QualificationFailure(
                    f"FP16 scaler did not stabilize on block {block.block_id}; initial_scale={initial_scale:g} "
                    f"current_scale={scale_after:g}"
                )
            continue
        break

    global_loss_sum = _global_loss_sum(float(local_loss.item()), distributed)
    return {
        "block_id": int(block.block_id),
        "loss": global_loss_sum / float(block.target_token_count),
        "gradient_norm": float(gradient_norm.detach().item()),
        "target_tokens": int(block.target_token_count),
        "overflow_retries": retries,
        "grad_scaler_scale": float(scaler.get_scale()),
    }


def _cpu_tree(value: Any) -> Any:
    import torch

    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, Mapping):
        return {key: _cpu_tree(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_cpu_tree(item) for item in value)
    if isinstance(value, list):
        return [_cpu_tree(item) for item in value]
    return value


def _worker(args: argparse.Namespace) -> int:
    import torch
    import torch.distributed as dist
    from torch.nn.parallel import DistributedDataParallel

    if args.worker_output is None:
        raise QualificationFailure("worker output path is required")
    dataset = resolve_dataset(args.dataset_dir)
    total_blocks = args.warmup_blocks + args.measure_blocks
    blocks = _load_blocks(dataset, total_blocks)

    distributed = args.worker == "ddp"
    if distributed:
        if not dist.is_available():
            raise QualificationFailure("torch.distributed is unavailable")
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", rank))
        if world_size != WORLD_SIZE:
            raise QualificationFailure(f"DDP worker requires world_size={WORLD_SIZE}")
        torch.cuda.set_device(local_rank)
        model, optimizer, scaler, config, model_config = _build_model(local_rank)
        model = DistributedDataParallel(
            model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            gradient_as_bucket_view=True,
        )
    else:
        rank, world_size, local_rank = 0, 1, 0
        torch.cuda.set_device(0)
        model, optimizer, scaler, config, model_config = _build_model(0)

    rows: list[dict[str, Any]] = []
    for index, block in enumerate(blocks):
        if distributed:
            dist.barrier()
        torch.cuda.synchronize(local_rank)
        started = time.perf_counter()
        row = _run_one_update(
            model,
            optimizer,
            scaler,
            config,
            block,
            distributed=distributed,
            rank=local_rank,
            world_size=world_size,
        )
        torch.cuda.synchronize(local_rank)
        if distributed:
            dist.barrier()
        elapsed = max(time.perf_counter() - started, 1e-12)
        row["elapsed_seconds"] = elapsed
        row["tokens_per_second"] = float(block.target_token_count) / elapsed
        row["warmup"] = index < args.warmup_blocks
        rows.append(row)

    if rank == 0:
        raw_model = model.module if distributed else model
        payload = {
            "mode": args.worker,
            "world_size": world_size,
            "model_config": {
                "architecture": model_config.architecture,
                "context_length": model_config.max_seq_len,
                "gdn_chunk_size": model_config.gdn_chunk_size,
            },
            "trainer_config": config.as_dict(),
            "rows": rows,
            "model_state": _cpu_tree(raw_model.state_dict()),
            "optimizer_state": _cpu_tree(optimizer.state_dict()),
        }
        args.worker_output.parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, args.worker_output)
    if distributed:
        dist.barrier()
        dist.destroy_process_group()
    return 0


def _worker_command(args: argparse.Namespace, worker: str, output: Path) -> list[str]:
    base = [
        str(Path(__file__).resolve()),
        "--dataset-dir",
        str(resolve_dataset(args.dataset_dir)),
        "--warmup-blocks",
        str(args.warmup_blocks),
        "--measure-blocks",
        str(args.measure_blocks),
        "--worker",
        worker,
        "--worker-output",
        str(output),
    ]
    if worker == "single":
        return [sys.executable, *base]
    return [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        f"--nproc-per-node={WORLD_SIZE}",
        *base,
    ]


def _run_child(command: Sequence[str], *, visible_devices: str) -> None:
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    env.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    subprocess.run(command, cwd=REPO, env=env, check=True)


def _tensor_metrics(left: Mapping[str, Any], right: Mapping[str, Any]) -> dict[str, float | int]:
    import torch

    if set(left) != set(right):
        raise QualificationFailure("state dictionaries have different keys")
    sq_error = 0.0
    sq_reference = 0.0
    maximum_abs = 0.0
    tensor_count = 0
    element_count = 0
    for key in sorted(left):
        a, b = left[key], right[key]
        if isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor):
            if a.shape != b.shape:
                raise QualificationFailure(f"state tensor shape mismatch for {key}: {a.shape} != {b.shape}")
            delta = a.double() - b.double()
            sq_error += float(delta.square().sum().item())
            sq_reference += float(a.double().square().sum().item())
            if delta.numel():
                maximum_abs = max(maximum_abs, float(delta.abs().max().item()))
            tensor_count += 1
            element_count += delta.numel()
        elif a != b:
            raise QualificationFailure(f"non-tensor state mismatch for {key}: {a!r} != {b!r}")
    return {
        "relative_l2": math.sqrt(sq_error / max(sq_reference, 1e-30)),
        "maximum_abs": maximum_abs,
        "tensor_count": tensor_count,
        "element_count": element_count,
    }


def _flatten_optimizer_tensors(value: Any, *, prefix: str = "root") -> dict[str, Any]:
    import torch

    flat: dict[str, Any] = {}
    if isinstance(value, torch.Tensor):
        flat[prefix] = value
    elif isinstance(value, Mapping):
        for key in sorted(value, key=lambda item: str(item)):
            flat.update(_flatten_optimizer_tensors(value[key], prefix=f"{prefix}.{key}"))
    elif isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            flat.update(_flatten_optimizer_tensors(item, prefix=f"{prefix}.{index}"))
    return flat


def _compare_rows(single: Sequence[Mapping[str, Any]], ddp: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(single) != len(ddp):
        raise QualificationFailure("single/DDP runs produced different row counts")
    loss_deltas = []
    gradient_relative_deltas = []
    for left, right in zip(single, ddp, strict=True):
        for key in ("block_id", "target_tokens", "warmup"):
            if left.get(key) != right.get(key):
                raise QualificationFailure(f"single/DDP row mismatch for {key}")
        loss_deltas.append(abs(float(left["loss"]) - float(right["loss"])))
        left_grad, right_grad = abs(float(left["gradient_norm"])), abs(float(right["gradient_norm"]))
        gradient_relative_deltas.append(
            abs(left_grad - right_grad) / max(left_grad, right_grad, 1e-12)
        )
    return {
        "maximum_loss_delta": max(loss_deltas, default=0.0),
        "maximum_gradient_relative_delta": max(gradient_relative_deltas, default=0.0),
    }


def _throughput(rows: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    measured = [float(row["tokens_per_second"]) for row in rows if not bool(row["warmup"])]
    if not measured:
        raise QualificationFailure("qualification produced no measured blocks")
    return {
        "median_tokens_per_second": statistics.median(measured),
        "mean_tokens_per_second": statistics.fmean(measured),
        "minimum_tokens_per_second": min(measured),
        "maximum_tokens_per_second": max(measured),
    }


def _load_worker_result(path: Path) -> Mapping[str, Any]:
    import torch

    try:
        value = torch.load(path, map_location="cpu", weights_only=False)
    except (OSError, RuntimeError) as error:
        raise QualificationFailure(f"cannot load worker result {path}: {error}") from error
    if not isinstance(value, Mapping):
        raise QualificationFailure(f"worker result is not a mapping: {path}")
    return value


def _parent(args: argparse.Namespace) -> int:
    dataset = resolve_dataset(args.dataset_dir)
    hardware = require_two_t4s()
    with tempfile.TemporaryDirectory(prefix="small-llm-dual-t4-") as temp_dir:
        temp = Path(temp_dir)
        single_path, ddp_path = temp / "single.pt", temp / "ddp.pt"
        _run_child(_worker_command(args, "single", single_path), visible_devices="0")
        _run_child(_worker_command(args, "ddp", ddp_path), visible_devices="0,1")
        single = _load_worker_result(single_path)
        ddp = _load_worker_result(ddp_path)

    single_rows = single.get("rows")
    ddp_rows = ddp.get("rows")
    if not isinstance(single_rows, Sequence) or not isinstance(ddp_rows, Sequence):
        raise QualificationFailure("worker result is missing training rows")
    row_comparison = _compare_rows(single_rows, ddp_rows)
    single_tps = _throughput(single_rows)
    ddp_tps = _throughput(ddp_rows)
    speedup = ddp_tps["median_tokens_per_second"] / single_tps["median_tokens_per_second"]

    single_model, ddp_model = single.get("model_state"), ddp.get("model_state")
    if not isinstance(single_model, Mapping) or not isinstance(ddp_model, Mapping):
        raise QualificationFailure("worker result is missing model state")
    parameter_metrics = _tensor_metrics(single_model, ddp_model)

    single_optimizer = _flatten_optimizer_tensors(single.get("optimizer_state"))
    ddp_optimizer = _flatten_optimizer_tensors(ddp.get("optimizer_state"))
    optimizer_metrics = _tensor_metrics(single_optimizer, ddp_optimizer)

    checks = {
        "loss_parity": row_comparison["maximum_loss_delta"] <= args.max_loss_delta,
        "gradient_parity": (
            row_comparison["maximum_gradient_relative_delta"] <= args.max_gradient_relative_delta
        ),
        "parameter_parity": (
            parameter_metrics["relative_l2"] <= args.max_parameter_relative_l2
            and parameter_metrics["maximum_abs"] <= args.max_parameter_abs
        ),
        "optimizer_parity": (
            optimizer_metrics["relative_l2"] <= args.max_optimizer_relative_l2
            and optimizer_metrics["maximum_abs"] <= args.max_optimizer_abs
        ),
        "throughput": speedup >= args.minimum_speedup,
    }
    status = "passed" if all(checks.values()) else "failed"
    report = {
        "status": status,
        "qualification": "single_t4_vs_exact_batch_two_t4_ddp",
        "dataset": str(dataset),
        "dataset_run_id": RUN_ID,
        "hardware": hardware,
        "contract": {
            "sequences_per_optimizer_block": SEQUENCES_PER_BLOCK,
            "single_t4_microbatch": MICROBATCH,
            "ddp_world_size": WORLD_SIZE,
            "ddp_sequences_per_rank": SEQUENCES_PER_BLOCK // WORLD_SIZE,
            "ddp_microbatches_per_rank": (SEQUENCES_PER_BLOCK // WORLD_SIZE) // MICROBATCH,
            "gradient_normalization": "world_size_times_local_loss_sum_over_global_target_tokens",
            "ddp_sync": "no_sync_first_local_microbatch_sync_second",
            "optimizer": "hybrid_muon_adamw",
            "precision": "fp16",
            "gdn_backend": "FLA-preferred CUDA path",
        },
        "blocks": {
            "warmup": args.warmup_blocks,
            "measured": args.measure_blocks,
        },
        "thresholds": {
            "minimum_median_speedup": args.minimum_speedup,
            "maximum_loss_delta": args.max_loss_delta,
            "maximum_gradient_relative_delta": args.max_gradient_relative_delta,
            "maximum_parameter_relative_l2": args.max_parameter_relative_l2,
            "maximum_parameter_abs": args.max_parameter_abs,
            "maximum_optimizer_relative_l2": args.max_optimizer_relative_l2,
            "maximum_optimizer_abs": args.max_optimizer_abs,
        },
        "checks": checks,
        "parity": {
            **row_comparison,
            "parameters": parameter_metrics,
            "optimizer": optimizer_metrics,
        },
        "throughput": {
            "single_t4": single_tps,
            "dual_t4_ddp": ddp_tps,
            "median_speedup": speedup,
        },
        "single_rows": list(single_rows),
        "ddp_rows": list(ddp_rows),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    if status != "passed":
        failed = ", ".join(name for name, passed in checks.items() if not passed)
        raise QualificationFailure(f"dual-T4 qualification failed: {failed}; report={args.output}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.worker is not None:
            return _worker(args)
        return _parent(args)
    except QualificationFailure as error:
        print(f"[dual-t4-qualification] ERROR: {error}", file=sys.stderr, flush=True)
        return 2
    except subprocess.CalledProcessError as error:
        print(
            f"[dual-t4-qualification] ERROR: worker exited with status {error.returncode}",
            file=sys.stderr,
            flush=True,
        )
        return error.returncode or 2


if __name__ == "__main__":
    raise SystemExit(main())
