#!/usr/bin/env python3
"""Measure real GDN-2 decay at the 20M/500M step-4000 checkpoint.

This is a forward-only diagnostic. It loads the actual checkpoint, reads the
first training microbatch from block 4000 (the block after checkpoint block
3999), runs the real model under trainer-style FP16 autocast, and records the
log-decay tensor that every GDN layer sends to its recurrence backend.

FLA is used only for forward execution here; forward parity is already
qualified. No backward pass, optimizer step, scheduler step, W&B write, or
checkpoint mutation occurs.

Typical Kaggle use:
    python kaggle/run_gdn2_step4000_decay_telemetry.py

If checkpoint auto-discovery cannot find the restored file:
    python kaggle/run_gdn2_step4000_decay_telemetry.py --checkpoint /path/to/checkpoint.pt
"""
from __future__ import annotations

import argparse
from array import array
import importlib
import json
import math
from pathlib import Path
import statistics
import subprocess
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

FLA_VERSION = "0.5.1"
TARGET_BLOCK = 4000
CONTEXT = 2048
STORED_TOKENS = CONTEXT + 1
SEQUENCES_PER_BLOCK = 16
DEFAULT_BATCH = 4
BOUNDARY_PASS = -0.50
BOUNDARY_FAIL = -0.75


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--dataset-dir", type=Path, default=None)
    parser.add_argument("--block", type=int, default=TARGET_BLOCK)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH)
    parser.add_argument(
        "--report",
        type=Path,
        default=Path("/kaggle/working/gdn2_step4000_decay_telemetry.json"),
    )
    return parser.parse_args()


def ensure_fla() -> None:
    try:
        from fla.ops.gdn2 import chunk_gdn2  # noqa: F401
        return
    except Exception:
        print(f"[setup] installing fla-core=={FLA_VERSION} --no-deps", flush=True)
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "--quiet", "--no-deps", f"fla-core=={FLA_VERSION}"],
        check=True,
    )
    for name in tuple(sys.modules):
        if name == "fla" or name.startswith("fla."):
            sys.modules.pop(name, None)
    importlib.invalidate_caches()
    from fla.ops.gdn2 import chunk_gdn2  # noqa: F401


def discover_checkpoint(explicit: Path | None) -> Path:
    if explicit is not None:
        path = explicit.expanduser().resolve()
        if not path.is_file():
            raise SystemExit(f"checkpoint not found: {path}")
        return path

    root = Path("/kaggle/working")
    candidates: list[Path] = []
    if root.is_dir():
        for pattern in ("*00004000*.pt", "*00004000*.pth", "*00004000*.ckpt", "checkpoint.pt"):
            candidates.extend(path for path in root.rglob(pattern) if path.is_file())
    # Prefer names that explicitly identify step 4000, then newest files.
    candidates = sorted(
        set(candidates),
        key=lambda p: ("4000" not in p.name and "00004000" not in p.name, -p.stat().st_mtime),
    )
    if not candidates:
        raise SystemExit(
            "No local checkpoint found under /kaggle/working. Pass --checkpoint with the restored step-4000 .pt path."
        )
    print(f"[checkpoint] auto-discovered {candidates[0]}", flush=True)
    return candidates[0]


def manifest_has_block(manifest: dict[str, Any], block: int) -> bool:
    if manifest.get("context_length") != CONTEXT or manifest.get("sequences_per_block") != SEQUENCES_PER_BLOCK:
        return False
    shards = manifest.get("shards")
    if not isinstance(shards, list):
        return False
    for entry in shards:
        if not isinstance(entry, dict) or entry.get("split") != "train":
            continue
        first = entry.get("first_block_id")
        last = entry.get("last_block_id")
        if isinstance(first, int) and isinstance(last, int) and first <= block <= last:
            return True
    return False


def discover_dataset(explicit: Path | None, block: int) -> tuple[Path, dict[str, Any]]:
    roots = [explicit.expanduser().resolve()] if explicit is not None else []
    if not roots:
        input_root = Path("/kaggle/input")
        if input_root.is_dir():
            roots = [p.parent for p in input_root.rglob("manifest.json")]
    matches: list[tuple[Path, dict[str, Any]]] = []
    for root in roots:
        manifest_path = root / "manifest.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if isinstance(manifest, dict) and manifest_has_block(manifest, block):
            matches.append((root, manifest))
    if not matches:
        raise SystemExit(
            f"Could not find the attached 2048x16 training dataset containing train block {block}. "
            "Pass --dataset-dir /kaggle/input/<500m-dataset-dir>."
        )
    # The 500M dataset is the one expected to contain block 4000; if more than
    # one matches, choose the manifest with the largest accepted-source count.
    matches.sort(key=lambda item: int(item[1].get("accepted_source_tokens", 0)), reverse=True)
    print(f"[dataset] {matches[0][0]}", flush=True)
    return matches[0]


def locate_block(root: Path, manifest: dict[str, Any], block: int) -> tuple[Path, int]:
    for raw in manifest["shards"]:
        if not isinstance(raw, dict) or raw.get("split") != "train":
            continue
        first = raw.get("first_block_id")
        last = raw.get("last_block_id")
        if isinstance(first, int) and isinstance(last, int) and first <= block <= last:
            filename = raw.get("filename")
            if not isinstance(filename, str):
                break
            path = (root / filename).resolve()
            if not path.is_file():
                raise SystemExit(f"manifest shard missing: {path}")
            local_sequence = (block - first) * SEQUENCES_PER_BLOCK
            return path, local_sequence
    raise SystemExit(f"train block {block} was not found in manifest shards")


def read_microbatch(shard: Path, local_sequence: int, batch_size: int) -> list[list[int]]:
    if not 1 <= batch_size <= SEQUENCES_PER_BLOCK:
        raise SystemExit(f"--batch-size must be 1..{SEQUENCES_PER_BLOCK}")
    count = batch_size * STORED_TOKENS
    offset_tokens = local_sequence * STORED_TOKENS
    with shard.open("rb") as handle:
        handle.seek(offset_tokens * 2)
        payload = handle.read(count * 2)
    if len(payload) != count * 2:
        raise SystemExit(
            f"short read from {shard}: got {len(payload)} bytes, expected {count * 2}"
        )
    values = array("H")
    values.frombytes(payload)
    if sys.byteorder != "little":
        values.byteswap()
    rows = []
    for index in range(batch_size):
        start = index * STORED_TOKENS
        rows.append(list(values[start : start + CONTEXT]))
    return rows


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        return float("nan")
    if len(sorted_values) == 1:
        return sorted_values[0]
    position = q * (len(sorted_values) - 1)
    lo = math.floor(position)
    hi = math.ceil(position)
    if lo == hi:
        return sorted_values[lo]
    frac = position - lo
    return sorted_values[lo] * (1 - frac) + sorted_values[hi] * frac


def tensor_stats(torch: Any, tensor: Any) -> dict[str, Any]:
    flat = tensor.detach().float().cpu().reshape(-1).tolist()
    flat.sort()
    total = len(flat)
    return {
        "count": total,
        "min": flat[0],
        "p01": quantile(flat, 0.01),
        "p05": quantile(flat, 0.05),
        "p50": quantile(flat, 0.50),
        "p95": quantile(flat, 0.95),
        "p99": quantile(flat, 0.99),
        "max": flat[-1],
        "fraction_le_minus_0_50": sum(value <= BOUNDARY_PASS for value in flat) / total,
        "fraction_le_minus_0_75": sum(value <= BOUNDARY_FAIL for value in flat) / total,
    }


def chunk_stats(torch: Any, tensor: Any) -> dict[str, Any]:
    # tensor: [B,T,H,K], log decay <= 0. Compare real 64-token regions with
    # the synthetic constant-g sweep using cumulative magnitude -sum(g).
    values: list[float] = []
    means: list[float] = []
    data = tensor.detach().float().cpu()
    for start in range(0, data.shape[1], 64):
        chunk = data[:, start : start + 64]
        if chunk.shape[1] == 0:
            continue
        magnitude = -chunk.sum(dim=1)  # [B,H,K]
        mean_decay = chunk.mean(dim=1)  # [B,H,K]
        values.extend(magnitude.reshape(-1).tolist())
        means.extend(mean_decay.reshape(-1).tolist())
    values.sort()
    means.sort()
    total = len(values)
    return {
        "count": total,
        "cumulative_magnitude_p50": quantile(values, 0.50),
        "cumulative_magnitude_p95": quantile(values, 0.95),
        "cumulative_magnitude_p99": quantile(values, 0.99),
        "cumulative_magnitude_max": values[-1],
        "mean_log_decay_p01": quantile(means, 0.01),
        "mean_log_decay_p50": quantile(means, 0.50),
        "fraction_mean_le_minus_0_50": sum(value <= BOUNDARY_PASS for value in means) / total,
        "fraction_mean_le_minus_0_75": sum(value <= BOUNDARY_FAIL for value in means) / total,
    }


class TelemetryBackend:
    def __init__(self, torch: Any, label: str, delegate: Any, sink: dict[str, Any]) -> None:
        self.torch = torch
        self.label = label
        self.delegate = delegate
        self.sink = sink
        self.chunk_size = getattr(delegate, "chunk_size", None)

    def __call__(self, q, k, v, log_decay, erase_gate, write_gate, initial_state=None):
        self.sink[self.label] = {
            "log_decay": tensor_stats(self.torch, log_decay),
            "chunks64": chunk_stats(self.torch, log_decay),
        }
        return self.delegate(q, k, v, log_decay, erase_gate, write_gate, initial_state)


def main() -> int:
    args = parse_args()
    import torch
    from model.config import ModelConfig
    from model.gdn2_stable import StableGatedDeltaNet2
    from model.model import SmallLLM

    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU required")
    ensure_fla()

    checkpoint_path = discover_checkpoint(args.checkpoint)
    dataset_root, manifest = discover_dataset(args.dataset_dir, args.block)
    shard, local_sequence = locate_block(dataset_root, manifest, args.block)
    rows = read_microbatch(shard, local_sequence, args.batch_size)

    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(raw, dict) or not isinstance(raw.get("model"), dict) or not isinstance(raw.get("model_config"), dict):
        raise SystemExit("checkpoint must contain `model` and `model_config`")
    config = ModelConfig(**raw["model_config"])
    if config.max_seq_len != CONTEXT:
        raise SystemExit(f"checkpoint max_seq_len={config.max_seq_len}, expected {CONTEXT}")

    model = SmallLLM(config)
    model.load_state_dict(raw["model"], strict=True)
    model = model.cuda().eval()

    telemetry: dict[str, Any] = {}
    gdn_index = 0
    for block_index, (kind, block) in enumerate(zip(model.layer_kinds, model.blocks, strict=True)):
        if kind not in {"gdn", "gdn-2"}:
            continue
        if not isinstance(block.mixer, StableGatedDeltaNet2):
            raise SystemExit(f"unexpected GDN mixer at block {block_index}: {type(block.mixer).__name__}")
        label = f"block_{block_index}_gdn_{gdn_index}"
        block.mixer.backend = TelemetryBackend(torch, label, block.mixer.backend, telemetry)
        gdn_index += 1

    input_ids = torch.tensor(rows, dtype=torch.long, device="cuda")
    print("=" * 78)
    print("Small-LLM step-4000 real-data GDN-2 decay telemetry")
    print("=" * 78)
    print(f"checkpoint: {checkpoint_path}")
    print(f"checkpoint_global_step: {raw.get('global_step')}")
    print(f"dataset_block: {args.block}  shard: {shard.name}  batch: {args.batch_size}x{CONTEXT}")
    print("mode: forward-only, fp32 parameters + fp16 autocast, FLA forward")

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.float16):
        _ = model(input_ids)

    any_individual_fail_region = False
    any_chunk_fail_region = False
    for label, row in telemetry.items():
        log = row["log_decay"]
        chunks = row["chunks64"]
        any_individual_fail_region |= log["fraction_le_minus_0_75"] > 0
        any_chunk_fail_region |= chunks["fraction_mean_le_minus_0_75"] > 0
        print(
            f"[{label}] g p01={log['p01']:.4f} p50={log['p50']:.4f} "
            f"p99={log['p99']:.4f} min={log['min']:.4f} | "
            f"frac(g<=-0.75)={100*log['fraction_le_minus_0_75']:.3f}% | "
            f"64tok cum p95={chunks['cumulative_magnitude_p95']:.2f} "
            f"max={chunks['cumulative_magnitude_max']:.2f} | "
            f"frac(mean64<=-0.75)={100*chunks['fraction_mean_le_minus_0_75']:.3f}%",
            flush=True,
        )

    report = {
        "probe": "step4000_real_data_gdn_decay_telemetry",
        "checkpoint": str(checkpoint_path),
        "checkpoint_global_step": raw.get("global_step"),
        "dataset_root": str(dataset_root),
        "dataset_block": args.block,
        "shard": str(shard),
        "batch_size": args.batch_size,
        "context": CONTEXT,
        "synthetic_amp_sweep": {
            "last_passing_tested_constant_g": BOUNDARY_PASS,
            "first_failing_tested_constant_g": BOUNDARY_FAIL,
            "first_failing_64tok_cumulative_magnitude": 48.0,
        },
        "layers": telemetry,
        "summary": {
            "any_individual_g_le_minus_0_75": any_individual_fail_region,
            "any_64tok_mean_g_le_minus_0_75": any_chunk_fail_region,
            "interpretation": (
                "real checkpoint overlaps the tested FLA failure region; do not resume chunk-GDN2 training"
                if any_chunk_fail_region
                else "real 64-token means do not reach the tested constant-g=-0.75 failure point in this microbatch; direct real-checkpoint backward remains required before resume"
            ),
        },
    }
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")

    print("=" * 78)
    print("SUMMARY")
    print("=" * 78)
    print(f"any_individual_g_le_minus_0.75: {any_individual_fail_region}")
    print(f"any_64tok_mean_g_le_minus_0.75: {any_chunk_fail_region}")
    print(report["summary"]["interpretation"])
    print(f"JSON report: {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
