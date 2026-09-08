"""Optional local artifacts and bounded profiling for dense/MoE pilots."""
from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict
import hashlib
from importlib import metadata
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time

import torch
from torch.nn import functional as F

from .identity import canonical_hash
from .precision import autocast_context


def file_hash(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def write_json(path: Path, value: object) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")
    os.replace(temporary, path)


def _version(package: str) -> str | None:
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def _model_hash(model) -> str:
    digest = hashlib.sha256()
    for name, value in model.state_dict().items():
        digest.update(name.encode())
        data = value.detach().cpu().contiguous().view(torch.uint8).numpy()
        digest.update(memoryview(data).cast("B"))
    return digest.hexdigest()


def _source_identity(commit: str | None) -> dict:
    root = Path(__file__).resolve().parents[1]
    digest = hashlib.sha256()
    files = sorted(path for directory in ("model", "MOE_model", "trainer", "dataset")
                   for path in (root / directory).rglob("*.py"))
    files.extend(path for path in (root / "pyproject.toml", root / "moe_pilot.py")
                 if path.is_file())
    for path in files:
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    try:
        head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root,
                                       text=True, stderr=subprocess.DEVNULL).strip()
        dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root,
                                            text=True, stderr=subprocess.DEVNULL).strip())
    except (FileNotFoundError, subprocess.CalledProcessError):
        head, dirty = None, None
    if commit and head and commit != head:
        raise ValueError("declared source commit differs from the checked-out source")
    return {"source_commit": commit or head, "source_tree_sha256": digest.hexdigest(),
            "source_tree_dirty": dirty}


def _expert_statistics(model) -> dict:
    """Sample existing gradients before validation clears them; no weight clones."""
    result = {}
    for name, module in model.named_modules():
        if not hasattr(module, "experts") or not hasattr(module, "router"):
            continue
        groups = [("router", module.router)] + [
            (f"expert_{i:02d}", expert) for i, expert in enumerate(module.experts)
        ]
        values, present = [], []
        for _, group in groups:
            parameters = list(group.parameters())
            gradients = [p.grad.detach() for p in parameters if p.grad is not None]
            weight_norm = torch.stack([p.detach().norm() for p in parameters]).norm()
            grad_norm = torch.stack([g.norm() for g in gradients]).norm() if gradients else weight_norm.new_zeros(())
            values.append(torch.stack((weight_norm, grad_norm)))
            present.append(len(gradients))
        # One device-to-host transfer for all sampled statistics of this layer.
        for (label, _), value, count in zip(groups, torch.stack(values).cpu().tolist(), present, strict=True):
            result[f"{name}/{label}"] = {
                "weight_l2": value[0], "gradient_l2_after_clipping": value[1] if count else None,
                "gradient_tensors_present": count,
            }
    return result


class RunObservation:
    """Owns artifacts only; never updates parameters, controller or data cursor."""

    def __init__(self, args, engine, model_config, trainer_config, validation_reader):
        self.args, self.engine = args, engine
        self.root = Path(args.experiment_dir)
        self.root.mkdir(parents=True, exist_ok=True)
        manifest = Path(args.dataset_manifest or (args.dataset_dir / "manifest.json"))
        model_identity = (model_config.as_dict() if hasattr(model_config, "as_dict")
                          else asdict(model_config))
        identity = {
            "version": 1, "model": model_identity, **_source_identity(args.source_commit),
            "trainer": trainer_config.as_dict(), "dataset_manifest_sha256": file_hash(manifest),
            "initialization": args.initialization, "sequences_per_block": args.sequences_per_block,
        }
        identity_path = self.root / "identity.json"
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != json.loads(json.dumps(identity)):
                raise ValueError("experiment directory belongs to a different model, recipe or dataset")
        else:
            write_json(identity_path, identity)
        self.segment = Path(tempfile.mkdtemp(
            prefix=f"from-{engine.global_step:08d}-", dir=self.root
        ))
        self.started = time.perf_counter()
        self.tokens_at_start = engine.consumed_tokens
        self.successful_update_seconds = 0.
        self.profiled_steps = set(args.profile_at_steps)
        self.probe = None
        if args.probe_sequences:
            reader = validation_reader(args, model_config)
            batch = next(reader.iter_from_start(1))
            if args.probe_sequences > batch.sequence_count:
                raise ValueError("requested probe exceeds the first validation block")
            self.probe = {
                "input_ids": batch.input_ids[:args.probe_sequences].cpu().clone(),
                "labels": batch.labels[:args.probe_sequences].cpu().clone(),
                "target_mask": batch.labels[:args.probe_sequences].ne(-100).cpu().clone(),
                "split": "validation", "block_id": batch.block_id,
                "dataset_manifest_sha256": identity["dataset_manifest_sha256"],
            }
            self.probe["sha256"] = canonical_hash({
                "inputs": self.probe["input_ids"].tolist(),
                "labels": self.probe["labels"].tolist(),
                "target_mask": self.probe["target_mask"].tolist(),
                "source": identity["dataset_manifest_sha256"],
                "block_id": batch.block_id,
            })
            probe_path = self.root / "probe.pt"
            if probe_path.exists():
                saved = torch.load(probe_path, map_location="cpu", weights_only=True)
                if saved["sha256"] != self.probe["sha256"]:
                    raise ValueError("frozen probe differs from the current validation data")
            else:
                self._save_tensor_artifact(probe_path, self.probe)

        manifest_payload = {
            **identity,
            "model_class": f"{type(engine.model).__module__}.{type(engine.model).__name__}",
            "model_state_sha256": _model_hash(engine.model),
            "parameters": sum(p.numel() for p in engine.model.parameters()),
            "optimizer_roles": {
                str(group.get("optimizer_role", "adamw")):
                sum(p.numel() for p in group["params"]) for group in engine.optimizer.param_groups
            },
            "versions": {p: _version(p) for p in ("torch", "triton", "fla-core")},
            "device": str(engine.device), "torch_cuda_version": torch.version.cuda,
            "cuda_capability": (torch.cuda.get_device_capability(engine.device)
                                if engine.device.type == "cuda" else None),
            "gpu": torch.cuda.get_device_name(engine.device) if engine.device.type == "cuda" else None,
            "float32_matmul_precision": torch.get_float32_matmul_precision(),
            "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "start_step": engine.global_step, "start_tokens": engine.consumed_tokens,
            "probe_sha256": self.probe["sha256"] if self.probe else None,
            "profile_at_steps": args.profile_at_steps,
            "checkpoint_at_steps": args.checkpoint_at_steps,
            "probe_lm_logits": args.probe_lm_logits,
        }
        if hasattr(engine, "parameter_accounting"):
            manifest_payload["parameter_accounting"] = engine.parameter_accounting.as_dict()
        write_json(self.segment / "manifest.json", manifest_payload)
        print(json.dumps({"training_identity": manifest_payload}, sort_keys=True), flush=True)

    @staticmethod
    def _save_tensor_artifact(path, payload):
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        os.replace(temporary, path)

    def event(self, kind: str, **fields):
        with (self.segment / "events.jsonl").open("a") as handle:
            handle.write(json.dumps({"event": kind, **fields}, allow_nan=False) + "\n")

    @contextmanager
    def profile(self, step: int):
        if step not in self.profiled_steps:
            yield
            return
        activities = [torch.profiler.ProfilerActivity.CPU]
        if self.engine.device.type == "cuda":
            activities.append(torch.profiler.ProfilerActivity.CUDA)
        started = time.perf_counter()
        handles, regions = [], {}
        # Bounded forward labels expose mixer/router/expert/dispatch cost in the trace.
        for name, module in self.engine.model.named_modules():
            if not (name.endswith((".mixer", ".ffn", ".router"))
                    or (".experts." in name and name.rsplit(".", 1)[-1].isdigit())):
                continue
            def enter(_module, _inputs, name=name):
                region = torch.profiler.record_function(f"module/{name}")
                region.__enter__()
                regions[name] = region
            def leave(_module, _inputs, _output, name=name):
                region = regions.pop(name, None)
                if region is not None:
                    region.__exit__(None, None, None)
            handles.extend((module.register_forward_pre_hook(enter),
                            module.register_forward_hook(leave, always_call=True)))
        self.engine._profile_regions = True
        try:
            with torch.profiler.profile(activities=activities) as profiler:
                with torch.profiler.record_function("session_step"):
                    yield
        finally:
            self.engine._profile_regions = False
            for handle in handles:
                handle.remove()
        path = self.segment / f"profile-step-{step:08d}.json"
        profiler.export_chrome_trace(str(path))
        sort_key = "self_cuda_time_total" if self.engine.device.type == "cuda" else "self_cpu_time_total"
        (path.with_suffix(".txt")).write_text(profiler.key_averages().table(sort_by=sort_key))
        self.event("profile", step=step, capture_and_export_seconds=time.perf_counter() - started,
                   byte_size=path.stat().st_size)

    def training(self, metrics):
        started = time.perf_counter()
        self.successful_update_seconds += metrics.elapsed_seconds
        sampled = {}
        if metrics.step in self.args.checkpoint_at_steps:
            sampled = _expert_statistics(self.engine.model)
        self.event("training", profiled=metrics.step in self.profiled_steps,
                   expert_statistics=sampled,
                   statistics_seconds=time.perf_counter() - started, **metrics.as_dict())

    def checkpoint(self, checkpoint_id: str, *, elapsed_seconds: float, byte_size: int | None):
        self.event("checkpoint", checkpoint_id=checkpoint_id,
                   elapsed_seconds=elapsed_seconds, byte_size=byte_size)
        if self.probe is None:
            return
        output = self.root / f"{checkpoint_id}-probe.pt"
        if output.exists():
            return
        started = time.perf_counter()
        model, device = self.engine.model, self.engine.device
        routes, handles = {}, []
        # Hooks exist only for the fixed probe, not in the training hot path.
        for name, module in model.named_modules():
            if hasattr(module, "selection_bias") and hasattr(module, "token_exposure"):
                routes[name] = []
                def capture(_module, _inputs, route, name=name):
                    routes[name].append({
                        "logits": route.logits.cpu(),
                        "expert_indices": route.expert_indices.cpu(),
                        "selected_probabilities": route.selected_probabilities.cpu(),
                        "selection_bias": _module.selection_bias.detach().cpu().clone(),
                    })
                handles.append(module.register_forward_hook(capture))
        was_training = model.training
        model.eval()
        ce, logits_saved = [], []
        try:
            with torch.inference_mode():
                for inputs, labels in zip(self.probe["input_ids"], self.probe["labels"], strict=True):
                    with autocast_context(self.engine.config.precision, device):
                        logits = model(inputs.unsqueeze(0).to(device))
                    losses = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]),
                                             labels.to(device), reduction="none")
                    ce.append(losses.cpu())
                    if self.args.probe_lm_logits:
                        logits_saved.append(logits.squeeze(0).to(device="cpu", dtype=torch.float16))
        finally:
            for handle in handles:
                handle.remove()
            model.train(was_training)
        payload = {
            "version": 1, "checkpoint_id": checkpoint_id, "step": self.engine.global_step,
            "checkpoint_manifest_sha256": file_hash(
                Path(self.args.checkpoint_dir) / checkpoint_id / "checkpoint.json"),
            "consumed_tokens": self.engine.consumed_tokens, "probe_sha256": self.probe["sha256"],
            "per_token_ce_fp32": torch.stack(ce), "routing": routes,
            "target_mask": self.probe["target_mask"],
        }
        if logits_saved:
            payload["lm_logits_fp16"] = torch.stack(logits_saved)
        self._save_tensor_artifact(output, payload)
        self.event("probe", checkpoint_id=checkpoint_id, byte_size=output.stat().st_size,
                   elapsed_seconds=time.perf_counter() - started)

    def finish(self, *, failed: bool):
        elapsed = time.perf_counter() - self.started
        tokens = self.engine.consumed_tokens - self.tokens_at_start
        summary = {
            "failed": failed, "successful_target_tokens": tokens,
            "session_elapsed_seconds": elapsed,
            "successful_update_seconds": self.successful_update_seconds,
            "session_tokens_per_second": tokens / elapsed if elapsed else 0.,
            "end_step": self.engine.global_step,
            "billing": "Not measured here; provider billed seconds/resources and credits are separate.",
        }
        write_json(self.segment / "summary.json", summary)
