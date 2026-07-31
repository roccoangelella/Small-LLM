"""Candidate model initializers and a small deterministic comparison helper."""

from __future__ import annotations

import copy
import math
from collections.abc import Iterable
from typing import Any

import torch
from torch import Tensor, nn


def _is_preserved_gdn_name(name: str) -> bool:
    lowered = name.lower()
    return "a_log" in lowered or "dt_bias" in lowered


def _is_norm(module: nn.Module) -> bool:
    return module.__class__.__name__.lower().endswith("rmsnorm") or "norm" in module.__class__.__name__.lower()


def _is_output_projection(name: str, module: nn.Module, model: nn.Module) -> bool:
    lowered = name.lower()
    if not isinstance(module, nn.Linear):
        return False
    if any(token in lowered for token in ("out_proj", "o_proj", "output_proj", "mixer_output")):
        return True
    # SwiGLU's down projection is the residual FFN output.
    if lowered.endswith(".down") or lowered.endswith(".down_proj"):
        return ".blocks." in lowered or "ffn" in lowered
    return False


def _embedding_weight_ids(model: nn.Module) -> set[int]:
    embedding = getattr(model, "token_embedding", None)
    return {id(parameter) for parameter in embedding.parameters()} if embedding is not None else set()


def initialize_model(model: nn.Module, method: str = "normal") -> nn.Module:
    """Apply one candidate initializer in place and return ``model``.

    ``normal`` uses a GPT-style 0.02 standard deviation.  ``xavier`` uses
    Xavier uniform for matrix weights.  GDN's reference state parameters are
    intentionally left untouched, because their constructor owns the
    reference-range and inverse-softplus initialization.
    """

    method = method.lower()
    if method not in {"normal", "xavier"}:
        raise ValueError("method must be 'normal' or 'xavier'")
    gdn_state: dict[int, Tensor] = {}
    for name, parameter in model.named_parameters():
        if _is_preserved_gdn_name(name):
            gdn_state[id(parameter)] = parameter.detach().clone()

    embedding_ids = _embedding_weight_ids(model)
    with torch.no_grad():
        for name, parameter in model.named_parameters():
            if _is_preserved_gdn_name(name):
                continue
            if parameter.ndim == 1:
                if name.lower().endswith("weight") and "norm" in name.lower():
                    parameter.fill_(1.0)
                elif name.lower().endswith("bias"):
                    parameter.zero_()
                continue
            if method == "normal":
                nn.init.normal_(parameter, mean=0.0, std=0.02)
            else:
                nn.init.xavier_uniform_(parameter)
            if id(parameter) in embedding_ids:
                semantic = getattr(getattr(model, "config", None), "semantic_vocab_size", parameter.shape[0])
                if parameter.shape[0] > semantic:
                    parameter[semantic:].zero_()

        for name, module in model.named_modules():
            if _is_norm(module) and hasattr(module, "weight"):
                module.weight.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                if not _is_preserved_gdn_name(f"{name}.bias"):
                    module.bias.zero_()

        layers = max(1, len(getattr(model, "blocks", ())))
        residual_scale = 1.0 / math.sqrt(2.0 * layers)
        for name, module in model.named_modules():
            if _is_output_projection(name, module, model):
                module.weight.mul_(residual_scale)

        for name, parameter in model.named_parameters():
            saved = gdn_state.get(id(parameter))
            if saved is not None:
                parameter.copy_(saved)
            if id(parameter) in embedding_ids and parameter.ndim >= 2:
                semantic = getattr(getattr(model, "config", None), "semantic_vocab_size", parameter.shape[0])
                if parameter.shape[0] > semantic:
                    parameter[semantic:].zero_()
    return model


def initialize_normal(model: nn.Module) -> nn.Module:
    return initialize_model(model, "normal")


def initialize_xavier(model: nn.Module) -> nn.Module:
    return initialize_model(model, "xavier")


# Short names make the two candidates convenient to pass as experiment
# callables while keeping ``initialize_model`` as the explicit API.
normal = initialize_normal
xavier = initialize_xavier


def _experiment_metrics(model: nn.Module, input_ids: Tensor, targets: Tensor) -> dict[str, float | bool]:
    model.zero_grad(set_to_none=True)
    logits = model(input_ids)
    variance = logits.float().var(unbiased=False)
    loss = torch.nn.functional.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
    loss.backward()
    squared = [parameter.grad.detach().float().pow(2).sum() for parameter in model.parameters() if parameter.grad is not None]
    gradient_norm = torch.sqrt(torch.stack(squared).sum()) if squared else torch.zeros(())
    finite = bool(torch.isfinite(logits).all() and torch.isfinite(loss) and torch.isfinite(gradient_norm))
    return {
        "forward_variance": float(variance.detach()),
        "loss": float(loss.detach()),
        "gradient_norm": float(gradient_norm.detach()),
        "finite": finite,
        "overflow": not finite,
    }


def compare_initializations(
    model_or_factory: nn.Module | Any,
    input_ids: Tensor | None = None,
    targets: Tensor | None = None,
    *,
    seed: int = 0,
) -> dict[str, dict[str, float | bool]]:
    """Compare both candidates without deciding which architecture is best.

    A model is cloned for each candidate, so the caller's weights and training
    state are not changed.  When no batch is supplied, deterministic integer
    inputs are made from the model's semantic vocabulary size.
    """

    results: dict[str, dict[str, float | bool]] = {}
    # ``manual_seed`` updates every CUDA generator, so preserve every visible
    # CUDA stream rather than just the prototype's current device.  The scope
    # also covers factory construction, which may itself consume randomness.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    with torch.random.fork_rng(devices=devices):
        prototype = model_or_factory if isinstance(model_or_factory, nn.Module) else model_or_factory()
        config = getattr(prototype, "config", None)
        parameter = next(prototype.parameters(), None)
        device = parameter.device if parameter is not None else torch.device("cpu")
        if input_ids is None:
            vocab = int(getattr(config, "semantic_vocab_size", 32))
            experiment_inputs = torch.arange(8, device=device, dtype=torch.long).reshape(2, 4) % vocab
        else:
            if input_ids.device != device:
                raise ValueError("input_ids must be on the model device")
            experiment_inputs = input_ids
        if targets is None:
            experiment_targets = experiment_inputs.clone()
        else:
            if targets.device != device:
                raise ValueError("targets must be on the model device")
            experiment_targets = targets
        for method in ("normal", "xavier"):
            # A common seed makes differences attributable to the initializer,
            # and fork_rng leaves a caller's training RNG untouched.
            torch.manual_seed(seed)
            candidate = copy.deepcopy(prototype)
            initialize_model(candidate, method)
            candidate.eval()
            results[method] = _experiment_metrics(candidate, experiment_inputs, experiment_targets)
    return results


__all__ = [
    "compare_initializations",
    "initialize_model",
    "initialize_normal",
    "initialize_xavier",
    "normal",
    "xavier",
]
