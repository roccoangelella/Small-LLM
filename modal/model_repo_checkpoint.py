"""Install Modal's split Hugging Face checkpoint/model transport.

The filename is retained as a compatibility import for deployed Modal entrypoints,
but the ADR-0055 model-repository checkpoint override is intentionally gone:

* exact-resume ``latest`` uses the runtime's mutable private Storage Bucket;
* validation ``best`` uses a dedicated per-run model repository;
* the trainer, rather than Modal, owns fail-closed best-repository replacement.

Existing model-repository checkpoints remain restore-only migration sources through
``runtime._restore_hf_checkpoint_if_needed``. Dataset buckets are unaffected.
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any, Mapping

import runtime as base_runtime
from profiles import canonical_run_id

_ORIGINAL_TRAINER_COMMAND = base_runtime._trainer_command
_ORIGINAL_RUNTIME_CONTRACT = base_runtime._runtime_contract
_ORIGINAL_ASSERT_CONTRACT = base_runtime._assert_contract
_ORIGINAL_RUN_TRAINING = base_runtime.run_training
_INSTALLED = False
_SAFE_REPO_COMPONENT = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_run_suffix(run_id: str) -> str:
    suffix = _SAFE_REPO_COMPONENT.sub("-", run_id).strip("-._")
    if not suffix:
        raise RuntimeError("run ID cannot derive a dedicated best-model repository")
    return suffix


def _dedicated_best_repo_id(run_id: str) -> str:
    """Resolve a model repository that cannot alias the shared legacy repository."""

    explicit = os.environ.get("SMALL_LLM_HF_BEST_MODEL_REPO_ID", "").strip()
    base = base_runtime._hf_model_repo_id()
    suffix = _safe_run_suffix(run_id)
    if explicit:
        candidate = explicit
    else:
        owner, separator, name = base.partition("/")
        if not separator or not owner or not name:
            raise RuntimeError("SMALL_LLM_HF_REPO_ID must be in owner/name form")
        candidate = f"{owner}/{name}-best-{suffix}"
    owner, separator, name = candidate.partition("/")
    if not separator or not owner or not name or "/" in name:
        raise RuntimeError("best-model repository ID must be in owner/name form")
    if candidate == base:
        raise RuntimeError("best-model repository must not alias the shared legacy model repository")
    if not name.endswith(f"-best-{suffix}"):
        raise RuntimeError("best-model repository is not dedicated to the active run ID")
    if len(name) > 96:
        raise RuntimeError("derived best-model repository name exceeds the Hugging Face limit")
    return candidate


def _trainer_command_split_store(*args: Any, **kwargs: Any) -> list[str]:
    """Keep Bucket latest flags and add the dedicated replace-on-improvement best repo."""

    command = _ORIGINAL_TRAINER_COMMAND(*args, **kwargs)
    if not bool(kwargs.get("online")):
        return command
    run_id = kwargs.get("wandb_run_id")
    if not isinstance(run_id, str) or not run_id:
        raise RuntimeError("online Modal training has no stable run ID for best-model publication")
    required = {
        "--remote-checkpoint-bucket",
        "--remote-create-bucket",
        "--remote-rolling-latest-only",
    }
    missing = sorted(required - set(command))
    if missing:
        raise RuntimeError(f"Modal Bucket-latest trainer command is missing {missing}")
    if "--remote-checkpoint-repo" in command:
        raise RuntimeError("Modal latest checkpoint command still targets a model repository")
    command += [
        "--best-model-repo",
        _dedicated_best_repo_id(run_id),
        "--best-model-recreate",
    ]
    return command


def _runtime_contract_split_store(*args: Any, **kwargs: Any) -> dict[str, Any]:
    result = dict(_ORIGINAL_RUNTIME_CONTRACT(*args, **kwargs))
    model = kwargs.get("model")
    tokens = kwargs.get("tokens")
    if model is None or tokens is None:
        raise RuntimeError("Modal runtime contract lacks model/token presets")
    run_id = canonical_run_id(model, tokens)
    transports = result.get("checkpoint_transports")
    if not isinstance(transports, Mapping):
        raise RuntimeError("Modal runtime contract has no checkpoint transport mapping")
    result["checkpoint_transports"] = {
        **dict(transports),
        "best_model": (
            f"dedicated Hugging Face model repository {_dedicated_best_repo_id(run_id)}; "
            "delete/recreate only after strict validation-loss improvement"
        ),
    }
    if result.get("source_migration"):
        result["source_migration"] = "live latest moved to HF Bucket; best split to dedicated model repo"
    return result


def _assert_contract_split_store(path: Path, expected: Mapping[str, Any]) -> None:
    """Permit only the authorized transport migration around frozen scientific state."""

    _ORIGINAL_ASSERT_CONTRACT(path, expected)
    actual = base_runtime._json(path)
    desired = expected.get("checkpoint_transports")
    if actual.get("checkpoint_transports") != desired:
        actual["checkpoint_transports"] = desired
        actual["checkpoint_transport_migration"] = (
            "latest: shared HF model repo -> mutable latest-only HF Bucket; "
            "best: dedicated replace-on-improvement model repo"
        )
        base_runtime._write_json(path, actual)


def run_training(**kwargs: Any) -> dict[str, object]:
    """Run the frozen Modal trainer with Bucket latest and dedicated model best."""

    install_model_repo_checkpoint_transport()
    os.environ["SMALL_LLM_MODAL_SOURCE_COMMIT"] = str(kwargs["source_commit"])
    result = dict(_ORIGINAL_RUN_TRAINING(**kwargs))
    model_preset, token_preset = base_runtime.resolve_presets(
        str(kwargs["model"]),
        str(kwargs["tokens"]),
    )
    run_id = canonical_run_id(model_preset, token_preset)
    result["hf_checkpoint_transport"] = {
        "bucket": base_runtime._hf_checkpoint_bucket_id(),
        "cadence_steps": base_runtime.HF_REMOTE_EVERY,
        "retention": "latest_only_mutable_bucket",
    }
    result["hf_best_model_transport"] = {
        "repo": _dedicated_best_repo_id(run_id),
        "selection": "strict_validation_loss_improvement",
        "replacement": "delete_recreate_dedicated_repo",
    }
    return result


def install_model_repo_checkpoint_transport() -> None:
    """Install the split transport while preserving the historical import name."""

    global _INSTALLED
    if _INSTALLED:
        return
    base_runtime._trainer_command = _trainer_command_split_store
    base_runtime._runtime_contract = _runtime_contract_split_store
    base_runtime._assert_contract = _assert_contract_split_store
    base_runtime.run_training = run_training
    _INSTALLED = True


__all__ = [
    "_dedicated_best_repo_id",
    "install_model_repo_checkpoint_transport",
    "run_training",
]
