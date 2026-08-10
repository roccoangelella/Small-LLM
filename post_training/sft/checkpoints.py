"""Verified native-checkpoint loading and SFT identity helpers."""

from __future__ import annotations

import json
import pickle
from pathlib import Path
import tempfile
from typing import Mapping

import torch

from dataset.src.joint_checkpoint import verify_local_manifest
from dataset.src.remote import sha256_path
from model.config import ModelConfig
from model.model import SmallLLM
from trainer.identity import canonical_hash
from trainer.post_pretraining_prompt_suite import download_verified_checkpoint


def _read_json(path: Path, *, label: str) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise RuntimeError(f"{label} is missing or invalid: {path}") from error
    if not isinstance(payload, Mapping):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return dict(payload)


def _model_config(raw: Mapping[str, object]) -> ModelConfig:
    values = dict(raw)
    pattern = values.get("layer_pattern")
    if isinstance(pattern, list):
        values["layer_pattern"] = tuple(pattern)
    return ModelConfig(**values)  # type: ignore[arg-type]


def load_verified_native_checkpoint(
    root: Path | str,
    *,
    device: str | torch.device = "cpu",
) -> tuple[SmallLLM, ModelConfig, dict[str, object]]:
    """Load model weights only after the complete local checkpoint is verified."""

    checkpoint_root = Path(root)
    verify_local_manifest(checkpoint_root)
    checkpoint = _read_json(checkpoint_root / "checkpoint.json", label="checkpoint.json")
    with (checkpoint_root / "trainer_state.pkl").open("rb") as handle:
        state = pickle.load(handle)
    if not isinstance(state, Mapping) or state.get("version") != 1:
        raise RuntimeError("trainer_state.pkl has an unsupported state version")
    raw_config = state.get("model_config")
    model_state = state.get("model")
    if not isinstance(raw_config, Mapping) or not isinstance(model_state, Mapping):
        raise RuntimeError("checkpoint does not contain self-describing model weights")
    config = _model_config(raw_config)
    model = SmallLLM(config)
    model.load_state_dict(model_state, strict=True)
    model.to(torch.device(device))

    consumed = state.get("consumed_tokens")
    if isinstance(consumed, bool) or not isinstance(consumed, int) or consumed < 0:
        raise RuntimeError("parent checkpoint has an invalid consumed_tokens counter")
    checkpoint_id = checkpoint.get("checkpoint_id")
    if not isinstance(checkpoint_id, str) or not checkpoint_id:
        raise RuntimeError("parent checkpoint has no checkpoint_id")
    identity = {
        "checkpoint_id": checkpoint_id,
        "configuration_hash": checkpoint.get("configuration_hash"),
        "source_hash": checkpoint.get("source_hash"),
        "schema_hash": checkpoint.get("schema_hash"),
        "local_manifest_sha256": sha256_path(checkpoint_root / "local_manifest.json"),
        "trainer_state_sha256": sha256_path(checkpoint_root / "trainer_state.pkl"),
        "consumed_tokens": consumed,
        "model_config": dict(raw_config),
    }
    identity["identity_sha256"] = canonical_hash(identity)
    return model, config, identity


def download_parent_checkpoint(
    *,
    repo_id: str,
    run_id: str,
    pointer: str = "best",
    token: str | None = None,
    revision: str | None = None,
    destination: Path | str | None = None,
) -> tuple[Path, dict[str, object]]:
    """Download one verified published parent checkpoint."""

    if pointer not in {"best", "latest"}:
        raise ValueError("parent pointer must be best or latest")
    if destination is None:
        destination_path = Path(tempfile.mkdtemp(prefix="small-llm-parent-"))
    else:
        destination_path = Path(destination)
        destination_path.mkdir(parents=True, exist_ok=True)
    root, remote = download_verified_checkpoint(
        repo_id=repo_id,
        run_id=run_id,
        token=token,
        revision=revision,
        pointer_name=pointer,
        destination=destination_path,
    )
    return root, dict(remote)


def sft_checkpoint_hashes(
    *,
    parent_identity: Mapping[str, object],
    bundle_manifest: Mapping[str, object],
    trainer_config: Mapping[str, object],
    template_identity: str = "small-llm-s0-v1",
    loss_identity: str = "assistant-only-ce-v1",
) -> tuple[str, str, str]:
    """Build coordinator identities that bind parent, SFT data, and objective."""

    bundle_hash = bundle_manifest.get("manifest_sha256")
    if not isinstance(bundle_hash, str) or len(bundle_hash) != 64:
        raise ValueError("SFT bundle manifest has no valid identity")
    parent_hash = parent_identity.get("identity_sha256")
    if not isinstance(parent_hash, str) or len(parent_hash) != 64:
        raise ValueError("parent checkpoint has no valid identity")
    configuration_hash = canonical_hash(
        {
            "version": 1,
            "parent": parent_hash,
            "trainer": dict(trainer_config),
            "template": template_identity,
            "loss": loss_identity,
        }
    )
    source_hash = canonical_hash(
        {
            "version": 1,
            "bundle": bundle_hash,
            "train_split": bundle_manifest.get("splits", {}).get("train")
            if isinstance(bundle_manifest.get("splits"), Mapping)
            else None,
        }
    )
    schema_hash = canonical_hash(
        {
            "version": 1,
            "context_length": bundle_manifest.get("context_length"),
            "optimizer_target_tokens": bundle_manifest.get("optimizer_target_tokens"),
            "template": template_identity,
            "loss": loss_identity,
            "token_dtype": "uint16",
            "target_mask": "binary",
        }
    )
    return configuration_hash, source_hash, schema_hash


__all__ = [
    "download_parent_checkpoint",
    "load_verified_native_checkpoint",
    "sft_checkpoint_hashes",
]
