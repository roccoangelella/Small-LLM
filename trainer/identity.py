"""Stable joint-checkpoint identities for model, trainer, and schema-v2 data."""

from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from pathlib import Path
from typing import Mapping

from .config import TrainerConfig


def _jsonable(value: object) -> object:
    if is_dataclass(value):
        return _jsonable(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    raise TypeError(f"checkpoint identity cannot encode {type(value).__name__}")


def canonical_hash(value: object) -> str:
    encoded = json.dumps(_jsonable(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _read_mapping(path: Path, *, label: str) -> Mapping[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"cannot read {label}: {path}") from error
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be an object")
    return payload


def checkpoint_identity(
    dataset_root: Path | str,
    *,
    model_config: object,
    trainer_config: TrainerConfig,
    manifest_path: Path | str | None = None,
    context_length: int | None = None,
    sequences_per_block: int | None = None,
) -> tuple[str, str, str]:
    """Return new-run coordinator identities from cache or Drive metadata."""

    root = Path(dataset_root)
    path = Path(manifest_path) if manifest_path is not None else root / "manifest.json"
    manifest = _read_mapping(path, label="dataset manifest")
    ordinary = manifest.get("schema_version") == 2
    drive = manifest.get("version") == 1 and isinstance(manifest.get("shards"), list)
    if not ordinary and not drive:
        raise ValueError("checkpoint identity requires schema-v2 or Drive shard metadata")
    production = manifest.get("production")
    production = production if isinstance(production, Mapping) else {}
    dataset_configuration_hash = (
        production.get("configuration_hash") if ordinary else manifest.get("configuration_hash")
    )
    configuration_hash = canonical_hash({
        "version": 1,
        "model": model_config,
        "trainer": trainer_config.as_dict(),
        "dataset_configuration_hash": dataset_configuration_hash,
    })

    work_plan_hash = manifest.get("work_plan_hash")
    if isinstance(work_plan_hash, str) and len(work_plan_hash) == 64:
        source_hash = work_plan_hash
    else:
        source_hash = canonical_hash([
            {
                "filename": shard.get("filename"),
                "byte_size": shard.get("byte_size"),
                "checksum": shard.get("checksum", shard.get("local_sha256")),
            }
            for shard in manifest.get("shards", [])
            if isinstance(shard, Mapping)
        ])

    schema = production.get("schema_hash") if ordinary else manifest.get("schema_hash")
    if isinstance(schema, str) and len(schema) == 64:
        schema_hash = schema
    else:
        manifest_context = manifest.get("context_length") if ordinary else context_length
        block_size = manifest.get("sequences_per_block") if ordinary else sequences_per_block
        schema_hash = canonical_hash({
            "schema_version": 2,
            "sequence_format": "context_plus_one",
            "context_length": manifest_context,
            "stored_tokens_per_sequence": (
                manifest.get("stored_tokens_per_sequence")
                if ordinary
                else manifest_context + 1 if isinstance(manifest_context, int) else None
            ),
            "sequences_per_block": block_size,
            "int_type": "uint16",
            "byte_order": "little",
        })
    return configuration_hash, source_hash, schema_hash


def saved_checkpoint_identity(checkpoint_root: Path | str) -> tuple[str, str, str]:
    """Read the immutable identities already recorded by a local restore."""

    payload = _read_mapping(Path(checkpoint_root) / "checkpoint.json", label="checkpoint metadata")
    result: list[str] = []
    for key in ("configuration_hash", "source_hash", "schema_hash"):
        value = payload.get(key)
        if not isinstance(value, str) or len(value) != 64:
            raise ValueError(f"checkpoint metadata has an invalid {key}")
        result.append(value)
    return result[0], result[1], result[2]


__all__ = ["canonical_hash", "checkpoint_identity", "saved_checkpoint_identity"]
