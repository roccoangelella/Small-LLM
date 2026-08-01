"""Read and normalize schema-v2 or restored Drive manifest geometry."""
from __future__ import annotations
import json
from pathlib import Path
from typing import Mapping

def load_manifest(path: Path) -> Mapping[str, object]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError) as error:
        raise ValueError(f"cannot read dataset manifest: {path}") from error
    if not isinstance(manifest, Mapping):
        raise ValueError("dataset manifest must be an object")
    return manifest

def resolve_geometry(manifest: Mapping[str, object], *, context_length: int | None,
                     sequences_per_block: int | None) -> tuple[int, int]:
    ordinary = manifest.get("schema_version") == 2
    drive = manifest.get("version") == 1 and isinstance(manifest.get("shards"), list)
    if not ordinary and not drive:
        raise ValueError("manifest is neither schema-v2 nor a Drive shard manifest")
    if ordinary and manifest.get("sequence_format") != "context_plus_one":
        raise ValueError("dataset manifest does not use context_plus_one records")
    recorded_context = manifest.get("context_length") if ordinary else None
    if context_length is None:
        context_length = recorded_context
    elif recorded_context is not None and context_length != recorded_context:
        raise ValueError("requested context length disagrees with the manifest")
    if isinstance(context_length, bool) or not isinstance(context_length, int) or context_length <= 0:
        raise ValueError("Drive manifests require an explicit positive context length")
    if ordinary and manifest.get("stored_tokens_per_sequence") != context_length + 1:
        raise ValueError("dataset manifest has inconsistent sequence geometry")
    recorded_block = manifest.get("sequences_per_block")
    if recorded_block is None:
        if sequences_per_block is None:
            raise ValueError("manifest has no sequences_per_block; provide it explicitly")
        block_size = sequences_per_block
    else:
        if isinstance(recorded_block, bool) or not isinstance(recorded_block, int) or recorded_block <= 0:
            raise ValueError("dataset manifest has an invalid sequences_per_block")
        block_size = recorded_block
        if sequences_per_block is not None and sequences_per_block != block_size:
            raise ValueError("requested sequences_per_block disagrees with the manifest")
    if block_size <= 0:
        raise ValueError("sequences_per_block must be positive")
    return context_length, block_size
