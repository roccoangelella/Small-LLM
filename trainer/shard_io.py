"""Verified reads from immutable schema-v2 shard locations."""
from __future__ import annotations
from collections.abc import Sequence
from .shard_layout import BlockLocation, file_hash
from .types import TokenBatch

def read_location(
    reader: object,
    item: BlockLocation,
    *,
    peer_locations: Sequence[BlockLocation] | None = None,
) -> TokenBatch:
    root = reader.root
    locations = tuple(peer_locations) if peer_locations is not None else reader._locations
    verified = reader._verified
    if item.path not in verified:
        if item.path.is_symlink() or not item.path.is_file():
            raise FileNotFoundError(f"missing shard {item.path}; restore or prefetch it before continuing")
        try:
            item.path.resolve().relative_to(root.resolve())
        except ValueError as error:
            raise RuntimeError(f"shard escapes dataset root: {item.path}") from error
        matching = [x for x in locations if x.path == item.path]
        if not matching:
            raise RuntimeError(f"no shard geometry is available for {item.path}")
        expected_size = max(x.offset + x.byte_size for x in matching)
        if item.path.stat().st_size != expected_size:
            raise RuntimeError(f"schema-v2 shard size mismatch: {item.path}")
        if reader.verify_checksums and file_hash(item.path) != item.checksum:
            raise RuntimeError(f"schema-v2 shard checksum mismatch: {item.path}")
        verified.add(item.path)
    with item.path.open("rb") as handle:
        handle.seek(item.offset)
        payload = handle.read(item.byte_size)
    if len(payload) != item.byte_size:
        raise RuntimeError(f"short read from schema-v2 shard: {item.path}")
    block = type("Block", (), {"schema_version": 2, "split": reader.split,
        "block_id": item.block_id, "sequence_count": item.sequences,
        "token_count": item.sequences * (reader.context_length + 1),
        "cumulative_source_tokens": 0, "payload": payload})()
    batch = reader.decoder.decode(block)
    return TokenBatch(batch.block_id, batch.split, batch.input_ids, batch.labels,
                      batch.sequence_count, batch.target_token_count, None)
