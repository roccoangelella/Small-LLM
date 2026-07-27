"""Streaming source access, GPT-2 decoding, and stable sampling identities."""

from __future__ import annotations

import hashlib
import logging
import math
import struct
import sys
from array import array
from functools import lru_cache
from typing import Any, Iterator

from dataset import config

from .models import SourceDocument


LOGGER = logging.getLogger(__name__)


def iter_climbmix_documents() -> Iterator[SourceDocument]:
    """Stream only NVIDIA's root tokenized JSONL shards in a stable order."""

    try:
        from datasets import load_dataset
    except ImportError as error:
        raise RuntimeError(
            "The streaming dependency is missing. Run `uv sync` before using the pipeline."
        ) from error

    data_url = (
        f"hf://datasets/{config.DATASET_REPOSITORY}@{config.DATASET_REVISION}/"
        f"{config.DATASET_DATA_FILES_GLOB}"
    )
    LOGGER.info("Opening streaming source %s at revision %s", config.DATASET_REPOSITORY, config.DATASET_REVISION)
    dataset = load_dataset(
        "json",
        data_files={config.DATASET_SPLIT: data_url},
        split=config.DATASET_SPLIT,
        streaming=True,
    )
    for source_index, row in enumerate(dataset):
        try:
            cluster_id = int(row["cluster_id"])
            tokens = [int(value) for value in row["tokens"]]
        except (KeyError, TypeError, ValueError) as error:
            raise ValueError(f"Malformed source row at index {source_index}") from error
        token_count = int(row.get("token_count", len(tokens)))
        if token_count != len(tokens):
            LOGGER.warning(
                "Token count mismatch at source index %d (%d metadata, %d values); using values",
                source_index,
                token_count,
                len(tokens),
            )
            token_count = len(tokens)
        yield SourceDocument(source_index, cluster_id, tokens, token_count)


@lru_cache(maxsize=1)
def get_tokenizer() -> Any:
    """Load NVIDIA's documented GPT-2 tokenizer only once per process."""

    try:
        import tiktoken
    except ImportError as error:
        raise RuntimeError(
            "The tokenizer dependency is missing. Run `uv sync` before using the pipeline."
        ) from error
    return tiktoken.get_encoding(config.TOKENIZER_ENCODING)


def decode_document(document: SourceDocument) -> str:
    """Decode a ClimbMix token sequence to the text used for filtering/output."""

    return get_tokenizer().decode(document.tokens)


def stable_document_id(document: SourceDocument) -> str:
    """Return a content-based stable ID without keeping raw tokens in artifacts."""

    digest = hashlib.sha256()
    digest.update(config.RANDOM_SEED.encode("utf-8"))
    digest.update(struct.pack("<I", document.cluster_id))
    packed_tokens = array("I", document.tokens)
    if sys.byteorder != "little":
        packed_tokens.byteswap()
    digest.update(packed_tokens.tobytes())
    return digest.hexdigest()


def stable_priority(document_id: str, namespace: str) -> int:
    """Return a reproducible uniform 64-bit priority; lower values win."""

    material = f"{config.RANDOM_SEED}:{namespace}:{document_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(material).digest()[:8], "big")


def deterministic_accept(document_id: str, rate: float, namespace: str = "select") -> bool:
    """Use a content hash for order-independent deterministic Bernoulli sampling."""

    if rate <= 0:
        return False
    if rate >= 1:
        return True
    return stable_priority(document_id, namespace) < math.floor(rate * (1 << 64))
