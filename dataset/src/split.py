"""Deterministic, order-independent train/validation assignment.

Each accepted document is assigned by a stable SHA-256 hash of the versioned
seed plus the record's permanent source identity (revision + filename +
absolute record-start byte offset).  The result is independent of processing
order, identical after resume, identical on another machine, and never places a
document in both outputs (the assignment is a pure function of identity).
"""

from __future__ import annotations

import hashlib
import math

from dataset import config


def _component(value: str) -> bytes:
    encoded = value.encode("utf-8")
    return len(encoded).to_bytes(8, "big") + encoded


def is_validation(
    *,
    seed: str,
    revision: str,
    filename: str,
    record_start: int,
    probability: float,
) -> bool:
    """Return True iff this document belongs to the validation split."""

    if probability <= 0.0:
        return False
    if probability >= 1.0:
        return True
    if not math.isfinite(probability):
        raise ValueError("validation probability must be finite")
    if record_start < 0:
        raise ValueError("record_start must be non-negative")
    # Length-prefix every variable component so identities cannot become
    # ambiguous when filenames or future seed formats contain separators.
    material = b"".join(
        (
            _component(config.SPLIT_HASH_VERSION),
            _component(seed),
            _component(revision),
            _component(filename),
            record_start.to_bytes(8, "big", signed=False),
        )
    )
    digest = hashlib.sha256(material).digest()
    draw = int.from_bytes(digest[:8], "big")
    threshold = int(probability * (1 << 64))
    return draw < threshold
