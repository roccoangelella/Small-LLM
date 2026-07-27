"""Little-endian uint16 token codec and end-of-document handling.

Tokens are written as unsigned 16-bit integers with an explicit little-endian
byte order, regardless of the host machine's native byte order.  There is no
header, no JSON framing, and no compression; documents are separated by the
end-of-document token (50256).
"""

from __future__ import annotations

import array
import sys
from collections.abc import Iterable

from dataset import config


def tokens_to_uint16_le_bytes(tokens: Iterable[int]) -> bytes:
    """Encode tokens as explicit little-endian uint16, two bytes each."""

    arr = array.array("H", tokens)
    if arr.itemsize != 2:
        raise RuntimeError(
            f"this Python build uses {arr.itemsize}-byte unsigned shorts; "
            "cannot produce the required uint16 corpus"
        )
    if sys.byteorder != "little":
        arr.byteswap()
    return arr.tobytes()


def insert_eod(tokens: list[int], *, eod: int = config.EOD_TOKEN_ID) -> list[int]:
    """Return tokens plus the EOD marker, duplicated only when needed.

    50256 is appended unless the document already ends with it, so two adjacent
    documents never share a single separator.
    """

    if tokens and tokens[-1] == eod:
        return tokens
    return tokens + [eod]


def decode_uint16_le(data: bytes) -> list[int]:
    """Decode explicit little-endian uint16 back to a list of token IDs."""

    if len(data) % 2 != 0:
        raise ValueError("byte length is not a multiple of two; corrupt uint16 stream")
    arr = array.array("H")
    arr.frombytes(data)
    if sys.byteorder != "little":
        arr.byteswap()
    return arr.tolist()
