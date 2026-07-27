"""Append-only little-endian uint16 corpus writer with crash-safe checkpoints.

Two output files are written: ``train.bin`` and ``validation.bin``.  Each writer
accumulates a configurable in-memory buffer (256 MiB by default) and soft-flushes
to the OS when the buffer fills, so the process never holds the whole corpus in
RAM.  A *durable checkpoint* only occurs when the caller explicitly calls
:func:`checkpoint`, which soft-flushes both buffers, ``fsync`` s both files, and
returns the confirmed on-disk byte sizes.  Bytes written between checkpoints are
recoverable: on resume the caller truncates both files to the last confirmed
sizes recorded in ``progress.json``, so uncommitted tail bytes are discarded and
reprocessed without duplication.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import IO

from dataset import config

from .bitio import insert_eod, tokens_to_uint16_le_bytes


LOGGER = logging.getLogger(__name__)


class BinaryCorpusWriter:
    """Buffered writer for the two packed token streams."""

    def __init__(
        self,
        train_path: Path,
        validation_path: Path,
        *,
        buffer_bytes: int,
        resume_sizes: tuple[int, int] = (0, 0),
    ) -> None:
        if buffer_bytes <= 0:
            raise ValueError("buffer_bytes must be positive")
        self._train_path = train_path
        self._validation_path = validation_path
        self._buffer_bytes = buffer_bytes
        train_path.parent.mkdir(parents=True, exist_ok=True)
        validation_path.parent.mkdir(parents=True, exist_ok=True)
        # Open in binary append mode.  On resume the caller has already
        # truncated both files to the confirmed checkpoint sizes, so the next
        # append lands exactly at the committed tail.
        self._train: IO[bytes] = train_path.open("ab")
        self._validation: IO[bytes] = validation_path.open("ab")
        actual_sizes = (train_path.stat().st_size, validation_path.stat().st_size)
        if actual_sizes != resume_sizes:
            self._train.close()
            self._validation.close()
            raise RuntimeError(
                "binary files do not match the confirmed resume sizes: "
                f"actual={actual_sizes}, confirmed={resume_sizes}"
            )
        self._train_buf = bytearray()
        self._validation_buf = bytearray()
        self._train_disk_bytes = resume_sizes[0]
        self._validation_disk_bytes = resume_sizes[1]
        # Total written bytes (both writers) since the last durable checkpoint.
        self._written_since_checkpoint = 0

    @property
    def train_path(self) -> Path:
        return self._train_path

    @property
    def validation_path(self) -> Path:
        return self._validation_path

    @property
    def written_since_checkpoint(self) -> int:
        """Bytes (both writers) accumulated since the last durable checkpoint."""

        return self._written_since_checkpoint

    def append(self, *, validation: bool, tokens: list[int]) -> int:
        """Append one document's source tokens plus an EOD marker.

        Returns the number of *written* bytes (source tokens + inserted EOD),
        i.e. two bytes per token.
        """

        written_tokens = insert_eod(tokens)
        encoded = tokens_to_uint16_le_bytes(written_tokens)
        byte_len = len(encoded)
        self._written_since_checkpoint += byte_len
        if validation:
            self._validation_buf.extend(encoded)
            if len(self._validation_buf) >= self._buffer_bytes:
                self._soft_flush(self._validation, self._validation_buf)
                self._validation_buf = bytearray()
        else:
            self._train_buf.extend(encoded)
            if len(self._train_buf) >= self._buffer_bytes:
                self._soft_flush(self._train, self._train_buf)
                self._train_buf = bytearray()
        return byte_len

    def checkpoint(self) -> tuple[int, int]:
        """Flush both buffers, fsync both files, and return confirmed sizes.

        Only this call produces bytes that resume can trust: after it returns,
        ``train.bin`` and ``validation.bin`` on-disk sizes equal the returned
        tuple, and ``progress.json`` may safely record them.
        """

        self._soft_flush(self._train, self._train_buf)
        self._train_buf = bytearray()
        self._soft_flush(self._validation, self._validation_buf)
        self._validation_buf = bytearray()
        self._train.flush()
        self._validation.flush()
        os.fsync(self._train.fileno())
        os.fsync(self._validation.fileno())
        confirmed = (self._train_disk_bytes, self._validation_disk_bytes)
        self._written_since_checkpoint = 0
        return confirmed

    def final_flush(self) -> tuple[int, int]:
        """Flush and fsync for finalization (same durability as checkpoint)."""

        return self.checkpoint()

    def flush_uncommitted(self) -> tuple[int, int]:
        """Write buffers without fsync or declaring a durable checkpoint.

        This is primarily a crash-test hook.  It reproduces the important
        failure window where binary tail bytes exist but ``progress.json`` still
        points at the previous confirmed sizes.
        """

        self._soft_flush(self._train, self._train_buf)
        self._train_buf = bytearray()
        self._soft_flush(self._validation, self._validation_buf)
        self._validation_buf = bytearray()
        self._train.flush()
        self._validation.flush()
        return self._train_disk_bytes, self._validation_disk_bytes

    def _soft_flush(self, handle: IO[bytes], buf: bytearray) -> None:
        """Write a buffer straight to the OS without fsync."""

        if buf:
            written = handle.write(buf)
            if written != len(buf):
                raise OSError(
                    f"short binary write: expected {len(buf)} bytes, wrote {written}"
                )
            handle.flush()
            self._track_disk_bytes(handle, len(buf))

    def _track_disk_bytes(self, handle: IO[bytes], n: int) -> None:
        if handle is self._train:
            self._train_disk_bytes += n
        else:
            self._validation_disk_bytes += n

    def close(self) -> None:
        """Close both file handles without altering the final checkpoint."""

        for handle in (self._train, self._validation):
            if handle is not None and not handle.closed:
                handle.close()
        self._train = None
        self._validation = None
