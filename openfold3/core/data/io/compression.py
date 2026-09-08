# Copyright 2026 AlQuraishi Laboratory
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Compression helpers whose readers use magic bytes instead of suffixes."""

from __future__ import annotations

import gzip
import lzma
import os
import tempfile
from pathlib import Path

import zstandard

_GZIP_MAGIC = b"\x1f\x8b"
_XZ_MAGIC = b"\xfd7zXZ\x00"
_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"


def read_text_auto(path: Path) -> str:
    """Read plain, gzip, xz, or zstd UTF-8 text based on file magic."""
    path = Path(path)
    with open(path, "rb") as handle:
        magic = handle.read(6)

    if magic.startswith(_GZIP_MAGIC):
        with gzip.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    if magic.startswith(_XZ_MAGIC):
        with lzma.open(path, "rt", encoding="utf-8") as handle:
            return handle.read()
    if magic.startswith(_ZSTD_MAGIC):
        with (
            open(path, "rb") as source,
            zstandard.ZstdDecompressor().stream_reader(source) as reader,
        ):
            return reader.read().decode("utf-8")
    return path.read_text(encoding="utf-8")


def write_zstd_text(path: Path, text: str) -> None:
    """Atomically write actual zstd-compressed UTF-8 text."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    compressed = zstandard.ZstdCompressor(level=3).compress(text.encode("utf-8"))
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(compressed)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise
