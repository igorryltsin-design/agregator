"""Utility helpers for scanning and fingerprints."""

from __future__ import annotations

import hashlib
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, List


def iter_files(root: Path, max_depth: int, follow_symlinks: bool, patterns: List[str]) -> Iterator[Path]:
    patterns = [pattern.lower().strip() for pattern in patterns if pattern]
    stack = [(root, 0)]
    visited = set()

    while stack:
        current, depth = stack.pop()
        if not current.exists():
            continue
        key = os.path.realpath(current)
        if key in visited:
            continue
        visited.add(key)

        for entry in current.iterdir():
            if entry.is_dir():
                if depth < max_depth:
                    if not entry.is_symlink() or follow_symlinks:
                        stack.append((entry, depth + 1))
            else:
                if any(entry.name.lower().endswith(pattern.replace("*", "")) for pattern in patterns):
                    yield entry


def entropy(data: bytes) -> float:
    if not data:
        return 0.0
    freq = {}
    for byte in data:
        freq[byte] = freq.get(byte, 0) + 1
    total = len(data)
    ent = 0.0
    for count in freq.values():
        p = count / total
        ent -= p * math.log2(p)
    return ent / 8.0


def rolling_hash(file_path: Path, block_size: int, hash_alg: str) -> str:
    h = hashlib.new(hash_alg)
    with file_path.open("rb") as fh:
        while chunk := fh.read(block_size):
            h.update(chunk)
    return h.hexdigest()


def chunk_signatures(file_path: Path, block_size: int, hash_alg: str) -> List[tuple[int, str, int]]:
    results = []
    with file_path.open("rb") as fh:
        offset = 0
        while chunk := fh.read(block_size):
            h = hashlib.new(hash_alg)
            h.update(chunk)
            results.append((offset, h.hexdigest(), len(chunk)))
            offset += len(chunk)
    return results


def guess_modality(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix in {".txt", ".pdf", ".doc", ".docx"}:
        return "text"
    if suffix in {".mp3", ".wav", ".flac"}:
        return "audio"
    if suffix in {".png", ".jpg", ".jpeg", ".gif"}:
        return "image"
    if suffix in {".mp4", ".mov", ".avi"}:
        return "video"
    if suffix in {".zip", ".tar", ".gz"}:
        return "archive"
    return "unknown"
