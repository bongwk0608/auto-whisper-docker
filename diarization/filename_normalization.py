from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


MAX_STEM_LENGTH = 160
SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789._-")


@dataclass(frozen=True)
class NormalizedFilename:
    original_filename: str
    safe_filename: str
    collision_index: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "original_filename": self.original_filename,
            "safe_filename": self.safe_filename,
            "collision_index": self.collision_index,
        }


def normalize_filename(filename: str) -> str:
    path = Path(filename)
    suffix = path.suffix.lower()
    stem = path.name[: -len(path.suffix)] if path.suffix else path.name
    normalized = unicodedata.normalize("NFKD", stem).encode("ascii", "ignore").decode("ascii")
    normalized = normalized.lower()
    safe = "".join(ch if ch in SAFE_CHARS else "_" for ch in normalized)
    safe = re.sub(r"_+", "_", safe).strip("._")
    if not safe:
        safe = "file"
    if len(safe) > MAX_STEM_LENGTH:
        safe = safe[:MAX_STEM_LENGTH].rstrip("._") or "file"
    return f"{safe}{suffix}"


def unique_normalized_filename(filename: str, existing: set[str] | None = None) -> NormalizedFilename:
    existing = existing if existing is not None else set()
    safe = normalize_filename(filename)
    if safe not in existing:
        existing.add(safe)
        return NormalizedFilename(filename, safe)

    path = Path(safe)
    suffix = path.suffix
    stem = path.name[: -len(suffix)] if suffix else path.name
    index = 1
    while True:
        candidate = f"{stem}_{index}{suffix}"
        if candidate not in existing:
            existing.add(candidate)
            return NormalizedFilename(filename, candidate, index)
        index += 1

