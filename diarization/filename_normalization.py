from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path


MAX_STEM_LENGTH = 160
SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyz0123456789._-")
WINDOWS_ILLEGAL_CHARS = set('<>:"/\\|?*')
WINDOWS_RESERVED_NAMES = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{index}" for index in range(1, 10)),
    *(f"lpt{index}" for index in range(1, 10)),
}
MAX_SAFE_COMPONENT_LENGTH = 180


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
    if safe.lower() in WINDOWS_RESERVED_NAMES:
        safe = f"{safe}_file"
    return f"{safe}{suffix}"


def parse_safe_output_policy(value: str | None) -> str:
    policy = (value or "auto").strip().lower()
    if policy not in {"auto", "true", "false"}:
        raise ValueError("SAFE_OUTPUT_FILENAMES must be one of: auto, true, false")
    return policy


def is_risky_filename(filename: str) -> bool:
    if filename in {"", ".", ".."}:
        return True
    if any(ord(ch) < 32 for ch in filename):
        return True
    if any(ch in WINDOWS_ILLEGAL_CHARS for ch in filename):
        return True
    if filename.rstrip(" .") != filename:
        return True
    if len(filename) > MAX_SAFE_COMPONENT_LENGTH:
        return True
    stem = Path(filename).stem.lower().rstrip(" .")
    return stem in WINDOWS_RESERVED_NAMES


def sanitize_kept_filename(filename: str) -> str:
    if filename in {"", ".", ".."}:
        return normalize_filename(filename or "file")
    cleaned = "".join("_" if ord(ch) < 32 or ch in {'/', '\\'} else ch for ch in filename)
    cleaned = cleaned.strip()
    return cleaned if cleaned and cleaned not in {".", ".."} else normalize_filename(filename)


def choose_output_filename(filename: str, policy: str) -> str:
    policy = parse_safe_output_policy(policy)
    if policy == "true":
        return normalize_filename(filename)
    if policy == "false":
        return sanitize_kept_filename(filename)
    if is_risky_filename(filename):
        return normalize_filename(filename)
    return filename


def unique_output_filename(filename: str, existing: set[str] | None = None, policy: str = "auto") -> NormalizedFilename:
    existing = existing if existing is not None else set()
    safe = choose_output_filename(filename, policy)
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


def safe_relative_path(path: Path, existing_by_parent: dict[Path, set[str]] | None = None, policy: str = "auto") -> Path:
    existing_by_parent = existing_by_parent if existing_by_parent is not None else {}
    policy = parse_safe_output_policy(policy)
    parts = [part for part in path.parts if part not in {"", ".", ".."}]
    safe_parts: list[str] = [choose_output_filename(part, policy) for part in parts[:-1]]
    if parts:
        parent = Path(*safe_parts) if safe_parts else Path(".")
        existing = existing_by_parent.setdefault(parent, set())
        safe_parts.append(unique_output_filename(parts[-1], existing, policy).safe_filename)
    return Path(*safe_parts) if safe_parts else Path(choose_output_filename(path.name or "file", policy))
