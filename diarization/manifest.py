from __future__ import annotations

import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any


DEFAULT_MANIFEST = {"version": 1, "jobs": {}}


def load_manifest(path: Path) -> dict[str, Any]:
    if not path.exists():
        return dict(DEFAULT_MANIFEST)
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    data.setdefault("version", 1)
    data.setdefault("jobs", {})
    return data


def save_manifest(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    Path(temp_name).replace(path)


def manifest_key(whisper_json: Path, output_base: Path) -> str:
    return f"{whisper_json.resolve()}::{output_base.resolve()}"


def update_job(
    manifest: dict[str, Any],
    key: str,
    status: str,
    whisper_json: Path,
    output_base: Path,
    audio_path: Path | None = None,
    output_paths: dict[str, Path] | None = None,
    error_message: str | None = None,
    cache_hit: bool | None = None,
) -> None:
    manifest.setdefault("jobs", {})[key] = {
        "status": status,
        "whisper_json": str(whisper_json),
        "audio_path": str(audio_path) if audio_path else "",
        "output_base": str(output_base),
        "output_paths": {name: str(path) for name, path in (output_paths or {}).items()},
        "error_message": error_message,
        "cache_hit": cache_hit,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }

