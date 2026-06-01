from __future__ import annotations

import hashlib
import json
import tempfile
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from diarization.backend import DiarizationConfig, SpeakerSegment


def cache_key(audio_path: Path, config: DiarizationConfig, audio_preprocess: str | None = None) -> str:
    stat = audio_path.stat()
    payload = {
        "audio_absolute_path": str(audio_path.resolve()),
        "audio_size": stat.st_size,
        "audio_mtime_ns": stat.st_mtime_ns,
        "backend": config.backend,
        "model": config.model,
        "num_speakers": config.num_speakers,
        "min_speakers": config.min_speakers,
        "max_speakers": config.max_speakers,
    }
    if audio_preprocess is not None:
        payload["audio_preprocess"] = audio_preprocess
    encoded = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / f"{key}.json"


def load_cached_segments(cache_dir: Path, key: str) -> list[SpeakerSegment] | None:
    path = cache_path(cache_dir, key)
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return [
        SpeakerSegment(float(item["start"]), float(item["end"]), str(item["speaker_label"]))
        for item in data.get("speaker_segments", [])
    ]


def save_cached_segments(cache_dir: Path, key: str, audio_path: Path, config: DiarizationConfig, segments: list[SpeakerSegment]) -> Path:
    cache_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "cache_key": key,
        "source_audio": str(audio_path),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "config": asdict(config),
        "speaker_segments": [segment.to_dict() for segment in segments],
    }
    path = cache_path(cache_dir, key)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=cache_dir, delete=False) as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    Path(temp_name).replace(path)
    return path
