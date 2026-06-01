from __future__ import annotations

import shutil
import subprocess
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator


AUDIO_PREPROCESS_VERSION = "pcm16k-mono-v1"
RISKY_AUDIO_EXTENSIONS = {
    ".aac",
    ".m4a",
    ".m4v",
    ".mp4",
    ".mov",
    ".mkv",
    ".webm",
    ".wma",
    ".avi",
    ".wmv",
    ".flv",
    ".ts",
    ".mts",
    ".m2ts",
    ".3gp",
    ".3g2",
    ".mpg",
    ".mpeg",
    ".vob",
    ".ogv",
}


def parse_audio_preprocess_mode(value: str | None) -> str:
    mode = (value or "auto").strip().lower()
    if mode not in {"auto", "always", "false"}:
        raise ValueError("DIARIZATION_AUDIO_PREPROCESS must be one of: auto, always, false")
    return mode


def should_preprocess_audio(audio_path: Path, mode: str) -> bool:
    if mode == "false":
        return False
    if mode == "always":
        return True
    return audio_path.suffix.lower() in RISKY_AUDIO_EXTENSIONS


def audio_preprocess_cache_tag(mode: str, audio_path: Path) -> str:
    if should_preprocess_audio(audio_path, mode):
        return f"{AUDIO_PREPROCESS_VERSION}:{mode}"
    return f"direct:{mode}"


@contextmanager
def prepared_pyannote_audio(
    audio_path: Path,
    mode: str,
    staging_dir: Path,
    verbose: bool = False,
) -> Iterator[Path]:
    if not should_preprocess_audio(audio_path, mode):
        yield audio_path
        return

    staging_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="pyannote-audio-", dir=staging_dir))
    target = temp_dir / "audio.wav"
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(target),
    ]
    if verbose:
        print(f"Preprocessing audio for Pyannote: source={audio_path} target={target}", flush=True)
    try:
        completed = subprocess.run(command, capture_output=True, text=True, check=False)
        if completed.returncode != 0:
            error_text = (completed.stderr or completed.stdout or "ffmpeg failed").strip()
            raise RuntimeError(f"Pyannote audio preprocessing failed for {audio_path}: {error_text}")
        if verbose:
            print(f"Using preprocessed Pyannote audio: {target}", flush=True)
        yield target
    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)
        if verbose:
            print(f"Removed preprocessed Pyannote audio: {target}", flush=True)
