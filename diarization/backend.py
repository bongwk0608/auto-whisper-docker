from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol


@dataclass(frozen=True)
class DiarizationConfig:
    backend: str = "pyannote"
    model: str = "pyannote/speaker-diarization-community-1"
    num_speakers: int | None = None
    min_speakers: int | None = None
    max_speakers: int | None = None


@dataclass(frozen=True)
class SpeakerSegment:
    start: float
    end: float
    speaker_label: str

    def to_dict(self) -> dict[str, object]:
        return {
            "start": self.start,
            "end": self.end,
            "speaker_label": self.speaker_label,
        }


class DiarizationBackend(Protocol):
    config: DiarizationConfig

    def diarize(self, audio_path: Path) -> list[SpeakerSegment]:
        """Return speaker timeline segments for an audio file."""

