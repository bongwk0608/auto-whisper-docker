from __future__ import annotations

import os
from pathlib import Path

from diarization.backend import DiarizationConfig, SpeakerSegment


class PyannoteDiarizationBackend:
    def __init__(self, config: DiarizationConfig, auth_token: str | None = None, device: str | None = None) -> None:
        self.config = config
        self.auth_token = auth_token if auth_token is not None else os.environ.get("PYANNOTE_AUTH_TOKEN", "")
        self.device = device
        if not self.auth_token:
            raise RuntimeError("PYANNOTE_AUTH_TOKEN is required for pyannote diarization.")

    def diarize(self, audio_path: Path) -> list[SpeakerSegment]:
        try:
            from pyannote.audio import Pipeline
            import torch
        except ImportError as exc:
            raise RuntimeError("pyannote.audio, torch, and torchaudio must be installed for diarization.") from exc

        pipeline = Pipeline.from_pretrained(self.config.model, token=self.auth_token)
        target_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(pipeline, "to"):
            pipeline.to(torch.device(target_device))

        kwargs: dict[str, int] = {}
        if self.config.num_speakers is not None:
            kwargs["num_speakers"] = self.config.num_speakers
        if self.config.min_speakers is not None:
            kwargs["min_speakers"] = self.config.min_speakers
        if self.config.max_speakers is not None:
            kwargs["max_speakers"] = self.config.max_speakers

        diarization = pipeline(str(audio_path), **kwargs)
        segments: list[SpeakerSegment] = []
        for turn, _track, speaker in diarization.itertracks(yield_label=True):
            segments.append(SpeakerSegment(float(turn.start), float(turn.end), str(speaker)))
        return sorted(segments, key=lambda item: (item.start, item.end, item.speaker_label))

