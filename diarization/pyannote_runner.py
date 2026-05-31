from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from diarization.backend import DiarizationConfig, SpeakerSegment


class PyannoteModelAccessError(RuntimeError):
    pass


def parse_tf32_mode(value: str | None) -> str:
    mode = (value or "false").strip().lower()
    if mode not in {"auto", "true", "false"}:
        raise ValueError("DIARIZATION_TF32 must be one of: auto, true, false")
    return mode


class PyannoteDiarizationBackend:
    def __init__(
        self,
        config: DiarizationConfig,
        auth_token: str | None = None,
        device: str | None = None,
        tf32_mode: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.config = config
        self.auth_token = auth_token if auth_token is not None else os.environ.get("PYANNOTE_AUTH_TOKEN", "")
        self.device = device
        self.tf32_mode = parse_tf32_mode(tf32_mode if tf32_mode is not None else os.environ.get("DIARIZATION_TF32"))
        self.verbose = verbose
        if not self.auth_token:
            raise RuntimeError("PYANNOTE_AUTH_TOKEN is required for pyannote diarization.")

    def diarize(self, audio_path: Path) -> list[SpeakerSegment]:
        pipeline, torch = self.load_pipeline()
        target_device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        if hasattr(pipeline, "to"):
            pipeline.to(torch.device(target_device))
        self.configure_tf32(torch, "before inference")

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

    def load_pipeline(self) -> tuple[Any, Any]:
        try:
            from pyannote.audio import Pipeline
            import torch
        except ImportError as exc:
            raise RuntimeError("pyannote.audio, torch, and torchaudio must be installed for diarization.") from exc

        self.log_torch_startup(torch)
        self.configure_tf32(torch, "before pipeline load")

        try:
            return Pipeline.from_pretrained(self.config.model, token=self.auth_token), torch
        except Exception as exc:
            message = str(exc)
            if "Cannot access gated repo" in message or "403" in message or "gated" in message.lower():
                raise PyannoteModelAccessError(
                    "Cannot access the configured Pyannote model. "
                    f"Open https://huggingface.co/{self.config.model}, accept the model terms with the "
                    "same Hugging Face account that owns PYANNOTE_AUTH_TOKEN, then create or use a token "
                    "with read access and update PYANNOTE_AUTH_TOKEN in .env."
                ) from exc
            raise

    def validate_access(self) -> None:
        self.load_pipeline()

    def configure_tf32(self, torch: Any, stage: str) -> None:
        if not torch.cuda.is_available():
            if self.verbose:
                print(f"Pyannote TF32 {stage}: CUDA unavailable, mode={self.tf32_mode}", flush=True)
            return

        before_matmul = torch.backends.cuda.matmul.allow_tf32
        before_cudnn = torch.backends.cudnn.allow_tf32
        if self.tf32_mode == "true":
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
        elif self.tf32_mode == "false":
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False

        after_matmul = torch.backends.cuda.matmul.allow_tf32
        after_cudnn = torch.backends.cudnn.allow_tf32
        if self.verbose:
            print(
                f"Pyannote TF32 {stage}: mode={self.tf32_mode} "
                f"matmul={before_matmul}->{after_matmul} cudnn={before_cudnn}->{after_cudnn}",
                flush=True,
            )

    def log_torch_startup(self, torch: Any) -> None:
        if not self.verbose:
            return
        cuda_available = torch.cuda.is_available()
        gpu_name = torch.cuda.get_device_name(0) if cuda_available else "none"
        print(
            f"Pyannote startup: model={self.config.model} cuda_available={cuda_available} "
            f"gpu={gpu_name} tf32_mode={self.tf32_mode} verbose={self.verbose}",
            flush=True,
        )
