from __future__ import annotations

import os
import gc
from pathlib import Path
from typing import Any

from diarization.backend import DiarizationConfig, SpeakerSegment
from diarization.progress import DiarizationProgressReporter, ProjectProgressHook


class PyannoteModelAccessError(RuntimeError):
    pass


class DiarizationOutOfMemoryError(RuntimeError):
    pass


def parse_tf32_mode(value: str | None) -> str:
    mode = (value or "false").strip().lower()
    if mode not in {"auto", "true", "false"}:
        raise ValueError("DIARIZATION_TF32 must be one of: auto, true, false")
    return mode


def parse_oom_fallback(value: str | None) -> str:
    mode = (value or "cpu").strip().lower()
    if mode not in {"cpu", "skip", "fail"}:
        raise ValueError("DIARIZATION_OOM_FALLBACK must be one of: cpu, skip, fail")
    return mode


def is_cuda_oom_error(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message or "unable to find an engine" in message


def cleanup_cuda_memory(verbose: bool = False) -> None:
    gc.collect()
    try:
        import torch
    except ImportError:
        return
    try:
        if not torch.cuda.is_available():
            return
        torch.cuda.empty_cache()
        ipc_collect = getattr(torch.cuda, "ipc_collect", None)
        if callable(ipc_collect):
            ipc_collect()
        if verbose:
            print("CUDA cleanup completed", flush=True)
    except Exception as exc:
        if verbose:
            print(f"CUDA cleanup skipped after error: {exc}", flush=True)


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

    def diarize(self, audio_path: Path, progress_reporter: DiarizationProgressReporter | None = None) -> list[SpeakerSegment]:
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

        if progress_reporter is None:
            diarization = pipeline(str(audio_path), **kwargs)
        else:
            progress_reporter.update("pyannote inference", force=True)
            diarization = self.run_pipeline_with_progress(pipeline, audio_path, kwargs, progress_reporter)
        annotation = self.unwrap_diarization_output(diarization)
        segments: list[SpeakerSegment] = []
        for turn, _track, speaker in annotation.itertracks(yield_label=True):
            segments.append(SpeakerSegment(float(turn.start), float(turn.end), str(speaker)))
        return sorted(segments, key=lambda item: (item.start, item.end, item.speaker_label))

    def unwrap_diarization_output(self, diarization: Any) -> Any:
        if hasattr(diarization, "itertracks"):
            return diarization
        for attr in ("speaker_diarization", "diarization", "annotation"):
            value = getattr(diarization, attr, None)
            if value is not None and hasattr(value, "itertracks"):
                return value
        if isinstance(diarization, dict):
            for key in ("speaker_diarization", "diarization", "annotation"):
                value = diarization.get(key)
                if value is not None and hasattr(value, "itertracks"):
                    return value
        raise TypeError(f"Pyannote output does not contain an iterable diarization annotation: {type(diarization).__name__}")

    def run_pipeline_with_progress(
        self,
        pipeline: Any,
        audio_path: Path,
        kwargs: dict[str, int],
        progress_reporter: DiarizationProgressReporter,
    ) -> Any:
        try:
            from pyannote.audio.pipelines.utils.hook import ProgressHook
        except ImportError:
            return pipeline(str(audio_path), hook=ProjectProgressHook(progress_reporter), **kwargs)

        with ProgressHook():
            return pipeline(str(audio_path), hook=ProjectProgressHook(progress_reporter), **kwargs)

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
        pipeline, _torch = self.load_pipeline()
        del pipeline
        cleanup_cuda_memory(verbose=self.verbose)

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
