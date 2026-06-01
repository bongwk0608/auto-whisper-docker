from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diarization.backend import DiarizationConfig, SpeakerSegment
from diarization.audio_preprocess import (
    audio_preprocess_cache_tag,
    parse_audio_preprocess_mode,
    prepared_pyannote_audio,
)
from diarization.export_speaker_transcript import export_speaker_outputs, speaker_outputs_complete
from diarization.filename_normalization import parse_safe_output_policy
from diarization.merge_whisper_speakers import MergeConfig, assign_speakers_to_whisper_segments
from diarization.progress import DiarizationProgressReporter, ProgressContext, parse_progress_enabled
from diarization.pyannote_runner import (
    DiarizationOutOfMemoryError,
    DiarizationRuntimeState,
    PyannoteDiarizationBackend,
    cleanup_runtime_memory,
    is_cuda_oom_error,
    parse_cuda_quarantine_after_oom,
    parse_gpu_memory_log,
    parse_gpu_memory_wait_seconds,
    parse_oom_fallback,
    parse_tf32_mode,
    parse_worker_timeout_seconds,
    parse_worker_mode,
    prepare_next_file_cuda_attempt,
)
from diarization.raw_cache import cache_key, load_cached_segments, save_cached_segments


def parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def parse_bool(value: str | None, default: bool = False) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def load_whisper_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or not isinstance(data.get("segments"), list):
        raise ValueError(f"Whisper JSON does not contain a segments list: {path}")
    return data


def speaker_segments_from_dicts(items: list[dict[str, Any]]) -> list[SpeakerSegment]:
    return [
        SpeakerSegment(float(item["start"]), float(item["end"]), str(item["speaker_label"]))
        for item in items
    ]


def read_gpu_memory() -> str | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.used,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
    except OSError as exc:
        return f"unavailable ({exc})"
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout or "nvidia-smi failed").strip()
        return f"unavailable ({message})"
    return f"{completed.stdout.strip()} MiB used/total"


def log_gpu_memory(label: str, enabled: bool) -> None:
    if not enabled:
        return
    memory = read_gpu_memory()
    print(f"GPU memory {label}: {memory}", flush=True)


def wait_for_gpu_memory_settle(wait_seconds: int, enabled: bool, label: str) -> None:
    if not enabled or wait_seconds <= 0:
        return
    started_at = time.perf_counter()
    deadline = started_at + wait_seconds
    print(
        f"GPU memory wait {label}: waiting up to {wait_seconds}s for driver/WSL memory accounting to settle",
        flush=True,
    )
    while time.perf_counter() < deadline:
        time.sleep(min(2.0, max(0.0, deadline - time.perf_counter())))
    memory = read_gpu_memory()
    elapsed = time.perf_counter() - started_at
    print(
        f"GPU memory wait {label}: completed after {elapsed:.1f}s; current={memory}. "
        "Small NVIDIA driver/context reservations may remain visible in WSL/Docker.",
        flush=True,
    )


def run_pyannote_worker(
    audio_path: Path,
    config: DiarizationConfig,
    device: str,
    tf32_mode: str | None,
    verbose: bool,
    gpu_memory_log: bool,
    worker_timeout_seconds: int | None = None,
    gpu_memory_wait_seconds: int | None = None,
) -> list[SpeakerSegment]:
    timeout_seconds = (
        parse_worker_timeout_seconds(os.environ.get("DIARIZATION_WORKER_TIMEOUT_SECONDS"))
        if worker_timeout_seconds is None
        else worker_timeout_seconds
    )
    memory_wait_seconds = (
        parse_gpu_memory_wait_seconds(os.environ.get("DIARIZATION_GPU_MEMORY_WAIT_SECONDS"))
        if gpu_memory_wait_seconds is None
        else gpu_memory_wait_seconds
    )
    with tempfile.TemporaryDirectory(prefix="pyannote-worker-") as temp_dir:
        output_path = Path(temp_dir) / "segments.json"
        command = [
            sys.executable,
            str(ROOT / "scripts" / "pyannote_worker.py"),
            "--audio",
            str(audio_path),
            "--cache-out",
            str(output_path),
            "--device",
            device,
            "--backend",
            config.backend,
            "--model",
            config.model,
            "--tf32",
            parse_tf32_mode(tf32_mode),
        ]
        if config.num_speakers is not None:
            command.extend(["--num-speakers", str(config.num_speakers)])
        if config.min_speakers is not None:
            command.extend(["--min-speakers", str(config.min_speakers)])
        if config.max_speakers is not None:
            command.extend(["--max-speakers", str(config.max_speakers)])
        if verbose:
            command.append("--verbose")
            print(f"Starting Pyannote worker: device={device} timeout={timeout_seconds}s", flush=True)
        log_gpu_memory(f"before worker {device}", gpu_memory_log)
        try:
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                check=False,
                timeout=timeout_seconds if timeout_seconds > 0 else None,
            )
        except subprocess.TimeoutExpired as exc:
            if exc.stdout:
                stdout = exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout
                print(stdout, end="" if stdout.endswith("\n") else "\n", flush=True)
            if exc.stderr:
                stderr = exc.stderr.decode() if isinstance(exc.stderr, bytes) else exc.stderr
                print(stderr, end="" if stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
            log_gpu_memory(f"after timed-out worker {device}", gpu_memory_log)
            wait_for_gpu_memory_settle(memory_wait_seconds, gpu_memory_log, f"after timed-out worker {device}")
            raise RuntimeError(f"Pyannote worker timed out after {timeout_seconds}s: device={device}") from None
        if completed.stdout:
            print(completed.stdout, end="" if completed.stdout.endswith("\n") else "\n", flush=True)
        if completed.stderr:
            print(completed.stderr, end="" if completed.stderr.endswith("\n") else "\n", file=sys.stderr, flush=True)
        log_gpu_memory(f"after worker {device}", gpu_memory_log)
        wait_for_gpu_memory_settle(memory_wait_seconds, gpu_memory_log, f"after worker {device}")
        payload: dict[str, Any] = {}
        if output_path.exists():
            payload = json.loads(output_path.read_text(encoding="utf-8"))
        if completed.returncode != 0 or payload.get("status") != "completed":
            message = str(payload.get("error_message") or completed.stderr or completed.stdout or "Pyannote worker failed")
            if payload.get("is_cuda_oom"):
                raise RuntimeError(f"CUDA error: out of memory ({message})")
            raise RuntimeError(message)
        if verbose:
            print("Pyannote worker exited: status=completed", flush=True)
        segments = payload.get("speaker_segments", [])
        if not isinstance(segments, list):
            raise RuntimeError("Pyannote worker returned invalid speaker_segments")
        return speaker_segments_from_dicts(segments)


def export_diarization_from_segments(
    audio_path: Path,
    whisper_json: Path,
    output_base: Path,
    config: DiarizationConfig,
    min_overlap_ratio: float,
    speaker_segments: list[SpeakerSegment],
    verbose: bool = False,
    progress: bool = False,
    progress_context: ProgressContext | None = None,
    filename_policy: str = "auto",
) -> dict[str, Path]:
    whisper_data = load_whisper_json(whisper_json)
    if verbose:
        print(f"Loaded Whisper JSON: segments={len(whisper_data['segments'])} path={whisper_json}", flush=True)
        print(f"Merging speakers into Whisper segments: {len(whisper_data['segments'])} segments min_overlap_ratio={min_overlap_ratio}", flush=True)
    if progress:
        DiarizationProgressReporter(progress_context).message(
            f"merging speakers into {len(whisper_data['segments'])} Whisper segments"
        )
    merged_segments = assign_speakers_to_whisper_segments(
        whisper_data["segments"],
        speaker_segments,
        MergeConfig(min_overlap_ratio=min_overlap_ratio),
    )
    return export_speaker_outputs(
        output_base,
        audio_path,
        whisper_json,
        whisper_data,
        speaker_segments,
        merged_segments,
        config,
        filename_policy=filename_policy,
    )


def run_single_diarization(
    audio_path: Path,
    whisper_json: Path,
    output_base: Path,
    config: DiarizationConfig,
    min_overlap_ratio: float,
    cache_dir: Path,
    force: bool = False,
    dry_run: bool = False,
    verbose: bool = False,
    tf32_mode: str | None = None,
    progress: bool = False,
    progress_context: ProgressContext | None = None,
    oom_fallback: str = "cpu",
    filename_policy: str = "auto",
    audio_preprocess: str = "always",
    audio_preprocess_dir: Path = Path("/tmp/auto-whisper-diarization"),
    runtime_state: DiarizationRuntimeState | None = None,
    cuda_quarantine_after_oom: bool = True,
    worker_mode: str = "always",
    gpu_memory_log: bool = False,
    worker_timeout_seconds: int = 7200,
    gpu_memory_wait_seconds: int = 0,
) -> tuple[dict[str, Path], bool]:
    started_at = time.perf_counter()
    if speaker_outputs_complete(output_base) and not force:
        print(f"Skipping existing diarization outputs: {output_base}", flush=True)
        return {}, False
    if dry_run:
        print(f"Would diarize {audio_path} with {whisper_json} -> {output_base}", flush=True)
        return {}, False

    speaker_segments, cache_hit = diarize_with_cache(
        audio_path,
        config,
        cache_dir,
        force=force,
        verbose=verbose,
        tf32_mode=tf32_mode,
        progress=progress,
        progress_context=progress_context,
        oom_fallback=oom_fallback,
        audio_preprocess=audio_preprocess,
        audio_preprocess_dir=audio_preprocess_dir,
        runtime_state=runtime_state,
        cuda_quarantine_after_oom=cuda_quarantine_after_oom,
        worker_mode=worker_mode,
        gpu_memory_log=gpu_memory_log,
        worker_timeout_seconds=worker_timeout_seconds,
        gpu_memory_wait_seconds=gpu_memory_wait_seconds,
    )

    paths = export_diarization_from_segments(
        audio_path,
        whisper_json,
        output_base,
        config,
        min_overlap_ratio,
        speaker_segments,
        verbose=verbose,
        progress=progress,
        progress_context=progress_context,
        filename_policy=filename_policy,
    )
    if verbose:
        elapsed = time.perf_counter() - started_at
        print(f"Exported diarization outputs: files={len(paths)} output_base={output_base} elapsed={elapsed:.1f}s", flush=True)
    return paths, cache_hit


def diarize_with_cache(
    audio_path: Path,
    config: DiarizationConfig,
    cache_dir: Path,
    force: bool = False,
    verbose: bool = False,
    tf32_mode: str | None = None,
    progress: bool = False,
    progress_context: ProgressContext | None = None,
    oom_fallback: str = "cpu",
    audio_preprocess: str = "always",
    audio_preprocess_dir: Path = Path("/tmp/auto-whisper-diarization"),
    runtime_state: DiarizationRuntimeState | None = None,
    cuda_quarantine_after_oom: bool = True,
    worker_mode: str = "always",
    gpu_memory_log: bool = False,
    worker_timeout_seconds: int = 7200,
    gpu_memory_wait_seconds: int = 0,
) -> tuple[list[SpeakerSegment], bool]:
    if runtime_state is None:
        runtime_state = DiarizationRuntimeState()
    preprocess_tag = audio_preprocess_cache_tag(audio_preprocess, audio_path)
    key = cache_key(audio_path, config, audio_preprocess=preprocess_tag)
    cached = None if force else load_cached_segments(cache_dir, key)
    if cached is None and not force:
        legacy_key = cache_key(audio_path, config)
        cached = load_cached_segments(cache_dir, legacy_key)
    if cached is not None:
        print(f"Raw diarization cache hit: {audio_path}", flush=True)
        return cached, True

    prepare_next_file_cuda_attempt(runtime_state, cuda_quarantine_after_oom, verbose=verbose)
    print(f"Raw diarization cache miss: {audio_path}", flush=True)
    if verbose:
        print(f"Starting Pyannote inference: model={config.model} audio={audio_path}", flush=True)
    inference_started_at = time.perf_counter()
    progress_reporter = DiarizationProgressReporter(progress_context) if progress else None
    cleanup_runtime_memory(verbose=verbose, label="Runtime cleanup before file")
    with prepared_pyannote_audio(audio_path, audio_preprocess, audio_preprocess_dir, verbose=verbose) as pyannote_audio_path:
        try:
            device = None if runtime_state.cuda_healthy else "cpu"
            if verbose and device == "cpu" and runtime_state.cuda_oom_quarantined:
                print("CUDA is quarantined; running Pyannote on CPU", flush=True)
            use_worker = worker_mode == "always" or (worker_mode == "on_oom" and runtime_state.use_worker_after_oom)
            if use_worker:
                worker_device = device or "cuda"
                speaker_segments = run_pyannote_worker(
                    pyannote_audio_path,
                    config,
                    worker_device,
                    tf32_mode,
                    verbose,
                    gpu_memory_log,
                    worker_timeout_seconds,
                    gpu_memory_wait_seconds,
                )
                cleanup_runtime_memory(verbose=verbose, label="Runtime cleanup after worker")
            else:
                backend = PyannoteDiarizationBackend(config, device=device, tf32_mode=tf32_mode, verbose=verbose)
                try:
                    speaker_segments = backend.diarize(pyannote_audio_path, progress_reporter=progress_reporter)
                finally:
                    del backend
                    cleanup_runtime_memory(verbose=verbose, label="Runtime cleanup after file")
        except Exception as exc:
            cleanup_runtime_memory(verbose=verbose, label="Runtime cleanup after failure")
            if not is_cuda_oom_error(exc):
                raise
            if oom_fallback != "cpu":
                raise DiarizationOutOfMemoryError(str(exc)) from exc
            if worker_mode == "on_oom":
                runtime_state.use_worker_after_oom = True
            if cuda_quarantine_after_oom and runtime_state.cuda_healthy:
                runtime_state.cuda_healthy = False
                runtime_state.cuda_oom_quarantined = True
                print("CUDA marked unhealthy after OOM; remaining uncached files will use CPU", flush=True)
            print(f"CUDA OOM detected; retrying same audio on CPU: {audio_path}", flush=True)
            if worker_mode == "always":
                speaker_segments = run_pyannote_worker(
                    pyannote_audio_path,
                    config,
                    "cpu",
                    tf32_mode,
                    verbose,
                    gpu_memory_log,
                    worker_timeout_seconds,
                    gpu_memory_wait_seconds,
                )
                cleanup_runtime_memory(verbose=verbose, label="Runtime cleanup after CPU worker fallback")
            else:
                cpu_backend = PyannoteDiarizationBackend(config, device="cpu", tf32_mode=tf32_mode, verbose=verbose)
                try:
                    speaker_segments = cpu_backend.diarize(pyannote_audio_path, progress_reporter=progress_reporter)
                finally:
                    del cpu_backend
                    cleanup_runtime_memory(verbose=verbose, label="Runtime cleanup after CPU fallback")
    if verbose:
        elapsed = time.perf_counter() - inference_started_at
        print(f"Finished Pyannote inference: speakers={len({segment.speaker_label for segment in speaker_segments})} segments={len(speaker_segments)} elapsed={elapsed:.1f}s", flush=True)
    save_cached_segments(cache_dir, key, audio_path, config, speaker_segments)
    if verbose:
        print(f"Saved raw diarization cache: {cache_dir / (key + '.json')}", flush=True)
    return speaker_segments, False


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run speaker diarization for one Whisper JSON transcript.")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--whisper-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--backend", default=os.environ.get("DIARIZATION_BACKEND", "pyannote"))
    parser.add_argument("--model", default=os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"))
    parser.add_argument("--min-overlap-ratio", type=float, default=float(os.environ.get("DIARIZATION_MIN_OVERLAP_RATIO", "0.3")))
    parser.add_argument("--num-speakers", type=int, default=parse_optional_int(os.environ.get("DIARIZATION_NUM_SPEAKERS")))
    parser.add_argument("--min-speakers", type=int, default=parse_optional_int(os.environ.get("DIARIZATION_MIN_SPEAKERS")))
    parser.add_argument("--max-speakers", type=int, default=parse_optional_int(os.environ.get("DIARIZATION_MAX_SPEAKERS")))
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("DIARIZATION_CACHE_DIR", "state/diarization-cache")))
    parser.add_argument("--safe-output-filenames", choices=["auto", "true", "false"], default=parse_safe_output_policy(os.environ.get("SAFE_OUTPUT_FILENAMES")))
    parser.add_argument("--audio-preprocess", choices=["auto", "always", "false"], default=parse_audio_preprocess_mode(os.environ.get("DIARIZATION_AUDIO_PREPROCESS")))
    parser.add_argument("--audio-preprocess-dir", type=Path, default=Path(os.environ.get("DIARIZATION_AUDIO_PREPROCESS_DIR", "/tmp/auto-whisper-diarization")))
    parser.add_argument("--tf32", choices=["auto", "true", "false"], default=parse_tf32_mode(os.environ.get("DIARIZATION_TF32")))
    parser.add_argument("--oom-fallback", choices=["cpu", "skip", "fail"], default=parse_oom_fallback(os.environ.get("DIARIZATION_OOM_FALLBACK")))
    parser.add_argument("--cuda-quarantine-after-oom", action="store_true", default=parse_cuda_quarantine_after_oom(os.environ.get("DIARIZATION_CUDA_QUARANTINE_AFTER_OOM")))
    parser.add_argument("--no-cuda-quarantine-after-oom", dest="cuda_quarantine_after_oom", action="store_false")
    parser.add_argument("--worker-mode", choices=["false", "on_oom", "always"], default=parse_worker_mode(os.environ.get("DIARIZATION_WORKER_MODE")))
    parser.add_argument("--gpu-memory-log", action="store_true", default=parse_gpu_memory_log(os.environ.get("DIARIZATION_GPU_MEMORY_LOG")))
    parser.add_argument("--no-gpu-memory-log", dest="gpu_memory_log", action="store_false")
    parser.add_argument("--worker-timeout-seconds", type=parse_worker_timeout_seconds, default=parse_worker_timeout_seconds(os.environ.get("DIARIZATION_WORKER_TIMEOUT_SECONDS")))
    parser.add_argument("--gpu-memory-wait-seconds", type=parse_gpu_memory_wait_seconds, default=parse_gpu_memory_wait_seconds(os.environ.get("DIARIZATION_GPU_MEMORY_WAIT_SECONDS")))
    parser.add_argument("--verbose", action="store_true", default=parse_bool(os.environ.get("DIARIZATION_VERBOSE"), False))
    progress_default = parse_progress_enabled(os.environ.get("DIARIZATION_PROGRESS"), parse_bool(os.environ.get("DIARIZATION_VERBOSE"), False))
    parser.add_argument("--progress", dest="progress", action="store_true", default=progress_default)
    parser.add_argument("--no-progress", dest="progress", action="store_false")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.backend != "pyannote":
        raise ValueError("Only the pyannote diarization backend is implemented.")
    if not args.audio.exists():
        raise FileNotFoundError(f"Audio file does not exist: {args.audio}")
    if not args.whisper_json.exists():
        raise FileNotFoundError(f"Whisper JSON does not exist: {args.whisper_json}")

    config = DiarizationConfig(args.backend, args.model, args.num_speakers, args.min_speakers, args.max_speakers)
    output_base = args.output_dir / args.whisper_json.name
    output_base = output_base.with_suffix("")
    paths, _cache_hit = run_single_diarization(
        args.audio,
        args.whisper_json,
        output_base,
        config,
        args.min_overlap_ratio,
        args.cache_dir,
        args.force,
        args.dry_run,
        args.verbose,
        args.tf32,
        args.progress and not args.dry_run,
        ProgressContext(file_index=1, file_total=1),
        args.oom_fallback,
        args.safe_output_filenames,
        args.audio_preprocess,
        args.audio_preprocess_dir,
        DiarizationRuntimeState(),
        args.cuda_quarantine_after_oom,
        args.worker_mode,
        args.gpu_memory_log,
        args.worker_timeout_seconds,
        args.gpu_memory_wait_seconds,
    )
    for name, path in paths.items():
        print(f"{name}: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
