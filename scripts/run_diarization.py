from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diarization.backend import DiarizationConfig, SpeakerSegment
from diarization.export_speaker_transcript import export_speaker_outputs, speaker_outputs_complete
from diarization.filename_normalization import parse_safe_output_policy
from diarization.merge_whisper_speakers import MergeConfig, assign_speakers_to_whisper_segments
from diarization.progress import DiarizationProgressReporter, ProgressContext, parse_progress_enabled
from diarization.pyannote_runner import (
    DiarizationOutOfMemoryError,
    PyannoteDiarizationBackend,
    cleanup_runtime_memory,
    is_cuda_oom_error,
    parse_oom_fallback,
    parse_tf32_mode,
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
) -> tuple[list[SpeakerSegment], bool]:
    key = cache_key(audio_path, config)
    cached = None if force else load_cached_segments(cache_dir, key)
    if cached is not None:
        print(f"Raw diarization cache hit: {audio_path}", flush=True)
        return cached, True

    print(f"Raw diarization cache miss: {audio_path}", flush=True)
    if verbose:
        print(f"Starting Pyannote inference: model={config.model} audio={audio_path}", flush=True)
    inference_started_at = time.perf_counter()
    progress_reporter = DiarizationProgressReporter(progress_context) if progress else None
    cleanup_runtime_memory(verbose=verbose, label="Runtime cleanup before file")
    try:
        backend = PyannoteDiarizationBackend(config, tf32_mode=tf32_mode, verbose=verbose)
        try:
            speaker_segments = backend.diarize(audio_path, progress_reporter=progress_reporter)
        finally:
            del backend
            cleanup_runtime_memory(verbose=verbose, label="Runtime cleanup after file")
    except Exception as exc:
        cleanup_runtime_memory(verbose=verbose, label="Runtime cleanup after failure")
        if not is_cuda_oom_error(exc):
            raise
        if oom_fallback != "cpu":
            raise DiarizationOutOfMemoryError(str(exc)) from exc
        print(f"CUDA OOM detected; retrying same audio on CPU: {audio_path}", flush=True)
        cpu_backend = PyannoteDiarizationBackend(config, device="cpu", tf32_mode=tf32_mode, verbose=verbose)
        try:
            speaker_segments = cpu_backend.diarize(audio_path, progress_reporter=progress_reporter)
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
    parser.add_argument("--tf32", choices=["auto", "true", "false"], default=parse_tf32_mode(os.environ.get("DIARIZATION_TF32")))
    parser.add_argument("--oom-fallback", choices=["cpu", "skip", "fail"], default=parse_oom_fallback(os.environ.get("DIARIZATION_OOM_FALLBACK")))
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
    )
    for name, path in paths.items():
        print(f"{name}: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
