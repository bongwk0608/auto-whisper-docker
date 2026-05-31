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
from diarization.merge_whisper_speakers import MergeConfig, assign_speakers_to_whisper_segments
from diarization.pyannote_runner import PyannoteDiarizationBackend, parse_tf32_mode
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
) -> tuple[dict[str, Path], bool]:
    started_at = time.perf_counter()
    if speaker_outputs_complete(output_base) and not force:
        print(f"Skipping existing diarization outputs: {output_base}", flush=True)
        return {}, False
    if dry_run:
        print(f"Would diarize {audio_path} with {whisper_json} -> {output_base}", flush=True)
        return {}, False

    whisper_data = load_whisper_json(whisper_json)
    if verbose:
        print(f"Loaded Whisper JSON: segments={len(whisper_data['segments'])} path={whisper_json}", flush=True)
    key = cache_key(audio_path, config)
    cached = None if force else load_cached_segments(cache_dir, key)
    cache_hit = cached is not None
    if cache_hit:
        print(f"Raw diarization cache hit: {audio_path}", flush=True)
        speaker_segments = cached
    else:
        print(f"Raw diarization cache miss: {audio_path}", flush=True)
        backend = PyannoteDiarizationBackend(config, tf32_mode=tf32_mode, verbose=verbose)
        if verbose:
            print(f"Starting Pyannote inference: model={config.model} audio={audio_path}", flush=True)
        inference_started_at = time.perf_counter()
        speaker_segments = backend.diarize(audio_path)
        if verbose:
            elapsed = time.perf_counter() - inference_started_at
            print(f"Finished Pyannote inference: speakers={len({segment.speaker_label for segment in speaker_segments})} segments={len(speaker_segments)} elapsed={elapsed:.1f}s", flush=True)
        save_cached_segments(cache_dir, key, audio_path, config, speaker_segments)
        if verbose:
            print(f"Saved raw diarization cache: {cache_dir / (key + '.json')}", flush=True)

    if verbose:
        print(f"Merging speakers into Whisper segments: min_overlap_ratio={min_overlap_ratio}", flush=True)
    merged_segments = assign_speakers_to_whisper_segments(
        whisper_data["segments"],
        speaker_segments,
        MergeConfig(min_overlap_ratio=min_overlap_ratio),
    )
    paths = export_speaker_outputs(output_base, audio_path, whisper_json, whisper_data, speaker_segments, merged_segments, config)
    if verbose:
        elapsed = time.perf_counter() - started_at
        print(f"Exported diarization outputs: files={len(paths)} output_base={output_base} elapsed={elapsed:.1f}s", flush=True)
    return paths, cache_hit


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
    parser.add_argument("--tf32", choices=["auto", "true", "false"], default=parse_tf32_mode(os.environ.get("DIARIZATION_TF32")))
    parser.add_argument("--verbose", action="store_true", default=parse_bool(os.environ.get("DIARIZATION_VERBOSE"), False))
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
    )
    for name, path in paths.items():
        print(f"{name}: {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
