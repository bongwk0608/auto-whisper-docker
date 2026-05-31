from __future__ import annotations

import argparse
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diarization.backend import DiarizationConfig
from diarization.export_speaker_transcript import output_base_for_whisper_json, speaker_outputs_complete
from diarization.manifest import load_manifest, manifest_key, save_manifest, update_job
from scripts.run_diarization import parse_optional_int, run_single_diarization


SUPPORTED_AUDIO_EXTENSIONS = [
    ".mp3",
    ".wav",
    ".m4a",
    ".flac",
    ".ogg",
    ".aac",
    ".wma",
    ".mp4",
    ".m4v",
    ".mov",
    ".mkv",
    ".webm",
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
]


def load_json_if_exists(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return data if isinstance(data, dict) else {}


def build_audio_lookup(state_path: Path) -> dict[str, Path]:
    state = load_json_if_exists(state_path)
    lookup: dict[str, Path] = {}
    for record in state.get("files", {}).values():
        if not isinstance(record, dict) or record.get("status") != "complete":
            continue
        input_dir = Path(str(record.get("input_dir", "")))
        relative_path = str(record.get("relative_path", ""))
        audio_path = input_dir / relative_path if input_dir and relative_path else None
        if not audio_path:
            continue
        for output in record.get("project_outputs", []) + record.get("overall_outputs", []):
            output_path = Path(str(output))
            if output_path.suffix.lower() == ".json":
                lookup[str(output_path.resolve())] = audio_path
    return lookup


def discover_audio_from_audio_dir(whisper_json: Path, transcripts_dir: Path, audio_dir: Path | None) -> Path | None:
    if audio_dir is None:
        return None
    try:
        relative = whisper_json.relative_to(transcripts_dir)
    except ValueError:
        relative = Path(whisper_json.name)
    stem = strip_timestamp_suffix(relative.with_suffix("").name)
    relative_parent = relative.parent
    for extension in SUPPORTED_AUDIO_EXTENSIONS:
        candidate = audio_dir / relative_parent / f"{stem}{extension}"
        if candidate.exists():
            return candidate
        upper_candidate = audio_dir / relative_parent / f"{stem}{extension.upper()}"
        if upper_candidate.exists():
            return upper_candidate
    return None


def strip_timestamp_suffix(stem: str) -> str:
    marker = "_created-"
    if marker not in stem:
        return stem
    return stem.split(marker, 1)[0]


def iter_whisper_json_files(transcripts_dir: Path) -> list[Path]:
    if not transcripts_dir.exists():
        return []
    return sorted(
        [
            path
            for path in transcripts_dir.rglob("*.json")
            if path.is_file()
            and not path.name.endswith(".speaker.json")
            and not path.name.endswith(".diarization.json")
            and path.name not in {"input-output-mapping.json"}
        ],
        key=lambda path: path.as_posix().lower(),
    )


def process_transcript_set(
    transcripts_dir: Path,
    output_dir: Path,
    audio_lookup: dict[str, Path],
    manifest: dict[str, Any],
    manifest_path: Path,
    config: DiarizationConfig,
    min_overlap_ratio: float,
    cache_dir: Path,
    audio_dir: Path | None,
    force: bool,
    dry_run: bool,
) -> tuple[int, int, int, int]:
    completed = 0
    skipped = 0
    missing_audio = 0
    failed = 0

    for whisper_json in iter_whisper_json_files(transcripts_dir):
        output_base = output_base_for_whisper_json(whisper_json, transcripts_dir, output_dir)
        key = manifest_key(whisper_json, output_base)
        audio_path = audio_lookup.get(str(whisper_json.resolve()))
        if audio_path is None:
            audio_path = discover_audio_from_audio_dir(whisper_json, transcripts_dir, audio_dir)

        print(f"Processing Whisper JSON: {whisper_json}", flush=True)
        if speaker_outputs_complete(output_base) and not force:
            skipped += 1
            update_job(manifest, key, "skipped_existing", whisper_json, output_base, audio_path)
            save_manifest(manifest_path, manifest)
            print(f"Skipped existing: {output_base}", flush=True)
            continue

        if audio_path is None or not audio_path.exists():
            missing_audio += 1
            update_job(manifest, key, "skipped_missing_audio", whisper_json, output_base, audio_path)
            save_manifest(manifest_path, manifest)
            print(f"Skipped missing audio for: {whisper_json}", flush=True)
            continue

        if dry_run:
            update_job(manifest, key, "pending", whisper_json, output_base, audio_path)
            print(f"Would process: audio={audio_path} whisper={whisper_json} output={output_base}", flush=True)
            continue

        try:
            output_paths, cache_hit = run_single_diarization(
                audio_path,
                whisper_json,
                output_base,
                config,
                min_overlap_ratio,
                cache_dir,
                force=force,
                dry_run=False,
            )
            completed += 1
            update_job(manifest, key, "completed", whisper_json, output_base, audio_path, output_paths, cache_hit=cache_hit)
            print(f"Completed: {output_base}", flush=True)
        except Exception as exc:
            failed += 1
            update_job(manifest, key, "failed", whisper_json, output_base, audio_path, error_message=str(exc))
            print(f"Failed: {whisper_json}: {exc}", file=sys.stderr, flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)
        finally:
            save_manifest(manifest_path, manifest)

    return completed, skipped, missing_audio, failed


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Backfill speaker diarization for existing Whisper JSON outputs.")
    parser.add_argument("--transcripts-dir", type=Path, default=Path("output"))
    parser.add_argument("--output-dir", type=Path, default=Path(os.environ.get("DIARIZATION_OUTPUT_DIR", "output_pyannote")))
    parser.add_argument("--overall-transcripts-dir", type=Path, default=Path("output_overall"))
    parser.add_argument("--overall-output-dir", type=Path, default=Path(os.environ.get("DIARIZATION_OVERALL_OUTPUT_DIR", "output_pyannote_overall")))
    parser.add_argument("--audio-dir", type=Path)
    parser.add_argument("--state-path", type=Path, default=Path("state/progress.json"))
    parser.add_argument("--manifest-path", type=Path, default=Path("state/diarization-progress.json"))
    parser.add_argument("--cache-dir", type=Path, default=Path(os.environ.get("DIARIZATION_CACHE_DIR", "state/diarization-cache")))
    parser.add_argument("--backend", default=os.environ.get("DIARIZATION_BACKEND", "pyannote"))
    parser.add_argument("--model", default=os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"))
    parser.add_argument("--min-overlap-ratio", type=float, default=float(os.environ.get("DIARIZATION_MIN_OVERLAP_RATIO", "0.3")))
    parser.add_argument("--num-speakers", type=int, default=parse_optional_int(os.environ.get("DIARIZATION_NUM_SPEAKERS")))
    parser.add_argument("--min-speakers", type=int, default=parse_optional_int(os.environ.get("DIARIZATION_MIN_SPEAKERS")))
    parser.add_argument("--max-speakers", type=int, default=parse_optional_int(os.environ.get("DIARIZATION_MAX_SPEAKERS")))
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.backend != "pyannote":
        raise ValueError("Only the pyannote diarization backend is implemented.")
    if not args.dry_run and not os.environ.get("PYANNOTE_AUTH_TOKEN"):
        raise RuntimeError("PYANNOTE_AUTH_TOKEN is required unless --dry-run is used.")

    config = DiarizationConfig(args.backend, args.model, args.num_speakers, args.min_speakers, args.max_speakers)
    audio_lookup = build_audio_lookup(args.state_path)
    manifest = load_manifest(args.manifest_path)

    totals = [0, 0, 0, 0]
    for transcripts_dir, output_dir in [
        (args.transcripts_dir, args.output_dir),
        (args.overall_transcripts_dir, args.overall_output_dir),
    ]:
        completed, skipped, missing_audio, failed = process_transcript_set(
            transcripts_dir,
            output_dir,
            audio_lookup,
            manifest,
            args.manifest_path,
            config,
            args.min_overlap_ratio,
            args.cache_dir,
            args.audio_dir,
            args.force,
            args.dry_run,
        )
        totals[0] += completed
        totals[1] += skipped
        totals[2] += missing_audio
        totals[3] += failed

    print(
        f"Diarization done. Completed: {totals[0]}. Skipped existing: {totals[1]}. "
        f"Skipped missing audio: {totals[2]}. Failed: {totals[3]}.",
        flush=True,
    )
    return 1 if totals[3] else 0


if __name__ == "__main__":
    raise SystemExit(main())

