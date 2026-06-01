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
from diarization.audio_preprocess import parse_audio_preprocess_mode
from diarization.export_speaker_transcript import output_base_for_whisper_json, speaker_outputs_complete
from diarization.filename_normalization import parse_safe_output_policy
from diarization.manifest import load_manifest, manifest_key, save_manifest, update_job
from diarization.progress import ProgressContext, parse_progress_enabled
from diarization.pyannote_runner import (
    DiarizationOutOfMemoryError,
    PyannoteModelAccessError,
    PyannoteDiarizationBackend,
    cleanup_runtime_memory,
    parse_oom_fallback,
    parse_tf32_mode,
)
from scripts.run_diarization import (
    diarize_with_cache,
    export_diarization_from_segments,
    parse_bool,
    parse_optional_int,
    run_single_diarization,
)


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


def equivalent_output_paths(path: Path) -> list[Path]:
    paths = [path]
    parts = path.parts
    if len(parts) >= 4 and parts[0] == "/" and parts[1] == "outputs" and parts[2].startswith("output-"):
        paths.append(ROOT / "output" / Path(*parts[3:]))
    elif len(parts) >= 3 and parts[0] == "/" and parts[1] == "overall-output":
        paths.append(ROOT / "output_overall" / Path(*parts[2:]))
    return paths


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
                for equivalent_path in equivalent_output_paths(output_path):
                    lookup[str(equivalent_path.resolve())] = audio_path
    return lookup


def build_state_output_jobs(
    state_path: Path,
    project_transcripts_dir: Path,
    project_output_dir: Path,
    overall_transcripts_dir: Path,
    overall_output_dir: Path,
    filename_policy: str,
) -> list[dict[str, Any]]:
    state = load_json_if_exists(state_path)
    jobs: list[dict[str, Any]] = []
    existing_names_by_output: dict[Path, dict[Path, set[str]]] = {}
    for record in state.get("files", {}).values():
        if not isinstance(record, dict) or record.get("status") != "complete":
            continue
        input_dir = Path(str(record.get("input_dir", "")))
        relative_path = str(record.get("relative_path", ""))
        audio_path = input_dir / relative_path if input_dir and relative_path else None
        if audio_path is None:
            continue

        targets: list[dict[str, Path]] = []
        for output in record.get("project_outputs", []):
            output_path = Path(str(output))
            if output_path.suffix.lower() != ".json":
                continue
            for transcript_path in equivalent_output_paths(output_path):
                if transcript_path.exists():
                    existing_names = existing_names_by_output.setdefault(project_output_dir, {})
                    targets.append(
                        {
                            "whisper_json": transcript_path,
                            "output_base": output_base_for_whisper_json(
                                transcript_path,
                                project_transcripts_dir,
                                project_output_dir,
                                existing_names,
                                filename_policy,
                            ),
                            "legacy_output_base": project_output_dir / relative_to_transcripts_dir(transcript_path, project_transcripts_dir).with_suffix(""),
                        }
                    )
                    break
        for output in record.get("overall_outputs", []):
            output_path = Path(str(output))
            if output_path.suffix.lower() != ".json":
                continue
            for transcript_path in equivalent_output_paths(output_path):
                if transcript_path.exists():
                    existing_names = existing_names_by_output.setdefault(overall_output_dir, {})
                    targets.append(
                        {
                            "whisper_json": transcript_path,
                            "output_base": output_base_for_whisper_json(
                                transcript_path,
                                overall_transcripts_dir,
                                overall_output_dir,
                                existing_names,
                                filename_policy,
                            ),
                            "legacy_output_base": overall_output_dir / relative_to_transcripts_dir(transcript_path, overall_transcripts_dir).with_suffix(""),
                        }
                    )
                    break
        if targets:
            jobs.append({"audio_path": audio_path, "targets": targets})
    return jobs


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


def relative_to_transcripts_dir(whisper_json: Path, transcripts_dir: Path) -> Path:
    try:
        return whisper_json.relative_to(transcripts_dir)
    except ValueError:
        return whisper_json.resolve().relative_to(transcripts_dir.resolve())


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
    verbose: bool = False,
    tf32_mode: str = "false",
    progress: bool = False,
    oom_fallback: str = "cpu",
    filename_policy: str = "auto",
    audio_preprocess: str = "auto",
    audio_preprocess_dir: Path = Path("/tmp/auto-whisper-diarization"),
) -> tuple[int, int, int, int]:
    completed = 0
    skipped = 0
    missing_audio = 0
    failed = 0
    entries: list[tuple[Path, Path, Path, str, Path | None]] = []
    existing_safe_names: dict[Path, set[str]] = {}

    for whisper_json in iter_whisper_json_files(transcripts_dir):
        output_base = output_base_for_whisper_json(
            whisper_json,
            transcripts_dir,
            output_dir,
            existing_safe_names,
            filename_policy,
        )
        legacy_output_base = output_dir / relative_to_transcripts_dir(whisper_json, transcripts_dir).with_suffix("")
        key = manifest_key(whisper_json, output_base)
        audio_path = audio_lookup.get(str(whisper_json.resolve()))
        if audio_path is None:
            audio_path = discover_audio_from_audio_dir(whisper_json, transcripts_dir, audio_dir)
        entries.append((whisper_json, output_base, legacy_output_base, key, audio_path))

    processable_total = sum(
        1
        for _whisper_json, output_base, legacy_output_base, _key, audio_path in entries
        if (
            force
            or not (speaker_outputs_complete(output_base) or speaker_outputs_complete(legacy_output_base))
        )
        and audio_path is not None
        and audio_path.exists()
    )
    processable_index = 0

    for whisper_json, output_base, legacy_output_base, key, audio_path in entries:
        print(f"Processing Whisper JSON: {whisper_json}", flush=True)
        if verbose:
            print(f"Resolved output base: {output_base}", flush=True)
            print(f"Resolved audio path: {audio_path or '<missing>'}", flush=True)
        complete_output_base = output_base if speaker_outputs_complete(output_base) else legacy_output_base
        if (speaker_outputs_complete(output_base) or speaker_outputs_complete(legacy_output_base)) and not force:
            skipped += 1
            update_job(manifest, key, "skipped_existing", whisper_json, complete_output_base, audio_path)
            save_manifest(manifest_path, manifest)
            print(f"Skipped existing: {complete_output_base}", flush=True)
            continue

        if audio_path is None or not audio_path.exists():
            missing_audio += 1
            update_job(manifest, key, "skipped_missing_audio", whisper_json, output_base, audio_path)
            save_manifest(manifest_path, manifest)
            print(f"Skipped missing audio for: {whisper_json}", flush=True)
            continue

        processable_index += 1
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
                verbose=verbose,
                tf32_mode=tf32_mode,
                progress=progress,
                progress_context=ProgressContext(file_index=processable_index, file_total=processable_total),
                oom_fallback=oom_fallback,
                filename_policy=filename_policy,
                audio_preprocess=audio_preprocess,
                audio_preprocess_dir=audio_preprocess_dir,
            )
            completed += 1
            update_job(manifest, key, "completed", whisper_json, output_base, audio_path, output_paths, cache_hit=cache_hit)
            print(f"Completed: {output_base}", flush=True)
        except Exception as exc:
            if isinstance(exc, DiarizationOutOfMemoryError) and oom_fallback == "fail":
                raise
            failed += 1
            update_job(manifest, key, "failed", whisper_json, output_base, audio_path, error_message=str(exc))
            print(f"Failed: {whisper_json}: {exc}", file=sys.stderr, flush=True)
            print(traceback.format_exc(), file=sys.stderr, flush=True)
        finally:
            save_manifest(manifest_path, manifest)

    return completed, skipped, missing_audio, failed


def process_state_output_jobs(
    jobs: list[dict[str, Any]],
    manifest: dict[str, Any],
    manifest_path: Path,
    config: DiarizationConfig,
    min_overlap_ratio: float,
    cache_dir: Path,
    force: bool,
    dry_run: bool,
    verbose: bool = False,
    tf32_mode: str = "false",
    progress: bool = False,
    oom_fallback: str = "cpu",
    filename_policy: str = "auto",
    audio_preprocess: str = "auto",
    audio_preprocess_dir: Path = Path("/tmp/auto-whisper-diarization"),
) -> tuple[int, int, int, int]:
    completed = 0
    skipped = 0
    missing_audio = 0
    failed = 0
    processable_jobs = [
        job
        for job in jobs
        if Path(job["audio_path"]).exists()
        and any(
            force
            or not (
                speaker_outputs_complete(target["output_base"])
                or speaker_outputs_complete(target["legacy_output_base"])
            )
            for target in job["targets"]
        )
    ]
    processable_total = len(processable_jobs)
    processable_index = 0

    for job in jobs:
        audio_path = Path(job["audio_path"])
        targets = job["targets"]
        print(f"Processing audio for paired diarization outputs: {audio_path}", flush=True)
        if audio_path is None or not audio_path.exists():
            missing_audio += 1
            for target in targets:
                key = manifest_key(target["whisper_json"], target["output_base"])
                update_job(manifest, key, "skipped_missing_audio", target["whisper_json"], target["output_base"], audio_path)
            save_manifest(manifest_path, manifest)
            print(f"Skipped missing audio for: {audio_path}", flush=True)
            continue

        pending_targets = [
            target
            for target in targets
            if force
            or not (
                speaker_outputs_complete(target["output_base"])
                or speaker_outputs_complete(target["legacy_output_base"])
            )
        ]
        if not pending_targets:
            skipped += 1
            for target in targets:
                complete_output_base = target["output_base"] if speaker_outputs_complete(target["output_base"]) else target["legacy_output_base"]
                key = manifest_key(target["whisper_json"], target["output_base"])
                update_job(manifest, key, "skipped_existing", target["whisper_json"], complete_output_base, audio_path)
            save_manifest(manifest_path, manifest)
            print(f"Skipped existing paired outputs: {audio_path}", flush=True)
            continue

        processable_index += 1
        if dry_run:
            for target in pending_targets:
                key = manifest_key(target["whisper_json"], target["output_base"])
                update_job(manifest, key, "pending", target["whisper_json"], target["output_base"], audio_path)
                print(f"Would process paired target: audio={audio_path} whisper={target['whisper_json']} output={target['output_base']}", flush=True)
            save_manifest(manifest_path, manifest)
            continue

        try:
            speaker_segments, cache_hit = diarize_with_cache(
                audio_path,
                config,
                cache_dir,
                force=force,
                verbose=verbose,
                tf32_mode=tf32_mode,
                progress=progress,
                progress_context=ProgressContext(file_index=processable_index, file_total=processable_total),
                oom_fallback=oom_fallback,
                audio_preprocess=audio_preprocess,
                audio_preprocess_dir=audio_preprocess_dir,
            )
            combined_output_paths: dict[str, Path] = {}
            for target_index, target in enumerate(pending_targets, start=1):
                output_paths = export_diarization_from_segments(
                    audio_path,
                    target["whisper_json"],
                    target["output_base"],
                    config,
                    min_overlap_ratio,
                    speaker_segments,
                    verbose=verbose,
                    progress=progress,
                    progress_context=ProgressContext(file_index=processable_index, file_total=processable_total),
                    filename_policy=filename_policy,
                )
                for name, path in output_paths.items():
                    combined_output_paths[f"target_{target_index}_{name}"] = path
                key = manifest_key(target["whisper_json"], target["output_base"])
                update_job(manifest, key, "completed", target["whisper_json"], target["output_base"], audio_path, output_paths, cache_hit=cache_hit)
                print(f"Completed: {target['output_base']}", flush=True)
            completed += 1
            del speaker_segments
            del combined_output_paths
        except Exception as exc:
            if isinstance(exc, DiarizationOutOfMemoryError) and oom_fallback == "fail":
                raise
            failed += 1
            error_message = str(exc)
            traceback_text = traceback.format_exc()
            for target in pending_targets:
                key = manifest_key(target["whisper_json"], target["output_base"])
                update_job(manifest, key, "failed", target["whisper_json"], target["output_base"], audio_path, error_message=error_message)
            print(f"Failed paired diarization for {audio_path}: {error_message}", file=sys.stderr, flush=True)
            print(traceback_text, file=sys.stderr, flush=True)
            del traceback_text
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
    parser.add_argument("--safe-output-filenames", choices=["auto", "true", "false"], default=parse_safe_output_policy(os.environ.get("SAFE_OUTPUT_FILENAMES")))
    parser.add_argument("--audio-preprocess", choices=["auto", "always", "false"], default=parse_audio_preprocess_mode(os.environ.get("DIARIZATION_AUDIO_PREPROCESS")))
    parser.add_argument("--audio-preprocess-dir", type=Path, default=Path(os.environ.get("DIARIZATION_AUDIO_PREPROCESS_DIR", "/tmp/auto-whisper-diarization")))
    parser.add_argument("--backend", default=os.environ.get("DIARIZATION_BACKEND", "pyannote"))
    parser.add_argument("--model", default=os.environ.get("DIARIZATION_MODEL", "pyannote/speaker-diarization-community-1"))
    parser.add_argument("--min-overlap-ratio", type=float, default=float(os.environ.get("DIARIZATION_MIN_OVERLAP_RATIO", "0.3")))
    parser.add_argument("--num-speakers", type=int, default=parse_optional_int(os.environ.get("DIARIZATION_NUM_SPEAKERS")))
    parser.add_argument("--min-speakers", type=int, default=parse_optional_int(os.environ.get("DIARIZATION_MIN_SPEAKERS")))
    parser.add_argument("--max-speakers", type=int, default=parse_optional_int(os.environ.get("DIARIZATION_MAX_SPEAKERS")))
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
    try:
        if args.backend != "pyannote":
            raise ValueError("Only the pyannote diarization backend is implemented.")
        if not args.dry_run and not os.environ.get("PYANNOTE_AUTH_TOKEN"):
            raise RuntimeError("PYANNOTE_AUTH_TOKEN is required unless --dry-run is used.")

        config = DiarizationConfig(args.backend, args.model, args.num_speakers, args.min_speakers, args.max_speakers)
        print(
            f"Diarization startup: model={config.model} verbose={args.verbose} "
            f"progress={args.progress and not args.dry_run} tf32={args.tf32} "
            f"oom_fallback={args.oom_fallback} filename_policy={args.safe_output_filenames} "
            f"audio_preprocess={args.audio_preprocess} dry_run={args.dry_run}",
            flush=True,
        )
        if not args.dry_run:
            try:
                PyannoteDiarizationBackend(config, tf32_mode=args.tf32, verbose=args.verbose).validate_access()
            except PyannoteModelAccessError as exc:
                print(f"Pyannote access check failed: {exc}", file=sys.stderr, flush=True)
                return 2

        manifest = load_manifest(args.manifest_path)
        totals = [0, 0, 0, 0]
        jobs = build_state_output_jobs(
            args.state_path,
            args.transcripts_dir,
            args.output_dir,
            args.overall_transcripts_dir,
            args.overall_output_dir,
            args.safe_output_filenames,
        )
        if jobs:
            completed, skipped, missing_audio, failed = process_state_output_jobs(
                jobs,
                manifest,
                args.manifest_path,
                config,
                args.min_overlap_ratio,
                args.cache_dir,
                args.force,
                args.dry_run,
                args.verbose,
                args.tf32,
                args.progress and not args.dry_run,
                args.oom_fallback,
                args.safe_output_filenames,
                args.audio_preprocess,
                args.audio_preprocess_dir,
            )
            totals = [completed, skipped, missing_audio, failed]
        else:
            audio_lookup = build_audio_lookup(args.state_path)
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
                    args.verbose,
                    args.tf32,
                    args.progress and not args.dry_run,
                    args.oom_fallback,
                    args.safe_output_filenames,
                    args.audio_preprocess,
                    args.audio_preprocess_dir,
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
    finally:
        cleanup_runtime_memory(verbose=args.verbose, label="Final runtime cleanup before exit")


if __name__ == "__main__":
    raise SystemExit(main())
