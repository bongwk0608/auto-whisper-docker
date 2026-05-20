from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import sys
import tempfile
import traceback
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Any, Iterator, NamedTuple

import torch
import whisper
from whisper.utils import get_writer


ALL_FORMATS = ["txt", "json", "tsv", "srt", "vtt"]


class InputOutputPair(NamedTuple):
    id: str
    input_dir: Path
    output_dir: Path


class PendingFile(NamedTuple):
    pair: InputOutputPair
    index: int
    total: int
    source_path: Path
    output_media_path: Path
    state_key: str
    display_key: str
    fingerprint: dict[str, Any]


class PreparedPair(NamedTuple):
    skipped: int
    pending: list[PendingFile]
    mapping: dict[str, Any] | None


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def parse_list(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


def parse_path_list(value: str) -> list[str]:
    return [item.strip() for item in value.split(";") if item.strip()]


def parse_formats(value: str) -> list[str]:
    requested = env("WHISPER_OUTPUT_FORMAT", value).lower()
    if requested == "all":
        return ALL_FORMATS
    formats = parse_list(requested)
    unknown = sorted(set(formats) - set(ALL_FORMATS))
    if unknown:
        raise ValueError(f"Unsupported WHISPER_OUTPUT_FORMAT value(s): {', '.join(unknown)}")
    if not formats:
        raise ValueError("WHISPER_OUTPUT_FORMAT must be one of txt,json,tsv,srt,vtt,all")
    return formats


def parse_fingerprint_mode(value: str) -> str:
    value = value.lower()
    if value not in {"metadata", "sha256"}:
        raise ValueError("FINGERPRINT_MODE must be metadata or sha256")
    return value


def parse_input_output_pairs() -> list[InputOutputPair]:
    raw_pairs = env("INPUT_OUTPUT_PAIRS")
    if raw_pairs:
        try:
            pairs_data = json.loads(raw_pairs)
        except json.JSONDecodeError as exc:
            raise ValueError("INPUT_OUTPUT_PAIRS must be valid JSON") from exc

        if not isinstance(pairs_data, list) or not pairs_data:
            raise ValueError("INPUT_OUTPUT_PAIRS must be a non-empty JSON array")

        pairs: list[InputOutputPair] = []
        for index, item in enumerate(pairs_data, start=1):
            if not isinstance(item, dict):
                raise ValueError("Each INPUT_OUTPUT_PAIRS item must be an object")
            input_value = str(item.get("input", "")).strip()
            output_value = str(item.get("output", "")).strip()
            if not input_value or not output_value:
                raise ValueError("Each INPUT_OUTPUT_PAIRS item must include input and output")
            pairs.append(
                InputOutputPair(
                    id=f"pair-{index:03d}",
                    input_dir=Path(input_value),
                    output_dir=Path(output_value),
                )
            )
        return pairs

    input_dir = Path(env("INPUT_DIR", "/input"))
    output_dir = Path(env("PROJECT_OUTPUT_DIR", "/project-output"))
    return [InputOutputPair(id="pair-001", input_dir=input_dir, output_dir=output_dir)]


def validate_pairs(pairs: list[InputOutputPair]) -> None:
    if not pairs:
        raise ValueError("At least one input/output pair is required")

    for pair in pairs:
        if not pair.input_dir.exists() or not pair.input_dir.is_dir():
            raise FileNotFoundError(f"Input directory does not exist or is not a directory: {pair.input_dir}")
        try:
            pair.output_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise OSError(f"Output directory cannot be created: {pair.output_dir}") from exc
        if not pair.output_dir.exists() or not pair.output_dir.is_dir():
            raise FileNotFoundError(f"Output path is not a directory: {pair.output_dir}")


def file_fingerprint(path: Path, mode: str) -> dict[str, Any]:
    stat = path.stat()
    fingerprint: dict[str, Any] = {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
    }
    if mode == "metadata":
        return fingerprint
    if mode != "sha256":
        raise ValueError("FINGERPRINT_MODE must be metadata or sha256")

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    fingerprint["sha256"] = digest.hexdigest()
    return fingerprint


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
        temp_name = handle.name
    Path(temp_name).replace(path)


def atomic_write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fieldnames})
        temp_name = handle.name
    Path(temp_name).replace(path)


def load_state(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"version": 1, "runs": {}, "files": {}}
    with path.open("r", encoding="utf-8") as handle:
        state = json.load(handle)
    state.setdefault("version", 1)
    state.setdefault("runs", {})
    state.setdefault("files", {})
    return state


def scan_files(input_dir: Path, extensions: list[str]) -> list[Path]:
    files = [
        path
        for path in input_dir.rglob("*")
        if path.is_file() and path.suffix.lower() in extensions
    ]
    return sorted(files, key=lambda path: path.relative_to(input_dir).as_posix().lower())


def source_folder_name(input_dir: Path) -> str:
    try:
        return input_dir.resolve().name or "input"
    except OSError:
        return input_dir.name or "input"


def make_run_id(input_dir: Path) -> str:
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in source_folder_name(input_dir))
    return f"{safe_name}{source_timestamp_suffix(input_dir)}"


def state_run_key(pair_id: str, run_id: str) -> str:
    return f"{pair_id}:{run_id}"


def expected_outputs(base: Path, formats: list[str]) -> list[Path]:
    return [base.with_suffix(f".{fmt}") for fmt in formats]


def format_file_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y%m%d%H%M%S")


def source_timestamps(path: Path) -> tuple[str, str]:
    stat = path.stat()
    created_timestamp = getattr(stat, "st_birthtime", stat.st_ctime)
    return format_file_timestamp(created_timestamp), format_file_timestamp(stat.st_mtime)


def source_timestamp_suffix(path: Path) -> str:
    created, modified = source_timestamps(path)
    return f"_created-{created}_modified-{modified}"


def timestamped_output_base(path: Path, timestamp_source: Path | None = None) -> Path:
    timestamp_path = timestamp_source or path
    return path.with_name(f"{path.stem}{source_timestamp_suffix(timestamp_path)}")


def outputs_exist(paths: list[Path]) -> bool:
    return all(path.exists() and path.is_file() for path in paths)


def fingerprint_matches(record_fingerprint: dict[str, Any] | None, fingerprint: dict[str, Any]) -> bool:
    if not isinstance(record_fingerprint, dict):
        return False
    return all(record_fingerprint.get(key) == value for key, value in fingerprint.items())


def is_complete(record: dict[str, Any] | None, fingerprint: dict[str, Any], formats: list[str]) -> bool:
    if not record or record.get("status") != "complete":
        return False
    if not fingerprint_matches(record.get("fingerprint"), fingerprint):
        return False
    if sorted(record.get("formats", [])) != sorted(formats):
        return False
    project_outputs = [Path(path) for path in record.get("project_outputs", [])]
    return bool(project_outputs) and outputs_exist(project_outputs)


def choose_device(requested: str) -> str:
    requested = requested.lower()
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested == "cuda" and not torch.cuda.is_available():
        print("Requested CUDA, but torch cannot see a GPU. Falling back to CPU.", flush=True)
        return "cpu"
    if requested not in {"cuda", "cpu"}:
        raise ValueError("WHISPER_DEVICE must be auto, cuda, or cpu")
    return requested


def choose_fp16(requested: str, device: str) -> bool:
    requested = requested.lower()
    if requested == "auto":
        return device == "cuda"
    if requested in {"1", "true", "yes", "on"}:
        return True
    if requested in {"0", "false", "no", "off"}:
        return False
    raise ValueError("WHISPER_FP16 must be auto, true, or false")


def parse_bool(value: str, name: str) -> bool:
    value = value.lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ValueError(f"{name} must be true or false")


def parse_task(value: str) -> str:
    value = value.lower()
    if value not in {"transcribe", "translate"}:
        raise ValueError("WHISPER_TASK must be transcribe or translate")
    return value


def write_outputs(result: dict[str, Any], source_path: Path, project_base: Path, formats: list[str]) -> list[Path]:
    project_base_no_suffix = timestamped_output_base(project_base, timestamp_source=source_path)
    project_base.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as temp_dir:
        temp_dir_path = Path(temp_dir)
        for fmt in formats:
            writer = get_writer(fmt, str(temp_dir_path))
            writer(result, str(source_path), {"max_line_width": None, "max_line_count": None, "highlight_words": False})

        temp_outputs = expected_outputs(temp_dir_path / source_path.stem, formats)
        if not outputs_exist(temp_outputs):
            missing = [str(path) for path in temp_outputs if not path.exists()]
            raise RuntimeError(f"Whisper did not create expected temporary output(s): {', '.join(missing)}")

        project_outputs = expected_outputs(project_base_no_suffix, formats)
        for temp_output, project_output in zip(temp_outputs, project_outputs):
            project_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(temp_output), str(project_output))

    return project_outputs


@contextmanager
def transcription_source(source_path: Path, local_staging: bool, staging_dir: Path) -> Iterator[Path]:
    if not local_staging:
        yield source_path
        return

    staging_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="auto-whisper-", dir=staging_dir) as temp_dir:
        staged_path = Path(temp_dir) / source_path.name
        shutil.copy2(source_path, staged_path)
        yield staged_path


def mark_failure(
    state: dict[str, Any],
    key: str,
    fingerprint: dict[str, Any],
    fingerprint_mode: str,
    local_staging: bool,
    error: BaseException,
) -> None:
    state["files"][key] = {
        "status": "failed",
        "fingerprint": fingerprint,
        "fingerprint_mode": fingerprint_mode,
        "local_staging": local_staging,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def host_path_for_pair(values: list[str], index: int) -> str:
    return values[index] if index < len(values) else ""


def build_mapping_record(
    pair: InputOutputPair,
    pair_index: int,
    run_id: str,
    run_output_dir: Path,
    files_found: int,
    formats: list[str],
    host_input_dirs: list[str],
    host_output_dirs: list[str],
    fingerprint_mode: str,
    local_staging: bool,
) -> dict[str, Any]:
    created, modified = source_timestamps(pair.input_dir)
    return {
        "pair_id": pair.id,
        "host_input_dir": host_path_for_pair(host_input_dirs, pair_index),
        "container_input_dir": str(pair.input_dir),
        "host_output_root": host_path_for_pair(host_output_dirs, pair_index),
        "container_output_root": str(pair.output_dir),
        "run_id": run_id,
        "run_output_dir": str(run_output_dir),
        "source_folder_name": source_folder_name(pair.input_dir),
        "input_created_timestamp": created,
        "input_modified_timestamp": modified,
        "formats": ",".join(formats),
        "fingerprint_mode": fingerprint_mode,
        "local_staging": local_staging,
        "recursive_scan_enabled": True,
        "supported_file_count": files_found,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def write_mapping_manifests(output_roots: list[Path], mappings: list[dict[str, Any]]) -> None:
    if not mappings:
        return

    data = {
        "version": 1,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "mappings": mappings,
    }
    fieldnames = [
        "pair_id",
        "host_input_dir",
        "container_input_dir",
        "host_output_root",
        "container_output_root",
        "run_id",
        "run_output_dir",
        "source_folder_name",
        "input_created_timestamp",
        "input_modified_timestamp",
        "formats",
        "fingerprint_mode",
        "local_staging",
        "recursive_scan_enabled",
        "supported_file_count",
        "updated_at",
    ]

    for output_root in sorted(set(output_roots), key=lambda path: str(path)):
        atomic_write_json(output_root / "input-output-mapping.json", data)
        atomic_write_csv(output_root / "input-output-mapping.csv", mappings, fieldnames)


def prepare_pair(
    pair: InputOutputPair,
    pair_index: int,
    state: dict[str, Any],
    formats: list[str],
    extensions: list[str],
    run_config: dict[str, Any],
    host_input_dirs: list[str],
    host_output_dirs: list[str],
    fingerprint_mode: str,
    local_staging: bool,
) -> PreparedPair:
    files = scan_files(pair.input_dir, extensions)
    print(f"Found {len(files)} supported file(s) under {pair.input_dir}.", flush=True)
    if not files:
        return PreparedPair(0, [], None)

    active_run_ids = state.setdefault("active_run_ids", {})
    active_run_id = active_run_ids.get(pair.id)
    if active_run_id and (pair.output_dir / active_run_id).exists():
        run_id = active_run_id
    else:
        run_id = make_run_id(pair.input_dir)
        active_run_ids[pair.id] = run_id

    run_output_dir = pair.output_dir / run_id
    run_key = state_run_key(pair.id, run_id)
    run_output_dir.mkdir(parents=True, exist_ok=True)
    state["runs"][run_key] = {
        "run_id": run_id,
        "pair_id": pair.id,
        "input_dir": str(pair.input_dir),
        "output_dir": str(run_output_dir),
        **run_config,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    mapping = build_mapping_record(
        pair,
        pair_index,
        run_id,
        run_output_dir,
        len(files),
        formats,
        host_input_dirs,
        host_output_dirs,
        fingerprint_mode,
        local_staging,
    )

    skipped = 0
    pending: list[PendingFile] = []

    for index, source_path in enumerate(files, start=1):
        relative = source_path.relative_to(pair.input_dir)
        relative_key = relative.as_posix()
        key = f"{pair.id}:{relative_key}"
        project_media_path = run_output_dir / relative
        fingerprint = file_fingerprint(source_path, fingerprint_mode)

        if is_complete(state["files"].get(key), fingerprint, formats):
            skipped += 1
            print(f"{pair.id} [{index}/{len(files)}] Skipping complete file: {relative_key}", flush=True)
            continue

        pending.append(PendingFile(pair, index, len(files), source_path, project_media_path, key, relative_key, fingerprint))

    if not pending:
        active_run_ids.pop(pair.id, None)
        state["runs"][run_key]["completed_at"] = datetime.now().isoformat(timespec="seconds")
        print(f"{pair.id} done. Completed: 0. Skipped: {skipped}. Failed: 0. Output: {run_output_dir}", flush=True)

    return PreparedPair(skipped, pending, mapping)


def main() -> int:
    state_dir = Path(env("STATE_DIR", "/state"))
    state_path = state_dir / "progress.json"
    model_name = env("WHISPER_MODEL", "small")
    language = env("WHISPER_LANGUAGE", "")
    task = parse_task(env("WHISPER_TASK", "transcribe"))
    condition_on_previous_text = parse_bool(env("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "true"), "WHISPER_CONDITION_ON_PREVIOUS_TEXT")
    verbose = parse_bool(env("WHISPER_VERBOSE", "false"), "WHISPER_VERBOSE")
    fingerprint_mode = parse_fingerprint_mode(env("FINGERPRINT_MODE", "metadata"))
    local_staging = parse_bool(env("LOCAL_STAGING", "false"), "LOCAL_STAGING")
    local_staging_dir = Path(env("LOCAL_STAGING_DIR", "/tmp/auto-whisper-staging"))
    formats = parse_formats("all")
    extensions = parse_list(env("SUPPORTED_EXTENSIONS", ".mp3,.wav,.m4a,.mp4,.mov,.mkv,.webm,.flac,.ogg,.aac,.wma"))
    pairs = parse_input_output_pairs()
    validate_pairs(pairs)
    host_input_dirs = parse_path_list(env("SOURCE_DIRS"))
    host_output_dirs = parse_path_list(env("OUTPUT_DIRS"))

    state = load_state(state_path)
    run_config = {
        "model": model_name,
        "formats": formats,
        "language": language,
        "task": task,
        "condition_on_previous_text": condition_on_previous_text,
        "verbose": verbose,
        "fingerprint_mode": fingerprint_mode,
        "local_staging": local_staging,
        "local_staging_dir": str(local_staging_dir) if local_staging else "",
    }

    skipped = 0
    pending: list[PendingFile] = []
    mappings: list[dict[str, Any]] = []
    for pair_index, pair in enumerate(pairs):
        prepared = prepare_pair(
            pair,
            pair_index,
            state,
            formats,
            extensions,
            run_config,
            host_input_dirs,
            host_output_dirs,
            fingerprint_mode,
            local_staging,
        )
        skipped += prepared.skipped
        pending.extend(prepared.pending)
        if prepared.mapping:
            mappings.append(prepared.mapping)

    write_mapping_manifests([pair.output_dir for pair in pairs], mappings)
    atomic_write_json(state_path, state)
    if not pending:
        print(f"Done. Completed: 0. Skipped: {skipped}. Failed: 0.", flush=True)
        return 0

    device = choose_device(env("WHISPER_DEVICE", "auto"))
    fp16 = choose_fp16(env("WHISPER_FP16", "auto"), device)
    print(f"Loading Whisper model '{model_name}' on {device} with fp16={fp16}.", flush=True)
    model = whisper.load_model(model_name, device=device)

    completed = 0
    failed = 0

    for item in pending:

        print(f"{item.pair.id} [{item.index}/{item.total}] Transcribing: {item.display_key}", flush=True)
        try:
            options: dict[str, Any] = {"fp16": fp16}
            if language:
                options["language"] = language
            options["task"] = task
            options["condition_on_previous_text"] = condition_on_previous_text
            options["verbose"] = verbose
            with transcription_source(item.source_path, local_staging, local_staging_dir) as source_for_transcription:
                result = model.transcribe(str(source_for_transcription), **options)
            project_outputs = write_outputs(result, item.source_path, item.output_media_path, formats)
            state["files"][item.state_key] = {
                "status": "complete",
                "pair_id": item.pair.id,
                "run_id": state["active_run_ids"][item.pair.id],
                "run_key": state_run_key(item.pair.id, state["active_run_ids"][item.pair.id]),
                "input_dir": str(item.pair.input_dir),
                "output_dir": str(item.pair.output_dir),
                "relative_path": item.display_key,
                "fingerprint": item.fingerprint,
                "fingerprint_mode": fingerprint_mode,
                "local_staging": local_staging,
                "formats": formats,
                "project_outputs": [str(path) for path in project_outputs],
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
            completed += 1
        except Exception as exc:
            failed += 1
            print(f"Failed: {item.pair.id}:{item.display_key}: {exc}", file=sys.stderr, flush=True)
            mark_failure(state, item.state_key, item.fingerprint, fingerprint_mode, local_staging, exc)
        finally:
            run_id = state["active_run_ids"].get(item.pair.id)
            run_key = state_run_key(item.pair.id, run_id) if run_id else ""
            if run_key in state["runs"]:
                state["runs"][run_key]["updated_at"] = datetime.now().isoformat(timespec="seconds")
            atomic_write_json(state_path, state)

    for pair in pairs:
        files = scan_files(pair.input_dir, extensions)
        incomplete = any(
            state["files"].get(f"{pair.id}:{path.relative_to(pair.input_dir).as_posix()}", {}).get("status") != "complete"
            for path in files
        )
        run_id = state.get("active_run_ids", {}).get(pair.id)
        run_key = state_run_key(pair.id, run_id) if run_id else ""
        if not incomplete and run_key in state["runs"]:
            state["active_run_ids"].pop(pair.id, None)
            state["runs"][run_key]["completed_at"] = datetime.now().isoformat(timespec="seconds")
    if not state.get("active_run_ids"):
        state.pop("active_run_ids", None)
        state.pop("active_run_id", None)
    atomic_write_json(state_path, state)

    print(
        f"Done. Completed: {completed}. Skipped: {skipped}. Failed: {failed}.",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        raise
