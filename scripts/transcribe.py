from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import tempfile
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Any

import torch
import whisper
from whisper.utils import get_writer


ALL_FORMATS = ["txt", "json", "tsv", "srt", "vtt"]


def env(name: str, default: str = "") -> str:
    return os.environ.get(name, default).strip()


def parse_list(value: str) -> list[str]:
    return [item.strip().lower() for item in value.split(",") if item.strip()]


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


def file_fingerprint(path: Path) -> dict[str, Any]:
    stat = path.stat()
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return {
        "size": stat.st_size,
        "mtime_ns": stat.st_mtime_ns,
        "sha256": digest.hexdigest(),
    }


def atomic_write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(data, handle, indent=2, sort_keys=True)
        handle.write("\n")
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
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    safe_name = "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in source_folder_name(input_dir))
    return f"{safe_name}_{timestamp}"


def expected_outputs(base: Path, formats: list[str]) -> list[Path]:
    return [base.with_suffix(f".{fmt}") for fmt in formats]


def format_file_timestamp(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp).strftime("%Y%m%d%H%M%S")


def source_timestamp_suffix(path: Path) -> str:
    stat = path.stat()
    created_timestamp = getattr(stat, "st_birthtime", stat.st_ctime)
    created = format_file_timestamp(created_timestamp)
    modified = format_file_timestamp(stat.st_mtime)
    return f"_created-{created}_modified-{modified}"


def timestamped_output_base(path: Path, timestamp_source: Path | None = None) -> Path:
    timestamp_path = timestamp_source or path
    return path.with_name(f"{path.stem}{source_timestamp_suffix(timestamp_path)}")


def outputs_exist(paths: list[Path]) -> bool:
    return all(path.exists() and path.is_file() for path in paths)


def is_complete(record: dict[str, Any] | None, fingerprint: dict[str, Any], formats: list[str]) -> bool:
    if not record or record.get("status") != "complete":
        return False
    if record.get("fingerprint") != fingerprint:
        return False
    if sorted(record.get("formats", [])) != sorted(formats):
        return False
    source_outputs = [Path(path) for path in record.get("source_outputs", [])]
    project_outputs = [Path(path) for path in record.get("project_outputs", [])]
    return outputs_exist(source_outputs) and outputs_exist(project_outputs)


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


def write_outputs(result: dict[str, Any], source_path: Path, project_base: Path, formats: list[str]) -> tuple[list[Path], list[Path]]:
    source_base = timestamped_output_base(source_path)
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

        source_outputs = expected_outputs(source_base, formats)
        project_outputs = expected_outputs(project_base_no_suffix, formats)
        for temp_output, source_output, project_output in zip(temp_outputs, source_outputs, project_outputs):
            source_output.parent.mkdir(parents=True, exist_ok=True)
            project_output.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(temp_output, source_output)
            shutil.copy2(temp_output, project_output)

    return source_outputs, project_outputs


def mark_failure(state: dict[str, Any], key: str, fingerprint: dict[str, Any], error: BaseException) -> None:
    state["files"][key] = {
        "status": "failed",
        "fingerprint": fingerprint,
        "error": str(error),
        "traceback": traceback.format_exc(),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def main() -> int:
    input_dir = Path(env("INPUT_DIR", "/input"))
    project_output_dir = Path(env("PROJECT_OUTPUT_DIR", "/project-output"))
    state_dir = Path(env("STATE_DIR", "/state"))
    state_path = state_dir / "progress.json"
    model_name = env("WHISPER_MODEL", "small")
    language = env("WHISPER_LANGUAGE", "")
    task = parse_task(env("WHISPER_TASK", "transcribe"))
    condition_on_previous_text = parse_bool(env("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "true"), "WHISPER_CONDITION_ON_PREVIOUS_TEXT")
    verbose = parse_bool(env("WHISPER_VERBOSE", "false"), "WHISPER_VERBOSE")
    formats = parse_formats("all")
    extensions = parse_list(env("SUPPORTED_EXTENSIONS", ".mp3,.wav,.m4a,.mp4,.mov,.mkv,.webm,.flac,.ogg,.aac,.wma"))

    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input directory does not exist or is not a directory: {input_dir}")

    files = scan_files(input_dir, extensions)
    print(f"Found {len(files)} supported file(s) under {input_dir}.", flush=True)
    if not files:
        return 0

    state = load_state(state_path)
    active_run_id = state.get("active_run_id")
    if active_run_id and (project_output_dir / active_run_id).exists():
        run_id = active_run_id
    else:
        run_id = make_run_id(input_dir)
        state["active_run_id"] = run_id

    run_output_dir = project_output_dir / run_id
    run_output_dir.mkdir(parents=True, exist_ok=True)
    state["runs"][run_id] = {
        "input_dir": str(input_dir),
        "output_dir": str(run_output_dir),
        "model": model_name,
        "formats": formats,
        "language": language,
        "task": task,
        "condition_on_previous_text": condition_on_previous_text,
        "verbose": verbose,
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }
    atomic_write_json(state_path, state)

    skipped = 0
    pending: list[tuple[int, Path, Path, str, dict[str, Any]]] = []

    for index, source_path in enumerate(files, start=1):
        relative = source_path.relative_to(input_dir)
        key = relative.as_posix()
        project_media_path = run_output_dir / relative
        fingerprint = file_fingerprint(source_path)

        if is_complete(state["files"].get(key), fingerprint, formats):
            skipped += 1
            print(f"[{index}/{len(files)}] Skipping complete file: {key}", flush=True)
            continue

        pending.append((index, source_path, project_media_path, key, fingerprint))

    if not pending:
        state.pop("active_run_id", None)
        state["runs"][run_id]["completed_at"] = datetime.now().isoformat(timespec="seconds")
        atomic_write_json(state_path, state)
        print(f"Done. Completed: 0. Skipped: {skipped}. Failed: 0. Output: {run_output_dir}", flush=True)
        return 0

    device = choose_device(env("WHISPER_DEVICE", "auto"))
    fp16 = choose_fp16(env("WHISPER_FP16", "auto"), device)
    print(f"Loading Whisper model '{model_name}' on {device} with fp16={fp16}.", flush=True)
    model = whisper.load_model(model_name, device=device)

    completed = 0
    failed = 0

    for index, source_path, project_media_path, key, fingerprint in pending:

        print(f"[{index}/{len(files)}] Transcribing: {key}", flush=True)
        try:
            options: dict[str, Any] = {"fp16": fp16}
            if language:
                options["language"] = language
            options["task"] = task
            options["condition_on_previous_text"] = condition_on_previous_text
            options["verbose"] = verbose
            result = model.transcribe(str(source_path), **options)
            source_outputs, project_outputs = write_outputs(result, source_path, project_media_path, formats)
            state["files"][key] = {
                "status": "complete",
                "run_id": run_id,
                "fingerprint": fingerprint,
                "formats": formats,
                "source_outputs": [str(path) for path in source_outputs],
                "project_outputs": [str(path) for path in project_outputs],
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
            completed += 1
        except Exception as exc:
            failed += 1
            print(f"Failed: {key}: {exc}", file=sys.stderr, flush=True)
            mark_failure(state, key, fingerprint, exc)
        finally:
            state["runs"][run_id]["updated_at"] = datetime.now().isoformat(timespec="seconds")
            atomic_write_json(state_path, state)

    incomplete = any(state["files"].get(path.relative_to(input_dir).as_posix(), {}).get("status") != "complete" for path in files)
    if not incomplete and failed == 0:
        state.pop("active_run_id", None)
        state["runs"][run_id]["completed_at"] = datetime.now().isoformat(timespec="seconds")
        atomic_write_json(state_path, state)

    print(
        f"Done. Completed: {completed}. Skipped: {skipped}. Failed: {failed}. Output: {run_output_dir}",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        raise
