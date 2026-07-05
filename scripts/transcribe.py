from __future__ import annotations

import csv
import gc
import hashlib
import json
import multiprocessing
import os
import shutil
import subprocess
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

from diarization.filename_normalization import parse_safe_output_policy, safe_relative_path


ALL_FORMATS = ["txt", "json", "tsv", "srt", "vtt"]
DEFAULT_SUPPORTED_EXTENSIONS = (
    ".mp3,.wav,.m4a,.flac,.ogg,.aac,.wma,"
    ".mp4,.m4v,.mov,.mkv,.webm,.avi,.wmv,.flv,.ts,.mts,.m2ts,.3gp,.3g2,.mpg,.mpeg,.vob,.ogv"
)


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
    safe_display_key: str
    fingerprint: dict[str, Any]


class PreparedPair(NamedTuple):
    skipped: int
    skipped_no_audio: int
    pending: list[PendingFile]
    mapping: dict[str, Any] | None


class CudaOutOfMemoryError(RuntimeError):
    pass


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


def parse_oom_fallback(value: str) -> str:
    value = value.lower()
    if value not in {"cpu", "skip", "fail"}:
        raise ValueError("WHISPER_OOM_FALLBACK must be one of: cpu, skip, fail")
    return value


def parse_worker_mode(value: str) -> str:
    value = value.lower()
    if value in {"0", "false", "no", "off"}:
        return "false"
    if value not in {"always", "on_oom", "false"}:
        raise ValueError("WHISPER_WORKER_MODE must be one of: always, on_oom, false")
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


def folder_name_from_path(value: str) -> str:
    normalized = value.strip().replace("\\", "/").rstrip("/")
    if not normalized:
        return ""
    return normalized.rsplit("/", 1)[-1] or ""


def safe_folder_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in "._-" else "_" for ch in value)


def source_display_name(input_dir: Path, host_input_dir: str = "") -> str:
    return folder_name_from_path(host_input_dir) or source_folder_name(input_dir)


def make_run_id(input_dir: Path, display_name: str = "") -> str:
    safe_name = safe_folder_name(display_name or source_folder_name(input_dir))
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


def is_skipped_no_audio(record: dict[str, Any] | None, fingerprint: dict[str, Any], formats: list[str]) -> bool:
    if not record or record.get("status") != "skipped_no_audio":
        return False
    if not fingerprint_matches(record.get("fingerprint"), fingerprint):
        return False
    return sorted(record.get("formats", [])) == sorted(formats)


def is_terminal_record(record: dict[str, Any] | None, fingerprint: dict[str, Any], formats: list[str]) -> bool:
    return is_complete(record, fingerprint, formats) or is_skipped_no_audio(record, fingerprint, formats)


def is_no_audio_error(error: BaseException) -> bool:
    message = str(error).lower()
    patterns = [
        "output file #0 does not contain any stream",
        "does not contain any stream",
        "does not contain an audio stream",
        "no audio stream",
        "audio stream missing",
        "missing audio stream",
        "no audio",
    ]
    return any(pattern in message for pattern in patterns)


def is_cuda_oom_error(error: BaseException) -> bool:
    if isinstance(error, CudaOutOfMemoryError):
        return True
    message = str(error).lower()
    patterns = [
        "cuda out of memory",
        "outofmemoryerror",
        "torch.cuda.outofmemoryerror",
        "cublas_status_alloc_failed",
        "cuda error: out of memory",
        "unable to allocate",
    ]
    return any(pattern in message for pattern in patterns) and "cuda" in message


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


def cleanup_runtime_memory(device: str = "") -> None:
    gc.collect()
    if device == "cuda" and getattr(torch, "cuda", None) is not None:
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass


def query_nvidia_smi_memory() -> tuple[str, int | None] | None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.used,memory.free",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None

    first_line = next((line.strip() for line in completed.stdout.splitlines() if line.strip()), "")
    if not first_line:
        return None
    parts = [part.strip() for part in first_line.split(",")]
    free_mb = None
    if len(parts) >= 4:
        try:
            free_mb = int(parts[3])
        except ValueError:
            free_mb = None
    return first_line, free_mb


def log_resource_preflight(model_name: str, device: str, fp16: bool, strict: bool) -> None:
    print(f"Whisper preflight: model={model_name} device={device} fp16={fp16}", flush=True)
    if device != "cuda":
        return

    gpu_memory = query_nvidia_smi_memory()
    if gpu_memory is None:
        print("Whisper preflight: nvidia-smi is not available inside the container.", flush=True)
        return

    raw_memory, free_mb = gpu_memory
    print(f"Whisper preflight GPU: {raw_memory}", flush=True)
    high_memory_models = {"medium", "large", "large-v1", "large-v2", "large-v3", "large-v3-turbo"}
    base_model = model_name.lower().split(".", 1)[0]
    if base_model in high_memory_models and free_mb is not None and free_mb < 5000:
        message = (
            f"WHISPER_MODEL={model_name} may exceed available CUDA memory "
            f"({free_mb} MiB free). Use WHISPER_MODEL=base for reliable unattended runs."
        )
        if strict:
            raise RuntimeError(message)
        print(f"Warning: {message}", flush=True)


def transcribe_in_process(
    model: Any,
    source_path: Path,
    options: dict[str, Any],
    local_staging: bool,
    local_staging_dir: Path,
) -> dict[str, Any]:
    with transcription_source(source_path, local_staging, local_staging_dir) as source_for_transcription:
        return model.transcribe(str(source_for_transcription), **options)


def transcribe_worker_entry(
    model_name: str,
    device: str,
    source_path: str,
    options: dict[str, Any],
    local_staging: bool,
    local_staging_dir: str,
    result_path: str,
    error_path: str,
) -> None:
    try:
        worker_model = whisper.load_model(model_name, device=device)
        result = transcribe_in_process(
            worker_model,
            Path(source_path),
            options,
            local_staging,
            Path(local_staging_dir),
        )
        atomic_write_json(Path(result_path), result)
    except Exception as exc:
        error_data = {
            "error": str(exc),
            "traceback": traceback.format_exc(),
            "cuda_oom": is_cuda_oom_error(exc),
        }
        atomic_write_json(Path(error_path), error_data)
        raise SystemExit(42 if is_cuda_oom_error(exc) else 1)
    finally:
        cleanup_runtime_memory(device)


def transcribe_in_worker(
    model_name: str,
    device: str,
    source_path: Path,
    options: dict[str, Any],
    local_staging: bool,
    local_staging_dir: Path,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory(prefix="auto-whisper-worker-") as temp_dir:
        temp_dir_path = Path(temp_dir)
        result_path = temp_dir_path / "result.json"
        error_path = temp_dir_path / "error.json"
        process = multiprocessing.Process(
            target=transcribe_worker_entry,
            args=(
                model_name,
                device,
                str(source_path),
                options,
                local_staging,
                str(local_staging_dir),
                str(result_path),
                str(error_path),
            ),
        )
        process.start()
        process.join()

        if process.exitcode == 0 and result_path.exists():
            return json.loads(result_path.read_text(encoding="utf-8"))

        error_data: dict[str, Any] = {}
        if error_path.exists():
            error_data = json.loads(error_path.read_text(encoding="utf-8"))
        message = error_data.get("error") or f"Whisper worker exited with code {process.exitcode}"
        if (
            process.exitcode == 42
            or (process.exitcode is not None and process.exitcode < 0)
            or error_data.get("cuda_oom")
            or is_cuda_oom_error(RuntimeError(message))
        ):
            raise CudaOutOfMemoryError(message)
        raise RuntimeError(message)


def transcribe_with_runtime(
    model: Any,
    model_name: str,
    device: str,
    source_path: Path,
    options: dict[str, Any],
    local_staging: bool,
    local_staging_dir: Path,
    use_worker: bool,
) -> dict[str, Any]:
    if use_worker:
        print(f"Using isolated Whisper worker on {device}.", flush=True)
        return transcribe_in_worker(model_name, device, source_path, options, local_staging, local_staging_dir)
    return transcribe_in_process(model, source_path, options, local_staging, local_staging_dir)


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


def relative_display_path(display_key: str) -> Path:
    return Path(*[part for part in display_key.split("/") if part])


def overall_output_base(overall_output_dir: Path, pair_id: str, display_key: str, source_path: Path) -> Path:
    return timestamped_output_base(overall_output_dir / pair_id / relative_display_path(display_key), timestamp_source=source_path)


def copy_overall_outputs(
    project_outputs: list[Path],
    source_path: Path,
    display_key: str,
    pair_id: str,
    overall_output_enabled: bool,
    overall_output_dir: Path,
    formats: list[str],
) -> list[Path]:
    if not overall_output_enabled:
        return []

    overall_base = overall_output_base(overall_output_dir, pair_id, display_key, source_path)
    overall_outputs = expected_outputs(overall_base, formats)
    for project_output, overall_output in zip(project_outputs, overall_outputs):
        overall_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(project_output, overall_output)
    return overall_outputs


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


def mark_interrupted(
    state: dict[str, Any],
    key: str,
    pair_id: str,
    fingerprint: dict[str, Any],
    fingerprint_mode: str,
    local_staging: bool,
    formats: list[str],
    error: BaseException,
) -> None:
    state["files"][key] = {
        "status": "interrupted",
        "pair_id": pair_id,
        "fingerprint": fingerprint,
        "fingerprint_mode": fingerprint_mode,
        "local_staging": local_staging,
        "formats": formats,
        "reason": "Whisper CUDA transcription ran out of memory before outputs were completed.",
        "error": str(error),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def mark_no_audio_skip(
    state: dict[str, Any],
    key: str,
    pair_id: str,
    fingerprint: dict[str, Any],
    fingerprint_mode: str,
    local_staging: bool,
    formats: list[str],
    error: BaseException,
) -> None:
    state["files"][key] = {
        "status": "skipped_no_audio",
        "pair_id": pair_id,
        "fingerprint": fingerprint,
        "fingerprint_mode": fingerprint_mode,
        "local_staging": local_staging,
        "formats": formats,
        "reason": "No usable audio stream was detected.",
        "error": str(error),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
    }


def host_path_for_pair(values: list[str], index: int) -> str:
    return values[index] if index < len(values) else ""


def build_mapping_record(
    pair: InputOutputPair,
    pair_index: int,
    display_name: str,
    run_id: str,
    run_output_dir: Path,
    files_found: int,
    formats: list[str],
    host_input_dirs: list[str],
    host_output_dirs: list[str],
    fingerprint_mode: str,
    local_staging: bool,
    overall_output_enabled: bool,
    overall_output_dir: Path,
    filename_policy: str,
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
        "source_folder_name": display_name,
        "input_created_timestamp": created,
        "input_modified_timestamp": modified,
        "formats": ",".join(formats),
        "fingerprint_mode": fingerprint_mode,
        "local_staging": local_staging,
        "overall_output_enabled": overall_output_enabled,
        "overall_output_root": str(overall_output_dir) if overall_output_enabled else "",
        "overall_pair_output_dir": str(overall_output_dir / pair.id) if overall_output_enabled else "",
        "filename_policy": filename_policy,
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
        "overall_output_enabled",
        "overall_output_root",
        "overall_pair_output_dir",
        "filename_policy",
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
    overall_output_enabled: bool,
    overall_output_dir: Path,
    filename_policy: str = "auto",
) -> PreparedPair:
    files = scan_files(pair.input_dir, extensions)
    print(f"Found {len(files)} supported file(s) under {pair.input_dir}.", flush=True)
    if not files:
        return PreparedPair(0, 0, [], None)

    active_run_ids = state.setdefault("active_run_ids", {})
    active_run_id = active_run_ids.get(pair.id)
    host_input_dir = host_path_for_pair(host_input_dirs, pair_index)
    display_name = source_display_name(pair.input_dir, host_input_dir)
    if active_run_id and (pair.output_dir / active_run_id).exists():
        run_id = active_run_id
    else:
        run_id = make_run_id(pair.input_dir, display_name)
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
        display_name,
        run_id,
        run_output_dir,
        len(files),
        formats,
        host_input_dirs,
        host_output_dirs,
        fingerprint_mode,
        local_staging,
        overall_output_enabled,
        overall_output_dir,
        filename_policy,
    )

    skipped = 0
    skipped_no_audio = 0
    pending: list[PendingFile] = []
    existing_safe_names: dict[Path, set[str]] = {}

    for index, source_path in enumerate(files, start=1):
        relative = source_path.relative_to(pair.input_dir)
        relative_key = relative.as_posix()
        safe_relative = safe_relative_path(relative, existing_safe_names, filename_policy)
        safe_relative_key = safe_relative.as_posix()
        key = f"{pair.id}:{relative_key}"
        project_media_path = run_output_dir / safe_relative
        fingerprint = file_fingerprint(source_path, fingerprint_mode)

        if is_complete(state["files"].get(key), fingerprint, formats):
            skipped += 1
            print(f"{pair.id} [{index}/{len(files)}] Skipping complete file: {relative_key}", flush=True)
            continue
        if is_skipped_no_audio(state["files"].get(key), fingerprint, formats):
            skipped_no_audio += 1
            print(f"{pair.id} [{index}/{len(files)}] Skipping no-audio file: {relative_key}", flush=True)
            continue

        pending.append(PendingFile(pair, index, len(files), source_path, project_media_path, key, relative_key, safe_relative_key, fingerprint))

    if not pending:
        active_run_ids.pop(pair.id, None)
        state["runs"][run_key]["completed_at"] = datetime.now().isoformat(timespec="seconds")
        print(f"{pair.id} done. Completed: 0. Skipped: {skipped}. Skipped no-audio: {skipped_no_audio}. Failed: 0. Output: {run_output_dir}", flush=True)

    return PreparedPair(skipped, skipped_no_audio, pending, mapping)


def main() -> int:
    state_dir = Path(env("STATE_DIR", "/state"))
    state_path = state_dir / "progress.json"
    model_name = env("WHISPER_MODEL", "small")
    language = env("WHISPER_LANGUAGE", "")
    task = parse_task(env("WHISPER_TASK", "transcribe"))
    condition_on_previous_text = parse_bool(env("WHISPER_CONDITION_ON_PREVIOUS_TEXT", "true"), "WHISPER_CONDITION_ON_PREVIOUS_TEXT")
    verbose = parse_bool(env("WHISPER_VERBOSE", "false"), "WHISPER_VERBOSE")
    oom_fallback = parse_oom_fallback(env("WHISPER_OOM_FALLBACK", "cpu"))
    worker_mode = parse_worker_mode(env("WHISPER_WORKER_MODE", "on_oom"))
    strict_resource_check = parse_bool(env("WHISPER_STRICT_RESOURCE_CHECK", "false"), "WHISPER_STRICT_RESOURCE_CHECK")
    fingerprint_mode = parse_fingerprint_mode(env("FINGERPRINT_MODE", "metadata"))
    local_staging = parse_bool(env("LOCAL_STAGING", "false"), "LOCAL_STAGING")
    local_staging_dir = Path(env("LOCAL_STAGING_DIR", "/tmp/auto-whisper-staging"))
    overall_output_enabled = parse_bool(env("OVERALL_OUTPUT_ENABLED", "true"), "OVERALL_OUTPUT_ENABLED")
    overall_output_dir = Path(env("OVERALL_OUTPUT_DIR", "/overall-output"))
    filename_policy = parse_safe_output_policy(env("SAFE_OUTPUT_FILENAMES", "auto"))
    formats = parse_formats("all")
    extensions = parse_list(env("SUPPORTED_EXTENSIONS", DEFAULT_SUPPORTED_EXTENSIONS))
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
        "oom_fallback": oom_fallback,
        "worker_mode": worker_mode,
        "fingerprint_mode": fingerprint_mode,
        "local_staging": local_staging,
        "local_staging_dir": str(local_staging_dir) if local_staging else "",
        "overall_output_enabled": overall_output_enabled,
        "overall_output_dir": str(overall_output_dir) if overall_output_enabled else "",
        "filename_policy": filename_policy,
    }

    skipped = 0
    skipped_no_audio = 0
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
            overall_output_enabled,
            overall_output_dir,
            filename_policy,
        )
        skipped += prepared.skipped
        skipped_no_audio += prepared.skipped_no_audio
        pending.extend(prepared.pending)
        if prepared.mapping:
            mappings.append(prepared.mapping)

    write_mapping_manifests([pair.output_dir for pair in pairs], mappings)
    atomic_write_json(state_path, state)
    if not pending:
        print(f"Done. Completed: 0. Skipped: {skipped}. Skipped no-audio: {skipped_no_audio}. Failed: 0.", flush=True)
        return 0

    device = choose_device(env("WHISPER_DEVICE", "auto"))
    fp16 = choose_fp16(env("WHISPER_FP16", "auto"), device)
    log_resource_preflight(model_name, device, fp16, strict_resource_check)
    model = None
    if worker_mode != "always":
        print(f"Loading Whisper model '{model_name}' on {device} with fp16={fp16}.", flush=True)
        model = whisper.load_model(model_name, device=device)

    completed = 0
    failed = 0
    use_worker_after_oom = False
    stop_after_failure = False

    for item in pending:

        print(f"{item.pair.id} [{item.index}/{item.total}] Transcribing: {item.display_key}", flush=True)
        try:
            options: dict[str, Any] = {"fp16": fp16}
            if language:
                options["language"] = language
            options["task"] = task
            options["condition_on_previous_text"] = condition_on_previous_text
            options["verbose"] = verbose
            use_worker = worker_mode == "always" or (worker_mode == "on_oom" and use_worker_after_oom)
            try:
                result = transcribe_with_runtime(
                    model,
                    model_name,
                    device,
                    item.source_path,
                    options,
                    local_staging,
                    local_staging_dir,
                    use_worker,
                )
            except Exception as exc:
                if not is_cuda_oom_error(exc):
                    raise
                mark_interrupted(
                    state,
                    item.state_key,
                    item.pair.id,
                    item.fingerprint,
                    fingerprint_mode,
                    local_staging,
                    formats,
                    exc,
                )
                atomic_write_json(state_path, state)
                use_worker_after_oom = True
                cleanup_runtime_memory(device)
                print(f"CUDA OOM during Whisper transcription: {item.pair.id}:{item.display_key}", file=sys.stderr, flush=True)
                if oom_fallback == "fail":
                    raise
                if oom_fallback == "skip":
                    failed += 1
                    mark_failure(state, item.state_key, item.fingerprint, fingerprint_mode, local_staging, exc)
                    continue

                print(f"Retrying Whisper transcription on CPU: {item.pair.id}:{item.display_key}", flush=True)
                cpu_options = dict(options)
                cpu_options["fp16"] = False
                result = transcribe_in_worker(
                    model_name,
                    "cpu",
                    item.source_path,
                    cpu_options,
                    local_staging,
                    local_staging_dir,
                )
            project_outputs = write_outputs(result, item.source_path, item.output_media_path, formats)
            overall_outputs = copy_overall_outputs(
                project_outputs,
                item.source_path,
                item.safe_display_key,
                item.pair.id,
                overall_output_enabled,
                overall_output_dir,
                formats,
            )
            state["files"][item.state_key] = {
                "status": "complete",
                "pair_id": item.pair.id,
                "run_id": state["active_run_ids"][item.pair.id],
                "run_key": state_run_key(item.pair.id, state["active_run_ids"][item.pair.id]),
                "input_dir": str(item.pair.input_dir),
                "output_dir": str(item.pair.output_dir),
                "relative_path": item.display_key,
                "original_relative_path": item.display_key,
                "safe_relative_path": item.safe_display_key,
                "filename_policy": filename_policy,
                "fingerprint": item.fingerprint,
                "fingerprint_mode": fingerprint_mode,
                "local_staging": local_staging,
                "formats": formats,
                "project_outputs": [str(path) for path in project_outputs],
                "overall_outputs": [str(path) for path in overall_outputs],
                "original_output_paths": [],
                "safe_output_paths": [str(path) for path in project_outputs + overall_outputs],
                "completed_at": datetime.now().isoformat(timespec="seconds"),
            }
            completed += 1
        except Exception as exc:
            if is_no_audio_error(exc):
                skipped_no_audio += 1
                print(f"Skipped no-audio: {item.pair.id}:{item.display_key}: {exc}", file=sys.stderr, flush=True)
                mark_no_audio_skip(state, item.state_key, item.pair.id, item.fingerprint, fingerprint_mode, local_staging, formats, exc)
            else:
                failed += 1
                print(f"Failed: {item.pair.id}:{item.display_key}: {exc}", file=sys.stderr, flush=True)
                mark_failure(state, item.state_key, item.fingerprint, fingerprint_mode, local_staging, exc)
                if is_cuda_oom_error(exc) and oom_fallback == "fail":
                    stop_after_failure = True
        finally:
            cleanup_runtime_memory(device)
            run_id = state["active_run_ids"].get(item.pair.id)
            run_key = state_run_key(item.pair.id, run_id) if run_id else ""
            if run_key in state["runs"]:
                state["runs"][run_key]["updated_at"] = datetime.now().isoformat(timespec="seconds")
            atomic_write_json(state_path, state)
        if stop_after_failure:
            break

    for pair in pairs:
        files = scan_files(pair.input_dir, extensions)
        incomplete = any(
            not is_terminal_record(
                state["files"].get(f"{pair.id}:{path.relative_to(pair.input_dir).as_posix()}"),
                file_fingerprint(path, fingerprint_mode),
                formats,
            )
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
        f"Done. Completed: {completed}. Skipped: {skipped}. Skipped no-audio: {skipped_no_audio}. Failed: {failed}.",
        flush=True,
    )
    return 1 if failed else 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"Fatal error: {exc}", file=sys.stderr)
        raise
