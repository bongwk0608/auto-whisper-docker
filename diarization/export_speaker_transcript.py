from __future__ import annotations

import csv
import json
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Any

from diarization.backend import DiarizationConfig, SpeakerSegment
from diarization.filename_normalization import parse_safe_output_policy, safe_relative_path, unique_normalized_filename


SPEAKER_FORMATS = ["json", "txt", "srt", "tsv", "vtt"]


def output_base_for_whisper_json(
    whisper_json: Path,
    transcripts_dir: Path,
    output_dir: Path,
    existing_safe_names: dict[Path, set[str]] | None = None,
    filename_policy: str = "auto",
) -> Path:
    try:
        relative = whisper_json.relative_to(transcripts_dir)
    except ValueError:
        relative = whisper_json.resolve().relative_to(transcripts_dir.resolve())
    return output_dir / safe_relative_path(relative, existing_safe_names, filename_policy).with_suffix("")


def speaker_output_paths(base: Path) -> dict[str, Path]:
    return {
        "speaker_json": base.with_suffix(".speaker.json"),
        "speaker_txt": base.with_suffix(".speaker.txt"),
        "speaker_srt": base.with_suffix(".speaker.srt"),
        "speaker_tsv": base.with_suffix(".speaker.tsv"),
        "speaker_vtt": base.with_suffix(".speaker.vtt"),
        "diarization_json": base.with_suffix(".diarization.json"),
    }


def speaker_outputs_complete(base: Path) -> bool:
    paths = speaker_output_paths(base)
    return all(paths[key].is_file() for key in ["speaker_json", "speaker_txt", "speaker_srt", "speaker_tsv", "speaker_vtt", "diarization_json"])


def export_speaker_outputs(
    output_base: Path,
    source_audio: Path,
    whisper_json: Path,
    whisper_data: dict[str, Any],
    speaker_segments: list[SpeakerSegment],
    merged_segments: list[dict[str, Any]],
    config: DiarizationConfig,
    status: str = "completed",
    warnings: list[str] | None = None,
    filename_policy: str = "auto",
) -> dict[str, Path]:
    paths = speaker_output_paths(output_base)
    for path in paths.values():
        path.parent.mkdir(parents=True, exist_ok=True)

    filename_metadata = {
        "audio": unique_normalized_filename(source_audio.name).to_dict(),
        "whisper_json": unique_normalized_filename(whisper_json.name).to_dict(),
    }
    filename_policy = parse_safe_output_policy(filename_policy)
    speaker_labels = sorted({segment.speaker_label for segment in speaker_segments})
    speaker_map = {label: "Unknown" for label in speaker_labels}
    if any(segment.get("assigned_speaker") == "UNKNOWN" for segment in merged_segments):
        speaker_map.setdefault("UNKNOWN", "Unknown")

    payload = {
        "source_whisper_json": str(whisper_json),
        "source_audio": str(source_audio),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "diarization": {
            "backend": config.backend,
            "model": config.model,
            "status": status,
            "num_speakers": config.num_speakers,
            "min_speakers": config.min_speakers,
            "max_speakers": config.max_speakers,
        },
        "speaker_map": speaker_map,
        "speaker_segments": [segment.to_dict() for segment in speaker_segments],
        "segments": merged_segments,
        "text": whisper_data.get("text", ""),
        "warnings": warnings or [],
        "filename_normalization": filename_metadata,
        "filename_policy": filename_policy,
    }
    raw_payload = {
        "source_audio": str(source_audio),
        "generated_at": payload["generated_at"],
        "diarization": payload["diarization"],
        "speaker_segments": payload["speaker_segments"],
        "filename_normalization": filename_metadata,
        "filename_policy": filename_policy,
    }

    _atomic_write_text(paths["speaker_json"], json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    _atomic_write_text(paths["diarization_json"], json.dumps(raw_payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n")
    _atomic_write_text(paths["speaker_txt"], render_txt(source_audio, whisper_json, config, status, speaker_map, merged_segments))
    _atomic_write_text(paths["speaker_srt"], render_srt(merged_segments))
    _atomic_write_text(paths["speaker_vtt"], render_vtt(merged_segments))
    write_tsv(paths["speaker_tsv"], merged_segments)
    return paths


def render_txt(
    source_audio: Path,
    whisper_json: Path,
    config: DiarizationConfig,
    status: str,
    speaker_map: dict[str, str],
    segments: list[dict[str, Any]],
) -> str:
    lines = [
        "# Speaker Transcript",
        "",
        f"Source audio: `{source_audio.name}`",
        f"Whisper transcript: `{whisper_json.name}`",
        f"Diarization backend: `{config.backend}`",
        f"Status: `{status}`",
        "",
        "## Speaker Map",
        "",
    ]
    for label, name in sorted(speaker_map.items()):
        lines.append(f"- {label}: {name}")
    lines.extend(["", "## Transcript", ""])
    for segment in segments:
        lines.append(f"[{format_timestamp(float(segment.get('start', 0.0)))} - {format_timestamp(float(segment.get('end', 0.0)))}] {segment.get('assigned_speaker', 'UNKNOWN')}:")
        lines.append(str(segment.get("text", "")).strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def render_srt(segments: list[dict[str, Any]]) -> str:
    blocks = []
    for index, segment in enumerate(segments, start=1):
        start = format_srt_timestamp(float(segment.get("start", 0.0)))
        end = format_srt_timestamp(float(segment.get("end", 0.0)))
        speaker = segment.get("assigned_speaker", "UNKNOWN")
        text = str(segment.get("text", "")).strip()
        blocks.append(f"{index}\n{start} --> {end}\n{speaker}: {text}")
    return "\n\n".join(blocks).rstrip() + "\n"


def render_vtt(segments: list[dict[str, Any]]) -> str:
    blocks = ["WEBVTT", ""]
    for segment in segments:
        start = format_vtt_timestamp(float(segment.get("start", 0.0)))
        end = format_vtt_timestamp(float(segment.get("end", 0.0)))
        speaker = segment.get("assigned_speaker", "UNKNOWN")
        text = str(segment.get("text", "")).strip()
        blocks.append(f"{start} --> {end}\n{speaker}: {text}\n")
    return "\n".join(blocks).rstrip() + "\n"


def write_tsv(path: Path, segments: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=["start", "end", "speaker", "text"], delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for segment in segments:
            writer.writerow(
                {
                    "start": f"{float(segment.get('start', 0.0)):.3f}",
                    "end": f"{float(segment.get('end', 0.0)):.3f}",
                    "speaker": segment.get("assigned_speaker", "UNKNOWN"),
                    "text": str(segment.get("text", "")).strip(),
                }
            )
        temp_name = handle.name
    Path(temp_name).replace(path)


def format_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours, rem = divmod(millis, 3_600_000)
    minutes, rem = divmod(rem, 60_000)
    secs, ms = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{ms:03d}"


def format_srt_timestamp(seconds: float) -> str:
    return format_timestamp(seconds).replace(".", ",")


def format_vtt_timestamp(seconds: float) -> str:
    return format_timestamp(seconds)


def _atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temp_name = handle.name
    Path(temp_name).replace(path)
