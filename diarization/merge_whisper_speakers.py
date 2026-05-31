from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from diarization.backend import SpeakerSegment


UNKNOWN = "UNKNOWN"
MULTI_SPEAKER_POSSIBLE = "MULTI_SPEAKER_POSSIBLE"


@dataclass(frozen=True)
class MergeConfig:
    min_overlap_ratio: float = 0.3
    ambiguity_ratio: float = 0.2


def overlap_seconds(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def assign_speakers_to_whisper_segments(
    whisper_segments: list[dict[str, Any]],
    speaker_segments: list[SpeakerSegment],
    config: MergeConfig | None = None,
) -> list[dict[str, Any]]:
    config = config or MergeConfig()
    merged: list[dict[str, Any]] = []

    for segment in whisper_segments:
        start = float(segment.get("start", 0.0))
        end = float(segment.get("end", start))
        duration = max(0.0, end - start)
        matches: list[tuple[SpeakerSegment, float, float]] = []

        for speaker_segment in speaker_segments:
            overlap = overlap_seconds(start, end, speaker_segment.start, speaker_segment.end)
            if overlap <= 0:
                continue
            ratio = overlap / duration if duration > 0 else 0.0
            matches.append((speaker_segment, overlap, ratio))

        matches.sort(key=lambda item: item[1], reverse=True)
        warnings = list(segment.get("warnings", []))
        assigned_speaker = UNKNOWN
        best_overlap = 0.0
        best_ratio = 0.0

        if matches:
            best_segment, best_overlap, best_ratio = matches[0]
            if best_ratio >= config.min_overlap_ratio:
                assigned_speaker = best_segment.speaker_label
            if _has_ambiguous_speaker_overlap(matches, best_segment.speaker_label, config.ambiguity_ratio):
                warnings.append(MULTI_SPEAKER_POSSIBLE)

        merged_segment = dict(segment)
        merged_segment["assigned_speaker"] = assigned_speaker
        merged_segment["overlap_seconds"] = round(best_overlap, 3)
        merged_segment["overlap_ratio"] = round(best_ratio, 3)
        merged_segment["speaker_confidence"] = round(best_ratio, 3)
        merged_segment["warnings"] = warnings
        merged.append(merged_segment)

    return merged


def _has_ambiguous_speaker_overlap(
    matches: list[tuple[SpeakerSegment, float, float]],
    best_label: str,
    ambiguity_ratio: float,
) -> bool:
    seen_other_speaker = False
    for speaker_segment, _overlap, ratio in matches[1:]:
        if speaker_segment.speaker_label != best_label and ratio >= ambiguity_ratio:
            seen_other_speaker = True
            break
    return seen_other_speaker

