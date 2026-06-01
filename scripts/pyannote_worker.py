from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from diarization.backend import DiarizationConfig
from diarization.pyannote_runner import PyannoteDiarizationBackend, is_cuda_oom_error, parse_tf32_mode


def parse_optional_int(value: str | None) -> int | None:
    if value is None or value == "":
        return None
    return int(value)


def write_payload(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Internal one-file Pyannote diarization worker.")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--cache-out", required=True, type=Path)
    parser.add_argument("--device", choices=["cuda", "cpu"], required=True)
    parser.add_argument("--backend", default="pyannote")
    parser.add_argument("--model", default="pyannote/speaker-diarization-community-1")
    parser.add_argument("--num-speakers", type=int, default=None)
    parser.add_argument("--min-speakers", type=int, default=None)
    parser.add_argument("--max-speakers", type=int, default=None)
    parser.add_argument("--tf32", choices=["auto", "true", "false"], default=parse_tf32_mode(None))
    parser.add_argument("--verbose", action="store_true")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        config = DiarizationConfig(
            args.backend,
            args.model,
            args.num_speakers,
            args.min_speakers,
            args.max_speakers,
        )
        backend = PyannoteDiarizationBackend(
            config,
            device=args.device,
            tf32_mode=args.tf32,
            verbose=args.verbose,
        )
        try:
            segments = backend.diarize(args.audio)
        finally:
            del backend
        write_payload(
            args.cache_out,
            {
                "status": "completed",
                "speaker_segments": [segment.to_dict() for segment in segments],
            },
        )
        return 0
    except Exception as exc:
        write_payload(
            args.cache_out,
            {
                "status": "failed",
                "error_message": str(exc),
                "is_cuda_oom": is_cuda_oom_error(exc),
                "traceback": traceback.format_exc(),
            },
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
