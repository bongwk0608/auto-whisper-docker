#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/app${PYTHONPATH:+:${PYTHONPATH}}"

echo "==> Starting Whisper CUDA transcription"
python /app/scripts/transcribe.py

echo "==> Whisper completed; starting CUDA diarization"
python /app/scripts/backfill_diarization.py \
  --transcripts-dir /app/output \
  --output-dir /app/output_pyannote \
  --overall-transcripts-dir /app/output_overall \
  --overall-output-dir /app/output_pyannote_overall \
  --state-path /app/state/progress.json \
  --manifest-path /app/state/diarization-progress.json

echo "==> Pipeline completed"
