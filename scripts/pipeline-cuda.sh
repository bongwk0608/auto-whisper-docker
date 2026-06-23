#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH="/app${PYTHONPATH:+:${PYTHONPATH}}"

if [ -z "${PYANNOTE_AUTH_TOKEN:-}" ]; then
  echo "Fatal error: PYANNOTE_AUTH_TOKEN is required before running pipeline-cuda." >&2
  echo "Add PYANNOTE_AUTH_TOKEN to .env, then rerun the setup helper so Compose picks it up." >&2
  echo "Example: sh ./scripts/init-env.sh --source-list-file ./input-folders.txt --cuda --model medium --output-format all" >&2
  exit 2
fi

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
