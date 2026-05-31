#!/usr/bin/env sh
set -eu

USE_CUDA=false
RUN_DIARIZATION=false
DIARIZATION_DRY_RUN=false
DIARIZATION_FORCE=false

while [ "$#" -gt 0 ]; do
  case "$1" in
    --cuda)
      USE_CUDA=true
      ;;
    --diarization)
      RUN_DIARIZATION=true
      ;;
    --diarization-dry-run)
      RUN_DIARIZATION=true
      DIARIZATION_DRY_RUN=true
      ;;
    --diarization-force)
      RUN_DIARIZATION=true
      DIARIZATION_FORCE=true
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 2
      ;;
  esac
  shift
done

echo "Running Whisper transcription..."
if [ "$USE_CUDA" = true ]; then
  docker compose --profile cuda run --rm whisper-cuda
else
  docker compose run --rm whisper
fi

if [ "$RUN_DIARIZATION" != true ]; then
  echo "Whisper complete. Diarization was not requested."
  exit 0
fi

echo "Running Pyannote diarization..."
set -- docker compose --profile diarization run --rm diarization-cuda python scripts/backfill_diarization.py
if [ "$DIARIZATION_DRY_RUN" = true ]; then
  set -- "$@" --dry-run
fi
if [ "$DIARIZATION_FORCE" = true ]; then
  set -- "$@" --force
fi
exec "$@"

