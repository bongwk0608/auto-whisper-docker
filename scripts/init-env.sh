#!/usr/bin/env sh
set -eu

SOURCE_DIR=""
MODEL="base"
OUTPUT_FORMAT="all"
CUDA="false"

usage() {
  echo "Usage: $0 --source-dir /path/to/audio-folder [--model base] [--output-format all] [--cuda]"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source-dir)
      SOURCE_DIR="${2:-}"
      shift 2
      ;;
    --model)
      MODEL="${2:-}"
      shift 2
      ;;
    --output-format)
      OUTPUT_FORMAT="${2:-}"
      shift 2
      ;;
    --cuda)
      CUDA="true"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [ -z "$SOURCE_DIR" ]; then
  printf "Enter the full audio/video source folder path: "
  IFS= read -r SOURCE_DIR
fi

if [ ! -d "$SOURCE_DIR" ]; then
  echo "Source directory does not exist or is not a folder: $SOURCE_DIR" >&2
  exit 1
fi

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
ENV_PATH="$PROJECT_ROOT/.env"
RESOLVED_SOURCE_DIR="$(CDPATH= cd -- "$SOURCE_DIR" && pwd)"
DEVICE="cpu"
FP16="false"
NVIDIA_VISIBLE_DEVICES="void"

if [ "$CUDA" = "true" ]; then
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "CUDA was requested, but nvidia-smi is not available. Run without --cuda for CPU mode." >&2
    exit 1
  fi
  if ! docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi >/dev/null 2>&1; then
    echo "CUDA was requested, but Docker GPU access is not working. Run without --cuda for CPU mode." >&2
    exit 1
  fi
  DEVICE="auto"
  FP16="auto"
  NVIDIA_VISIBLE_DEVICES="all"
fi

cat > "$ENV_PATH" <<EOF
SOURCE_DIR=$RESOLVED_SOURCE_DIR
WHISPER_MODEL=$MODEL
WHISPER_OUTPUT_FORMAT=$OUTPUT_FORMAT
WHISPER_LANGUAGE=
WHISPER_TASK=transcribe
WHISPER_DEVICE=$DEVICE
WHISPER_DOWNLOAD_DEVICE=cpu
WHISPER_FP16=$FP16
WHISPER_CONDITION_ON_PREVIOUS_TEXT=true
WHISPER_VERBOSE=false
SUPPORTED_EXTENSIONS=.mp3,.wav,.m4a,.mp4,.mov,.mkv,.webm,.flac,.ogg,.aac,.wma
NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES
NVIDIA_DRIVER_CAPABILITIES=compute,utility
EOF

echo "Wrote $ENV_PATH"
echo "Selected WHISPER_DEVICE=$DEVICE"
