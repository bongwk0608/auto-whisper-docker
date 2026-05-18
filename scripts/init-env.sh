#!/usr/bin/env sh
set -eu

SOURCE_DIRS=""
OUTPUT_DIRS=""
MODEL="base"
OUTPUT_FORMAT="all"
CUDA="false"

usage() {
  echo "Usage: $0 --source-dir /path/to/audio-folder --output-dir /path/to/transcripts [--source-dir ... --output-dir ...] [--model base] [--output-format all] [--cuda]"
}

append_value() {
  if [ -z "$1" ]; then
    printf '%s' "$2"
  else
    printf '%s;%s' "$1" "$2"
  fi
}

count_values() {
  if [ -z "$1" ]; then
    echo 0
  else
    awk -F';' '{print NF}' <<EOF
$1
EOF
  fi
}

value_at() {
  INDEX="$1"
  VALUES="$2"
  awk -v index="$INDEX" -F';' '{print $index}' <<EOF
$VALUES
EOF
}

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

yaml_quote() {
  printf '"%s"' "$(json_escape "$1")"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source-dir)
      SOURCE_DIRS="$(append_value "$SOURCE_DIRS" "${2:-}")"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIRS="$(append_value "$OUTPUT_DIRS" "${2:-}")"
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

PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
ENV_PATH="$PROJECT_ROOT/.env"
OVERRIDE_PATH="$PROJECT_ROOT/docker-compose.override.yml"

if [ -z "$SOURCE_DIRS" ]; then
  while :; do
    printf "Enter the full audio/video source folder path: "
    IFS= read -r SOURCE_DIR
    printf "Enter the full transcript output folder path: "
    IFS= read -r OUTPUT_DIR
    SOURCE_DIRS="$(append_value "$SOURCE_DIRS" "$SOURCE_DIR")"
    OUTPUT_DIRS="$(append_value "$OUTPUT_DIRS" "$OUTPUT_DIR")"

    printf "Add another source/output pair? [y/N]: "
    IFS= read -r MORE
    case "$MORE" in
      y|Y|yes|YES) ;;
      *) break ;;
    esac
  done
fi

SOURCE_COUNT="$(count_values "$SOURCE_DIRS")"
OUTPUT_COUNT="$(count_values "$OUTPUT_DIRS")"
if [ "$SOURCE_COUNT" -eq 1 ] && [ "$OUTPUT_COUNT" -eq 0 ]; then
  OUTPUT_DIRS="$PROJECT_ROOT/output"
  OUTPUT_COUNT=1
fi
if [ "$SOURCE_COUNT" -ne "$OUTPUT_COUNT" ]; then
  echo "Source and output folder counts must match." >&2
  exit 1
fi
if [ "$SOURCE_COUNT" -eq 0 ]; then
  echo "At least one source/output pair is required." >&2
  exit 1
fi

RESOLVED_SOURCE_DIRS=""
RESOLVED_OUTPUT_DIRS=""
INDEX=1
while [ "$INDEX" -le "$SOURCE_COUNT" ]; do
  SOURCE_DIR="$(value_at "$INDEX" "$SOURCE_DIRS")"
  OUTPUT_DIR="$(value_at "$INDEX" "$OUTPUT_DIRS")"

  if [ ! -d "$SOURCE_DIR" ]; then
    echo "Source directory does not exist or is not a folder: $SOURCE_DIR" >&2
    exit 1
  fi
  mkdir -p "$OUTPUT_DIR"

  RESOLVED_SOURCE="$(CDPATH= cd -- "$SOURCE_DIR" && pwd)"
  RESOLVED_OUTPUT="$(CDPATH= cd -- "$OUTPUT_DIR" && pwd)"
  RESOLVED_SOURCE_DIRS="$(append_value "$RESOLVED_SOURCE_DIRS" "$RESOLVED_SOURCE")"
  RESOLVED_OUTPUT_DIRS="$(append_value "$RESOLVED_OUTPUT_DIRS" "$RESOLVED_OUTPUT")"
  INDEX=$((INDEX + 1))
done

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

INPUT_OUTPUT_PAIRS="["
INDEX=1
while [ "$INDEX" -le "$SOURCE_COUNT" ]; do
  if [ "$INDEX" -gt 1 ]; then
    INPUT_OUTPUT_PAIRS="${INPUT_OUTPUT_PAIRS},"
  fi
  INPUT_OUTPUT_PAIRS="${INPUT_OUTPUT_PAIRS}{\"input\":\"/inputs/input-$(printf '%03d' "$INDEX")\",\"output\":\"/outputs/output-$(printf '%03d' "$INDEX")\"}"
  INDEX=$((INDEX + 1))
done
INPUT_OUTPUT_PAIRS="${INPUT_OUTPUT_PAIRS}]"

cat > "$ENV_PATH" <<EOF
SOURCE_DIRS=$RESOLVED_SOURCE_DIRS
OUTPUT_DIRS=$RESOLVED_OUTPUT_DIRS
INPUT_OUTPUT_PAIRS=$INPUT_OUTPUT_PAIRS
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

{
  echo "services:"
  echo "  whisper:"
  echo "    volumes:"
  INDEX=1
  while [ "$INDEX" -le "$SOURCE_COUNT" ]; do
    SOURCE_DIR="$(value_at "$INDEX" "$RESOLVED_SOURCE_DIRS")"
    echo "      - type: bind"
    echo "        source: $(yaml_quote "$SOURCE_DIR")"
    echo "        target: /inputs/input-$(printf '%03d' "$INDEX")"
    INDEX=$((INDEX + 1))
  done
  INDEX=1
  while [ "$INDEX" -le "$OUTPUT_COUNT" ]; do
    OUTPUT_DIR="$(value_at "$INDEX" "$RESOLVED_OUTPUT_DIRS")"
    echo "      - type: bind"
    echo "        source: $(yaml_quote "$OUTPUT_DIR")"
    echo "        target: /outputs/output-$(printf '%03d' "$INDEX")"
    INDEX=$((INDEX + 1))
  done
  echo "  whisper-cuda:"
  echo "    volumes:"
  INDEX=1
  while [ "$INDEX" -le "$SOURCE_COUNT" ]; do
    SOURCE_DIR="$(value_at "$INDEX" "$RESOLVED_SOURCE_DIRS")"
    echo "      - type: bind"
    echo "        source: $(yaml_quote "$SOURCE_DIR")"
    echo "        target: /inputs/input-$(printf '%03d' "$INDEX")"
    INDEX=$((INDEX + 1))
  done
  INDEX=1
  while [ "$INDEX" -le "$OUTPUT_COUNT" ]; do
    OUTPUT_DIR="$(value_at "$INDEX" "$RESOLVED_OUTPUT_DIRS")"
    echo "      - type: bind"
    echo "        source: $(yaml_quote "$OUTPUT_DIR")"
    echo "        target: /outputs/output-$(printf '%03d' "$INDEX")"
    INDEX=$((INDEX + 1))
  done
} > "$OVERRIDE_PATH"

echo "Wrote $ENV_PATH"
echo "Wrote $OVERRIDE_PATH"
echo "Configured $SOURCE_COUNT source/output pair(s)"
echo "Selected WHISPER_DEVICE=$DEVICE"
