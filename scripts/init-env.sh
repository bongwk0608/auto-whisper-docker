#!/usr/bin/env sh
set -eu

SOURCE_DIRS=""
OUTPUT_DIRS=""
SOURCE_LIST_FILE=""
MODEL="base"
OUTPUT_FORMAT="all"
CUDA="false"

usage() {
  echo "Usage: $0 [--source-list-file ./input-folders.txt] [--source-dir /path/to/audio-folder ...] [--output-dir /path/to/transcripts ...] [--model base] [--output-format all] [--cuda]"
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
  awk -v field_index="$INDEX" -F';' '{print $field_index}' <<EOF
$VALUES
EOF
}

json_escape() {
  printf '%s' "$1" | sed 's/\\/\\\\/g; s/"/\\"/g'
}

yaml_quote() {
  printf '"%s"' "$(json_escape "$1")"
}

normalize_host_path() {
  RAW_PATH="$1"
  case "$RAW_PATH" in
    [A-Za-z]:*)
      DRIVE="$(printf '%s' "$RAW_PATH" | cut -c 1 | tr '[:upper:]' '[:lower:]')"
      REST="$(printf '%s' "$RAW_PATH" | cut -c 3- | sed 's#\\#/#g')"
      case "$REST" in
        "") printf '/mnt/%s' "$DRIVE" ;;
        /*) printf '/mnt/%s%s' "$DRIVE" "$REST" ;;
        *) printf '/mnt/%s/%s' "$DRIVE" "$REST" ;;
      esac
      ;;
    *)
      printf '%s' "$RAW_PATH"
      ;;
  esac
}

read_source_list_file() {
  LIST_PATH="$1"
  if [ ! -f "$LIST_PATH" ]; then
    echo "Source list file does not exist: $LIST_PATH" >&2
    exit 1
  fi

  while IFS= read -r LINE || [ -n "$LINE" ]; do
    TRIMMED="$(printf '%s' "$LINE" | sed 's/^[[:space:]]*//; s/[[:space:]]*$//')"
    case "$TRIMMED" in
      ""|\#*) ;;
      *) SOURCE_DIRS="$(append_value "$SOURCE_DIRS" "$TRIMMED")" ;;
    esac
  done < "$LIST_PATH"
}

while [ "$#" -gt 0 ]; do
  case "$1" in
    --source-list-file)
      SOURCE_LIST_FILE="${2:-}"
      shift 2
      ;;
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
OVERALL_OUTPUT_PATH="$PROJECT_ROOT/output_overall"

if [ -n "$SOURCE_LIST_FILE" ]; then
  read_source_list_file "$SOURCE_LIST_FILE"
fi

if [ -z "$SOURCE_DIRS" ]; then
  while :; do
    printf "Enter the full audio/video source folder path: "
    IFS= read -r SOURCE_DIR
    SOURCE_DIRS="$(append_value "$SOURCE_DIRS" "$SOURCE_DIR")"

    printf "Add another source folder? [y/N]: "
    IFS= read -r MORE
    case "$MORE" in
      y|Y|yes|YES) ;;
      *) break ;;
    esac
  done
fi

SOURCE_COUNT="$(count_values "$SOURCE_DIRS")"
OUTPUT_COUNT="$(count_values "$OUTPUT_DIRS")"
if [ "$OUTPUT_COUNT" -eq 0 ]; then
  INDEX=1
  while [ "$INDEX" -le "$SOURCE_COUNT" ]; do
    OUTPUT_DIRS="$(append_value "$OUTPUT_DIRS" "$PROJECT_ROOT/output")"
    INDEX=$((INDEX + 1))
  done
  OUTPUT_COUNT="$SOURCE_COUNT"
fi
if [ "$SOURCE_COUNT" -ne "$OUTPUT_COUNT" ]; then
  echo "Output folders are optional, but if provided their count must match source folders." >&2
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
  SOURCE_DIR="$(normalize_host_path "$(value_at "$INDEX" "$SOURCE_DIRS")")"
  OUTPUT_DIR="$(normalize_host_path "$(value_at "$INDEX" "$OUTPUT_DIRS")")"

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

mkdir -p "$OVERALL_OUTPUT_PATH"

existing_env_value() {
  KEY="$1"
  DEFAULT_VALUE="$2"
  if [ -f "$ENV_PATH" ]; then
    LINE="$(grep -m 1 "^${KEY}=" "$ENV_PATH" || true)"
    if [ -n "$LINE" ]; then
      printf '%s' "${LINE#*=}"
      return
    fi
  fi
  printf '%s' "$DEFAULT_VALUE"
}

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

PYANNOTE_AUTH_TOKEN="$(existing_env_value PYANNOTE_AUTH_TOKEN "")"
WHISPER_LANGUAGE="$(existing_env_value WHISPER_LANGUAGE "")"
WHISPER_TASK="$(existing_env_value WHISPER_TASK "transcribe")"
WHISPER_CONDITION_ON_PREVIOUS_TEXT="$(existing_env_value WHISPER_CONDITION_ON_PREVIOUS_TEXT "false")"
WHISPER_VERBOSE="$(existing_env_value WHISPER_VERBOSE "true")"
FINGERPRINT_MODE="$(existing_env_value FINGERPRINT_MODE "metadata")"
LOCAL_STAGING="$(existing_env_value LOCAL_STAGING "false")"
LOCAL_STAGING_DIR="$(existing_env_value LOCAL_STAGING_DIR "/tmp/auto-whisper-staging")"
OVERALL_OUTPUT_ENABLED="$(existing_env_value OVERALL_OUTPUT_ENABLED "true")"
OVERALL_OUTPUT_DIR="$(existing_env_value OVERALL_OUTPUT_DIR "/overall-output")"
SAFE_OUTPUT_FILENAMES="$(existing_env_value SAFE_OUTPUT_FILENAMES "auto")"
SUPPORTED_EXTENSIONS="$(existing_env_value SUPPORTED_EXTENSIONS ".mp3,.wav,.m4a,.flac,.ogg,.aac,.wma,.mp4,.m4v,.mov,.mkv,.webm,.avi,.wmv,.flv,.ts,.mts,.m2ts,.3gp,.3g2,.mpg,.mpeg,.vob,.ogv")"
PYANNOTE_METRICS_ENABLED="$(existing_env_value PYANNOTE_METRICS_ENABLED "0")"
DIARIZATION_BACKEND="$(existing_env_value DIARIZATION_BACKEND "pyannote")"
DIARIZATION_MODEL="$(existing_env_value DIARIZATION_MODEL "pyannote/speaker-diarization-community-1")"
DIARIZATION_VERBOSE="$(existing_env_value DIARIZATION_VERBOSE "false")"
DIARIZATION_PROGRESS="$(existing_env_value DIARIZATION_PROGRESS "")"
DIARIZATION_TF32="$(existing_env_value DIARIZATION_TF32 "false")"
DIARIZATION_OOM_FALLBACK="$(existing_env_value DIARIZATION_OOM_FALLBACK "cpu")"
DIARIZATION_CUDA_QUARANTINE_AFTER_OOM="$(existing_env_value DIARIZATION_CUDA_QUARANTINE_AFTER_OOM "false")"
DIARIZATION_CUDA_DEBUG_ERRORS="$(existing_env_value DIARIZATION_CUDA_DEBUG_ERRORS "false")"
DIARIZATION_WORKER_MODE="$(existing_env_value DIARIZATION_WORKER_MODE "always")"
DIARIZATION_GPU_MEMORY_LOG="$(existing_env_value DIARIZATION_GPU_MEMORY_LOG "false")"
DIARIZATION_WORKER_TIMEOUT_SECONDS="$(existing_env_value DIARIZATION_WORKER_TIMEOUT_SECONDS "7200")"
DIARIZATION_GPU_MEMORY_WAIT_SECONDS="$(existing_env_value DIARIZATION_GPU_MEMORY_WAIT_SECONDS "0")"
DIARIZATION_AUDIO_PREPROCESS="$(existing_env_value DIARIZATION_AUDIO_PREPROCESS "always")"
DIARIZATION_AUDIO_PREPROCESS_DIR="$(existing_env_value DIARIZATION_AUDIO_PREPROCESS_DIR "/tmp/auto-whisper-diarization")"
DIARIZATION_MIN_OVERLAP_RATIO="$(existing_env_value DIARIZATION_MIN_OVERLAP_RATIO "0.3")"
DIARIZATION_NUM_SPEAKERS="$(existing_env_value DIARIZATION_NUM_SPEAKERS "")"
DIARIZATION_MIN_SPEAKERS="$(existing_env_value DIARIZATION_MIN_SPEAKERS "")"
DIARIZATION_MAX_SPEAKERS="$(existing_env_value DIARIZATION_MAX_SPEAKERS "")"
DIARIZATION_OUTPUT_DIR="$(existing_env_value DIARIZATION_OUTPUT_DIR "/app/output_pyannote")"
DIARIZATION_OVERALL_OUTPUT_DIR="$(existing_env_value DIARIZATION_OVERALL_OUTPUT_DIR "/app/output_pyannote_overall")"
DIARIZATION_CACHE_DIR="$(existing_env_value DIARIZATION_CACHE_DIR "/app/state/diarization-cache")"
HF_HOME="$(existing_env_value HF_HOME "/cache/huggingface")"
TORCH_HOME="$(existing_env_value TORCH_HOME "/cache/torch")"

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
WHISPER_LANGUAGE=$WHISPER_LANGUAGE
WHISPER_TASK=$WHISPER_TASK
WHISPER_DEVICE=$DEVICE
WHISPER_DOWNLOAD_DEVICE=cpu
WHISPER_FP16=$FP16
WHISPER_CONDITION_ON_PREVIOUS_TEXT=$WHISPER_CONDITION_ON_PREVIOUS_TEXT
WHISPER_VERBOSE=$WHISPER_VERBOSE
FINGERPRINT_MODE=$FINGERPRINT_MODE
LOCAL_STAGING=$LOCAL_STAGING
LOCAL_STAGING_DIR=$LOCAL_STAGING_DIR
OVERALL_OUTPUT_ENABLED=$OVERALL_OUTPUT_ENABLED
OVERALL_OUTPUT_DIR=$OVERALL_OUTPUT_DIR
SAFE_OUTPUT_FILENAMES=$SAFE_OUTPUT_FILENAMES
SUPPORTED_EXTENSIONS=$SUPPORTED_EXTENSIONS
NVIDIA_VISIBLE_DEVICES=$NVIDIA_VISIBLE_DEVICES
NVIDIA_DRIVER_CAPABILITIES=compute,utility
PYANNOTE_AUTH_TOKEN=$PYANNOTE_AUTH_TOKEN
PYANNOTE_METRICS_ENABLED=$PYANNOTE_METRICS_ENABLED
DIARIZATION_BACKEND=$DIARIZATION_BACKEND
DIARIZATION_MODEL=$DIARIZATION_MODEL
DIARIZATION_VERBOSE=$DIARIZATION_VERBOSE
DIARIZATION_PROGRESS=$DIARIZATION_PROGRESS
DIARIZATION_TF32=$DIARIZATION_TF32
DIARIZATION_OOM_FALLBACK=$DIARIZATION_OOM_FALLBACK
DIARIZATION_CUDA_QUARANTINE_AFTER_OOM=$DIARIZATION_CUDA_QUARANTINE_AFTER_OOM
DIARIZATION_CUDA_DEBUG_ERRORS=$DIARIZATION_CUDA_DEBUG_ERRORS
DIARIZATION_WORKER_MODE=$DIARIZATION_WORKER_MODE
DIARIZATION_GPU_MEMORY_LOG=$DIARIZATION_GPU_MEMORY_LOG
DIARIZATION_WORKER_TIMEOUT_SECONDS=$DIARIZATION_WORKER_TIMEOUT_SECONDS
DIARIZATION_GPU_MEMORY_WAIT_SECONDS=$DIARIZATION_GPU_MEMORY_WAIT_SECONDS
DIARIZATION_AUDIO_PREPROCESS=$DIARIZATION_AUDIO_PREPROCESS
DIARIZATION_AUDIO_PREPROCESS_DIR=$DIARIZATION_AUDIO_PREPROCESS_DIR
DIARIZATION_MIN_OVERLAP_RATIO=$DIARIZATION_MIN_OVERLAP_RATIO
DIARIZATION_NUM_SPEAKERS=$DIARIZATION_NUM_SPEAKERS
DIARIZATION_MIN_SPEAKERS=$DIARIZATION_MIN_SPEAKERS
DIARIZATION_MAX_SPEAKERS=$DIARIZATION_MAX_SPEAKERS
DIARIZATION_OUTPUT_DIR=$DIARIZATION_OUTPUT_DIR
DIARIZATION_OVERALL_OUTPUT_DIR=$DIARIZATION_OVERALL_OUTPUT_DIR
DIARIZATION_CACHE_DIR=$DIARIZATION_CACHE_DIR
HF_HOME=$HF_HOME
TORCH_HOME=$TORCH_HOME
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
  echo "  pipeline-cuda:"
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
if [ "$CUDA" = "true" ] && [ -z "$PYANNOTE_AUTH_TOKEN" ]; then
  echo "PYANNOTE_AUTH_TOKEN is blank. Set it in .env before running pipeline-cuda or diarization."
fi
