#!/usr/bin/env sh
set -eu

DRIVES="${1:-X Y Z}"
CHECK_TIMEOUT_SECONDS="${CHECK_TIMEOUT_SECONDS:-5}"

if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
else
  SUDO="sudo"
fi

usage() {
  echo "Usage: $0 [\"X Y Z\"]"
  echo "Mounts Windows mapped network drives into WSL at /mnt/<drive>."
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if command -v net.exe >/dev/null 2>&1; then
  NET_EXE="$(command -v net.exe)"
elif [ -x /mnt/c/Windows/System32/net.exe ]; then
  NET_EXE="/mnt/c/Windows/System32/net.exe"
else
  echo "net.exe is not available. Run this script from WSL on Windows." >&2
  exit 1
fi

for DRIVE in $DRIVES; do
  LETTER="$(printf '%s' "$DRIVE" | tr '[:lower:]' '[:upper:]' | sed 's/://g')"
  case "$LETTER" in
    [A-Z]) ;;
    *)
      echo "Skipping invalid drive letter: $DRIVE" >&2
      continue
      ;;
  esac

  UNC_PATH="$("$NET_EXE" use "${LETTER}:" 2>/dev/null | tr -d '\r' | awk '
    /^Remote name[[:space:]]+/ {
      sub(/^Remote name[[:space:]]+/, "")
      print
      exit
    }
  ')"

  if [ -z "$UNC_PATH" ]; then
    echo "${LETTER}: is not currently mapped in Windows. Skipping."
    continue
  fi

  MOUNT_POINT="/mnt/$(printf '%s' "$LETTER" | tr '[:upper:]' '[:lower:]')"

  if mountpoint -q "$MOUNT_POINT"; then
    if timeout "$CHECK_TIMEOUT_SECONDS" ls "$MOUNT_POINT" >/dev/null 2>&1; then
      echo "$MOUNT_POINT is already mounted and reachable."
      continue
    fi

    echo "$MOUNT_POINT is mounted but not reachable. Remounting."
    $SUDO umount -lf "$MOUNT_POINT" 2>/dev/null || true
  fi

  echo "Mounting ${LETTER}: ($UNC_PATH) at $MOUNT_POINT"
  $SUDO mkdir -p "$MOUNT_POINT"
  $SUDO mount -t drvfs "$UNC_PATH" "$MOUNT_POINT"
done
