#!/usr/bin/env sh
set -eu

DRIVES="${1:-X Y Z}"
PROJECT_ROOT="$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)"
MOUNT_SCRIPT="$PROJECT_ROOT/scripts/mount-wsl-network-drives.sh"
SERVICE_PATH="/etc/systemd/system/auto-whisper-network-drives.service"
TIMER_PATH="/etc/systemd/system/auto-whisper-network-drives.timer"

usage() {
  echo "Usage: $0 [\"X Y Z\"]"
  echo "Installs a WSL systemd timer that mounts mapped Windows network drives at boot and retries every minute."
}

if [ "${1:-}" = "-h" ] || [ "${1:-}" = "--help" ]; then
  usage
  exit 0
fi

if [ ! -d /run/systemd/system ]; then
  cat >&2 <<EOF
systemd is not running in this WSL distro.

Enable it by creating or editing /etc/wsl.conf:

[boot]
systemd=true

Then from Windows PowerShell run:

wsl --shutdown

Start WSL again and rerun this installer.
EOF
  exit 1
fi

if [ ! -f "$MOUNT_SCRIPT" ]; then
  echo "Mount script not found: $MOUNT_SCRIPT" >&2
  exit 1
fi

sudo tee "$SERVICE_PATH" >/dev/null <<EOF
[Unit]
Description=Mount Auto Whisper Windows network drives in WSL
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
ExecStart=/bin/sh "$MOUNT_SCRIPT" "$DRIVES"
EOF

sudo tee "$TIMER_PATH" >/dev/null <<EOF
[Unit]
Description=Retry Auto Whisper Windows network drive mounts

[Timer]
OnBootSec=20s
OnUnitActiveSec=60s
AccuracySec=10s
Persistent=true
Unit=auto-whisper-network-drives.service

[Install]
WantedBy=timers.target
EOF

sudo systemctl daemon-reload
sudo systemctl enable --now auto-whisper-network-drives.timer
sudo systemctl start auto-whisper-network-drives.service || true

echo "Installed auto-whisper-network-drives.timer for drives: $DRIVES"
echo "Check status with:"
echo "  systemctl status auto-whisper-network-drives.timer"
echo "  systemctl status auto-whisper-network-drives.service"
echo "View logs with:"
echo "  journalctl -u auto-whisper-network-drives.service -n 50 --no-pager"
