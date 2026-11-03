#!/bin/bash
# Restart the Waggle Dance Viewer. Kills any dangling instance first.
#
# Usage:
#   ./restart_server.sh                        # defaults below
#   ./restart_server.sh ckpt/other_run/best.pth
#
# Environment overrides:
#   PORT             port to serve on            (default 5050)
#   CHECKPOINT       model .pth to load          (default ckpt/clean_split_v1/best.pth)
#   DEVICE           cuda:0 | cpu | unset=auto
#   EXTERNAL_VIDEOS  directory of unannotated video, skipped if it does not exist

PORT="${PORT:-5050}"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SERVER="$SCRIPT_DIR/viewer/server.py"

CHECKPOINT="${1:-${CHECKPOINT:-$SCRIPT_DIR/ckpt/clean_split_v1/best.pth}}"
EXTERNAL="${EXTERNAL_VIDEOS:-/mnt/horus/bee_dance_videos_tim}"

# The venv usually sits in the repo, but on some machines it is one level up
# (shared between checkouts), so fall back before giving up.
if   [ -x "$SCRIPT_DIR/.venv/bin/python" ];    then VENV="$SCRIPT_DIR/.venv/bin/python"
elif [ -x "$SCRIPT_DIR/../.venv/bin/python" ]; then VENV="$SCRIPT_DIR/../.venv/bin/python"
else
    echo "ERROR: no virtualenv found at $SCRIPT_DIR/.venv or $SCRIPT_DIR/../.venv" >&2
    exit 1
fi

echo "Checking for dangling server instances..."

# 1. Kill anything listening on the port
PIDS_PORT=$(lsof -ti tcp:$PORT 2>/dev/null)
if [ -n "$PIDS_PORT" ]; then
    echo "  Killing PIDs on port $PORT: $PIDS_PORT"
    kill $PIDS_PORT 2>/dev/null
    sleep 1
    # Force kill if still alive
    kill -9 $PIDS_PORT 2>/dev/null
fi

# 2. Kill any python processes running server.py
PIDS_SERVER=$(pgrep -f "python.*viewer/server.py" 2>/dev/null)
if [ -n "$PIDS_SERVER" ]; then
    echo "  Killing server.py PIDs: $PIDS_SERVER"
    kill $PIDS_SERVER 2>/dev/null
    sleep 1
    kill -9 $PIDS_SERVER 2>/dev/null
fi

ARGS=(--host 0.0.0.0 --port "$PORT")

if [ -f "$CHECKPOINT" ]; then
    ARGS+=(--checkpoint "$CHECKPOINT")
    echo "Checkpoint: $CHECKPOINT"
else
    # Not fatal -- the viewer still serves ground truth, just no predictions.
    echo "WARNING: checkpoint not found at $CHECKPOINT"
    echo "         starting without it: predictions and evaluation disabled."
fi

[ -n "${DEVICE:-}" ] && ARGS+=(--device "$DEVICE")

if [ -d "$EXTERNAL" ]; then
    ARGS+=(--external-videos "$EXTERNAL")
else
    echo "Note: external video dir not present ($EXTERNAL), skipping."
fi

echo "Starting server on port $PORT..."
exec "$VENV" "$SERVER" "${ARGS[@]}"
