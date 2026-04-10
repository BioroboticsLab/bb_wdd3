#!/bin/bash
# Restart the Waggle Dance Annotation Viewer server on port 5050.
# Kills any dangling instances first.

PORT=5050
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv/bin/python"
SERVER="$SCRIPT_DIR/viewer/server.py"
CHECKPOINT="$SCRIPT_DIR/ckpt/noobj_rebalance_v2/best.pth"

# Kill any existing server processes on this port or matching server.py
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

EXTERNAL="/mnt/horus/bee_dance_videos_tim"

echo "Starting server on port $PORT..."
exec "$VENV" "$SERVER" --host 0.0.0.0 --port $PORT --checkpoint "$CHECKPOINT" --external-videos "$EXTERNAL"
