#!/bin/bash
# Train or resume a waggle detection model.
# Usage:
#   ./train.sh                          # fresh run with auto-generated name
#   ./train.sh clean_split_v1           # fresh run with custom name
#   ./train.sh clean_split_v1 --resume  # resume from latest checkpoint
#
# Logs to ckpt/<run_name>/training.log (via tee, so you see it live).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
VENV="$SCRIPT_DIR/.venv/bin/python"
CONFIG="$SCRIPT_DIR/configs/config.yaml"

RUN_NAME="${1:-run_$(date +%Y%m%d-%H%M%S)}"
CKPT_DIR="$SCRIPT_DIR/ckpt/$RUN_NAME"
LOGFILE="$CKPT_DIR/training.log"

# Build command
CMD="$VENV $SCRIPT_DIR/main.py --config_path $CONFIG --run_name $RUN_NAME"

if [[ "${2:-}" == "--resume" ]]; then
    LATEST="$CKPT_DIR/latest.pth"
    if [[ ! -f "$LATEST" ]]; then
        echo "ERROR: No checkpoint found at $LATEST"
        exit 1
    fi
    CMD="$CMD --resume $LATEST --reset_scheduler"
    echo "Resuming from $LATEST"
elif [[ "${2:-}" == "--resume-best" ]]; then
    BEST="$CKPT_DIR/best.pth"
    if [[ ! -f "$BEST" ]]; then
        echo "ERROR: No checkpoint found at $BEST"
        exit 1
    fi
    CMD="$CMD --resume $BEST --reset_scheduler"
    echo "Resuming from $BEST"
else
    echo "Starting fresh run: $RUN_NAME"
fi

mkdir -p "$CKPT_DIR"

# Copy config for reproducibility
cp "$CONFIG" "$CKPT_DIR/config.yaml"

echo "Run name:   $RUN_NAME"
echo "Checkpoint: $CKPT_DIR"
echo "Log file:   $LOGFILE"
echo "Command:    $CMD"
echo "---"

# Run with tee so output goes to both terminal and log file
exec $CMD 2>&1 | tee -a "$LOGFILE"
