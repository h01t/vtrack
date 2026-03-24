#!/bin/bash
# Remote training on CUDA workstation (blackbox)
# Usage: ./scripts/train_remote.sh [extra args for train.py]
# Example: ./scripts/train_remote.sh --epochs 100 --batch 32

set -e

REMOTE="blackbox"
REMOTE_DIR="~/object-det"
REMOTE_VENV="source ~/object-det/.venv/bin/activate"

echo "=== Syncing project files to $REMOTE ==="
rsync -avz --exclude '.venv' --exclude 'data' --exclude 'models' \
    --exclude 'outputs' --exclude 'runs' --exclude '__pycache__' \
    --exclude '.git' --exclude '*.jpg' --exclude '*.png' \
    "$(dirname "$0")/../" "$REMOTE:$REMOTE_DIR/"

echo ""
echo "=== Starting training on $REMOTE (CUDA) ==="
ssh "$REMOTE" "$REMOTE_VENV && cd $REMOTE_DIR && python3 scripts/train.py \
    --device cuda \
    --data kitti.yaml \
    --epochs 50 \
    --batch 16 \
    --name vehicle_v1 \
    $*"

echo ""
echo "=== Pulling trained weights back ==="
mkdir -p "$(dirname "$0")/../models"
rsync -avz "$REMOTE:$REMOTE_DIR/runs/detect/outputs/training/vehicle_v1/weights/" \
    "$(dirname "$0")/../models/"

echo ""
echo "=== Done! Weights saved to models/ ==="
ls -la "$(dirname "$0")/../models/"
