"""AutoTrain-compatible training script for vehicle detection.

AutoTrain's agent modifies hyperparameters in this file between iterations.
The last line of stdout must be valid JSON with metric values.
Supports checkpoint resume via AUTOTRAIN_RESUME_FROM env var.
"""

import os
from pathlib import Path

from ultralytics import YOLO

from vtrack.settings import (
    apply_ultralytics_datasets_dir,
    checkpoint_dir,
    resolve_dataset_config,
)


def main():
    apply_ultralytics_datasets_dir()
    data_path = resolve_dataset_config(
        os.environ.get("VTRACK_KITTI_YAML", "kitti.yaml")
    )
    if not Path(data_path).is_file():
        print(f"ERROR: Dataset config not found at {data_path}")
        print('{"mAP": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0}')
        return

    # Check if we should resume from a checkpoint
    resume_from = os.environ.get("AUTOTRAIN_RESUME_FROM")
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        model = YOLO(resume_from)
    else:
        model = YOLO("yolo11n.pt")

    host_checkpoints = checkpoint_dir()
    host_checkpoints.mkdir(parents=True, exist_ok=True)

    # --- Hyperparameters (AutoTrain agent modifies these) ---
    results = model.train(
        data=data_path,
        epochs=10,
        imgsz=640,
        batch=8,
        device="cuda",
        project="outputs/training",
        name="autotrain",
        exist_ok=True,
        pretrained=True,
        lr0=0.01,
        cos_lr=True,
        fliplr=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        verbose=True,
        workers=4,
    )

    weights_dir = Path(getattr(results, "save_dir", "outputs/training/autotrain")) / "weights"
    for filename in ("best.pt", "last.pt"):
        source = weights_dir / filename
        if source.is_file():
            target = host_checkpoints / f"autotrain_{filename}"
            target.write_bytes(source.read_bytes())

    # Print metrics as JSON for AutoTrain extraction
    metrics = results.results_dict
    map50 = metrics.get("metrics/mAP50(B)", 0.0)
    map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
    precision = metrics.get("metrics/precision(B)", 0.0)
    recall = metrics.get("metrics/recall(B)", 0.0)

    print(
        f'{{"mAP": {map50:.4f}, "mAP50_95": {map50_95:.4f}, '
        f'"precision": {precision:.4f}, "recall": {recall:.4f}}}'
    )


if __name__ == "__main__":
    main()
