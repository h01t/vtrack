"""AutoTrain-compatible training script for vehicle detection.

AutoTrain's agent modifies hyperparameters in this file between iterations.
The last line of stdout must be valid JSON with metric values.
Supports checkpoint resume via AUTOTRAIN_RESUME_FROM env var.
"""

import os

from ultralytics import YOLO


def main():
    # Check if we should resume from a checkpoint
    resume_from = os.environ.get("AUTOTRAIN_RESUME_FROM")
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        model = YOLO(resume_from)
    else:
        model = YOLO("yolo11s.pt")

    # --- Hyperparameters (AutoTrain agent modifies these) ---
    results = model.train(
        data="datasets/kitti/kitti.yaml",
        epochs=50,
        imgsz=640,
        batch=16,
        device="cuda",
        project="outputs/training",
        name="autotrain",
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        lr0=0.001,
        patience=10,
        save=True,
        plots=True,
        amp=False,
    )

    # Print metrics as JSON for AutoTrain extraction
    metrics = results.results_dict
    map50 = metrics.get("metrics/mAP50(B)", 0.0)
    map50_95 = metrics.get("metrics/mAP50-95(B)", 0.0)
    precision = metrics.get("metrics/precision(B)", 0.0)
    recall = metrics.get("metrics/recall(B)", 0.0)

    print(f'{{"mAP": {map50:.4f}, "mAP50_95": {map50_95:.4f}, "precision": {precision:.4f}, "recall": {recall:.4f}}}')


if __name__ == "__main__":
    main()
