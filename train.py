"""AutoTrain-compatible training script for vehicle detection.

AutoTrain's agent modifies hyperparameters in this file between iterations.
The last line of stdout must be valid JSON with metric values.
Supports checkpoint resume via AUTOTRAIN_RESUME_FROM env var.
"""

import os

from ultralytics import YOLO


def main():
    # Check dataset exists
    import yaml
    data_path = "datasets/kitti/kitti.yaml"
    if not os.path.exists(data_path):
        print(f"ERROR: Dataset config not found at {data_path}")
        print('{"mAP": 0.0, "mAP50_95": 0.0, "precision": 0.0, "recall": 0.0}')
        return
    
    # Check if we should resume from a checkpoint
    resume_from = os.environ.get("AUTOTRAIN_RESUME_FROM")
    if resume_from and os.path.exists(resume_from):
        print(f"Resuming from checkpoint: {resume_from}")
        model = YOLO(resume_from)
    else:
        model = YOLO("yolov8n.pt")

    # --- Hyperparameters (AutoTrain agent modifies these) ---
    results = model.train(
        data=data_path,
        epochs=10,
        imgsz=320,
        batch=2,
        device="cuda",
        project="outputs/training",
        name="autotrain",
        exist_ok=True,
        pretrained=True,
        lr0=0.001,
        verbose=True,
        workers=0,
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
