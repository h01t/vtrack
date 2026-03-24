"""Evaluate a trained model and compare against pretrained baseline."""

import argparse

from ultralytics import YOLO


def evaluate(model_path: str, data: str, name: str):
    """Run validation and print metrics."""
    model = YOLO(model_path)
    metrics = model.val(
        data=data,
        project="outputs/evaluation",
        name=name,
        exist_ok=True,
        plots=True,
    )

    print(f"\n--- {name} ---")
    print(f"mAP@0.5:      {metrics.box.map50:.4f}")
    print(f"mAP@0.5:0.95: {metrics.box.map:.4f}")
    print(f"Precision:     {metrics.box.mp:.4f}")
    print(f"Recall:        {metrics.box.mr:.4f}")

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate vehicle detection model")
    parser.add_argument("--model", required=True, help="Model weights path")
    parser.add_argument("--data", default="kitti.yaml", help="Dataset config")
    parser.add_argument("--name", default="eval", help="Run name")
    parser.add_argument(
        "--baseline", default="yolo11n.pt", help="Pretrained model for comparison"
    )
    parser.add_argument("--compare", action="store_true", help="Also run baseline evaluation")
    args = parser.parse_args()

    # Evaluate fine-tuned model
    finetuned = evaluate(args.model, args.data, f"{args.name}_finetuned")

    # Optionally compare with baseline
    if args.compare:
        baseline = evaluate(args.baseline, args.data, f"{args.name}_baseline")
        print("\n--- Comparison ---")
        delta_map50 = finetuned.box.map50 - baseline.box.map50
        delta_map = finetuned.box.map - baseline.box.map
        print(f"mAP@0.5 delta:      {delta_map50:+.4f}")
        print(f"mAP@0.5:0.95 delta: {delta_map:+.4f}")


if __name__ == "__main__":
    main()
