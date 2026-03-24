"""Fine-tune YOLOv11 on a vehicle detection dataset."""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv11 for vehicle detection")
    parser.add_argument("--model", default="yolo11n.pt", help="Base model weights")
    parser.add_argument("--data", default="kitti.yaml", help="Dataset config (auto-downloads)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="mps", help="Device: mps, cpu, or cuda")
    parser.add_argument("--name", default="vehicle_v1", help="Run name")
    parser.add_argument("--no-amp", action="store_true", help="Disable mixed precision (fixes MPS crashes)")
    args = parser.parse_args()

    model = YOLO(args.model)

    results = model.train(
        amp=not args.no_amp,
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project="outputs/training",
        name=args.name,
        exist_ok=True,
        pretrained=True,
        optimizer="auto",
        lr0=0.01,
        patience=10,
        save=True,
        plots=True,
    )

    # Print final metrics
    print("\n--- Training Complete ---")
    print(f"Best weights: outputs/training/{args.name}/weights/best.pt")

    return results


if __name__ == "__main__":
    main()
