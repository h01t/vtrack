"""Run detection (without tracking) on a video file."""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vtrack.detect import VehicleDetector


def main():
    parser = argparse.ArgumentParser(description="Detect vehicles in a video")
    parser.add_argument("source", help="Video file path")
    parser.add_argument("--model", default="yolo11n.pt", help="Model weights")
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--save", action="store_true", help="Save annotated video")
    args = parser.parse_args()

    detector = VehicleDetector(model_path=args.model, confidence=args.confidence)

    frame_count = 0
    for result in detector.detect_video(args.source, save=args.save):
        boxes = result.boxes
        frame_count += 1
        if frame_count % 30 == 0:
            print(f"Frame {frame_count}: {len(boxes)} vehicles detected")

    print(f"\nProcessed {frame_count} frames total.")


if __name__ == "__main__":
    main()
