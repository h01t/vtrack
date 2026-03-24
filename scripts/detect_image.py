"""Quick smoke test: run YOLOv11 pretrained detection on a sample image."""

import sys
from pathlib import Path

# Add src to path so vtrack is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from vtrack.detect import VehicleDetector


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else "https://ultralytics.com/images/bus.jpg"

    detector = VehicleDetector(model_path="yolo11n.pt")
    results = detector.detect_image(source)

    for r in results:
        boxes = r.boxes
        print(f"\nDetected {len(boxes)} vehicle(s):")
        for box in boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            name = detector.class_names.get(cls_id, f"class_{cls_id}")
            print(f"  - {name}: {conf:.2f}")

        # Show the annotated image
        r.show()


if __name__ == "__main__":
    main()
