"""Shared configuration and constants."""

# COCO class IDs for vehicles
VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

VEHICLE_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# Default model
DEFAULT_MODEL = "yolo11n.pt"

# Detection defaults
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640
