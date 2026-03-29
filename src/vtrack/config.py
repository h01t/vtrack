"""Shared configuration and constants."""

# COCO class IDs for vehicles (used with pretrained yolo11n.pt)
COCO_VEHICLE_CLASSES = [2, 3, 5, 7]  # car, motorcycle, bus, truck

COCO_VEHICLE_NAMES = {
    2: "car",
    3: "motorcycle",
    5: "bus",
    7: "truck",
}

# KITTI class IDs (used with fine-tuned model)
KITTI_CLASSES = [0, 1, 2, 3, 4, 5, 6, 7]  # all 8 KITTI classes
KITTI_VEHICLE_CLASSES = [0, 1, 2, 6]  # car, van, truck, tram

KITTI_NAMES = {
    0: "car",
    1: "van",
    2: "truck",
    3: "pedestrian",
    4: "person_sitting",
    5: "cyclist",
    6: "tram",
    7: "misc",
}

# Default model
DEFAULT_MODEL = "yolo11n.pt"
DEFAULT_TRACKER = "bytetrack"
DEFAULT_DATASET = "kitti.yaml"

# Detection defaults
DEFAULT_MIN_CONFIDENCE = 0.25
DEFAULT_CONFIDENCE = DEFAULT_MIN_CONFIDENCE  # compatibility alias
DEFAULT_TRACK_CONFIDENCE = 0.10
DEFAULT_IOU = 0.7
DEFAULT_IMGSZ = 640
DEFAULT_MAX_DET = 300
DEFAULT_HALF = False
DEFAULT_VID_STRIDE = 1
DEFAULT_STREAM_BUFFER = False
DEFAULT_AGNOSTIC_NMS = False

DEFAULT_BENCHMARK_TRACKERS = ("bytetrack", "bytetrack-occlusion", "botsort")
