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

# Detection defaults
DEFAULT_CONFIDENCE = 0.25
DEFAULT_IOU = 0.45
DEFAULT_IMGSZ = 640


def get_class_config(model_path: str) -> tuple[list[int] | None, dict[int, str]]:
    """Return (class_filter, class_names) based on model type.

    For KITTI-trained models: no class filter (all classes are relevant), KITTI names.
    For COCO pretrained: filter to vehicle classes only.
    """
    if "best" in model_path or "last" in model_path or "vehicle" in model_path:
        # Fine-tuned model — all classes are vehicle-related, no filter needed
        return None, KITTI_NAMES
    # Pretrained COCO model — filter to vehicle classes
    return COCO_VEHICLE_CLASSES, COCO_VEHICLE_NAMES
