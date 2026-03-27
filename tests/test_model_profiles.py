from vtrack.config import COCO_VEHICLE_CLASSES, KITTI_NAMES
from vtrack.model_profiles import profile_from_names, resolve_model_profile


class FakeYOLO:
    def __init__(self, model_path: str):
        if model_path == "kitti.pt":
            self.names = KITTI_NAMES
        else:
            self.names = {
                0: "person",
                1: "bicycle",
                2: "car",
                3: "motorcycle",
                4: "airplane",
                5: "bus",
                6: "train",
                7: "truck",
            }


def test_profile_from_names_recognizes_kitti_metadata() -> None:
    profile = profile_from_names(KITTI_NAMES, source="models/best.pt")

    assert profile.class_filter is None
    assert profile.class_names == KITTI_NAMES


def test_profile_from_names_recognizes_coco_vehicle_filter() -> None:
    profile = profile_from_names(
        {
            0: "person",
            1: "bicycle",
            2: "car",
            3: "motorcycle",
            4: "airplane",
            5: "bus",
            6: "train",
            7: "truck",
        },
        source="yolo11n.pt",
    )

    assert profile.class_filter == COCO_VEHICLE_CLASSES
    assert profile.class_names[2] == "car"


def test_resolve_model_profile_uses_checkpoint_metadata(monkeypatch) -> None:
    monkeypatch.setattr("vtrack.model_profiles.YOLO", FakeYOLO)

    profile = resolve_model_profile("kitti.pt")

    assert profile.class_filter is None
    assert profile.class_names == KITTI_NAMES
