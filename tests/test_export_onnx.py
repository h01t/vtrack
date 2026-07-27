from pathlib import Path

from vtrack.settings import ProjectPaths
from vtrack.workflows import run_export_onnx


class FakeYOLO:
    def __init__(self, model_path: str):
        self.model_path = model_path

    def export(self, format: str, imgsz: int):
        assert format == "onnx"
        assert imgsz == 640
        out = Path(self.model_path).with_suffix(".onnx")
        out.write_bytes(b"onnx")
        return str(out)


def test_run_export_onnx_copies_to_models_and_mirror(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr("vtrack.workflows.YOLO", FakeYOLO)
    mirror = tmp_path / "mirror"
    monkeypatch.setenv("VTRACK_CHECKPOINT_DIR", str(mirror))

    root = tmp_path / "project"
    models = root / "models"
    models.mkdir(parents=True)
    source = models / "best.pt"
    source.write_bytes(b"pt")

    paths = ProjectPaths(root=root)
    target = run_export_onnx(model_path=source, paths=paths, imgsz=640)

    assert target == (models / "best.onnx").resolve()
    assert target.read_bytes() == b"onnx"
    assert (mirror / "best.onnx").read_bytes() == b"onnx"
