import json
from pathlib import Path

import pytest

from vtrack.artifacts import (
    compute_baseline_delta,
    extract_metrics_summary,
    publish_artifact_bundle,
)
from vtrack.settings import ProjectPaths


class FakeBoxMetrics:
    map50 = 0.84
    map = 0.61
    mp = 0.88
    mr = 0.76

    def class_result(self, index: int) -> tuple[float, float, float, float]:
        values = {
            0: (0.92, 0.89, 0.95, 0.77),
            1: (0.86, 0.70, 0.81, 0.49),
        }
        return values[index]


class FakeMetrics:
    box = FakeBoxMetrics()
    names = {0: "car", 1: "cyclist"}
    speed = {"inference": 40.4, "postprocess": 0.1}


def test_extract_metrics_summary_includes_per_class_values() -> None:
    summary = extract_metrics_summary(FakeMetrics())

    assert summary["overall"]["map50"] == 0.84
    assert summary["per_class"][0]["class_name"] == "car"
    assert summary["per_class"][1]["map50_95"] == 0.49


def test_compute_baseline_delta_compares_overall_metrics() -> None:
    delta = compute_baseline_delta(
        {"overall": {"map50": 0.84, "map50_95": 0.61, "precision": 0.88, "recall": 0.76}},
        {"overall": {"map50": 0.02, "map50_95": 0.01, "precision": 0.17, "recall": 0.04}},
    )

    assert delta["map50"] == pytest.approx(0.82)
    assert delta["precision"] == pytest.approx(0.71)


def test_publish_artifact_bundle_writes_summary_manifest_and_selected_files(tmp_path: Path) -> None:
    paths = ProjectPaths(root=tmp_path)
    paths.ensure_runtime_dirs()

    raw_dir = paths.raw_evaluation_dir / "demo_eval"
    weights_dir = raw_dir / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    (raw_dir / "confusion_matrix.png").write_bytes(b"png")
    (raw_dir / "metrics.csv").write_text("metric,value\nmap50,0.84\n", encoding="utf-8")
    (weights_dir / "best.pt").write_bytes(b"weights")

    bundle = publish_artifact_bundle(
        paths=paths,
        run_type="eval",
        run_id="demo_eval",
        summary={"finetuned": {"overall": {"map50": 0.84}}},
        command=["vtrack", "evaluate", "--model", "models/best.pt"],
        raw_output_path=raw_dir,
        dataset_path="/tmp/kitti.yaml",
        checkpoint_path=str(weights_dir / "best.pt"),
    )

    manifest = json.loads(bundle.manifest_path.read_text(encoding="utf-8"))
    summary = json.loads(bundle.summary_path.read_text(encoding="utf-8"))

    assert manifest["run_id"] == "demo_eval"
    assert summary["finetuned"]["overall"]["map50"] == 0.84
    assert (bundle.bundle_dir / "plots" / "confusion_matrix.png").exists()
    assert (bundle.bundle_dir / "files" / "metrics.csv").exists()
    assert (bundle.bundle_dir / "weights" / "best.pt").exists()
