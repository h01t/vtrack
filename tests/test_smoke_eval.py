import os
from pathlib import Path

import pytest

from vtrack.settings import EvaluationConfig, ProjectPaths
from vtrack.workflows import run_evaluation

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best.pt"
DATASET_PATH = Path("/Users/grmim/Dev/datasets/kitti/kitti.yaml")


@pytest.mark.smoke
def test_local_evaluation_smoke(tmp_path: Path) -> None:
    if os.environ.get("VTRACK_RUN_SMOKE") != "1":
        pytest.skip("Set VTRACK_RUN_SMOKE=1 to run smoke evaluation.")
    if not MODEL_PATH.exists():
        pytest.skip("Local best model is not available.")
    if not DATASET_PATH.exists():
        pytest.skip("Local KITTI dataset config is not available.")

    paths = ProjectPaths(root=tmp_path)
    result = run_evaluation(
        evaluation=EvaluationConfig(
            model_path=str(MODEL_PATH),
            data=str(DATASET_PATH),
            name="smoke_eval",
            compare=False,
        ),
        paths=paths,
        command=["pytest", "-m", "smoke"],
    )

    summary = result["summary"]["finetuned"]["overall"]
    summary_path = result["bundle"].summary_path

    assert 0.84 <= summary["map50"] <= 0.86
    assert 0.60 <= summary["map50_95"] <= 0.62
    assert 0.87 <= summary["precision"] <= 0.89
    assert 0.74 <= summary["recall"] <= 0.77
    assert summary_path.exists()
