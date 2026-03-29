import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models" / "best.pt"
VIDEO_PATH = ROOT / "data" / "test-video.mp4"
VTRACK_BIN = Path(sys.executable).with_name("vtrack")


@pytest.mark.smoke
def test_demo_cli_smoke() -> None:
    if os.environ.get("VTRACK_RUN_SMOKE") != "1":
        pytest.skip("Set VTRACK_RUN_SMOKE=1 to run smoke demo inference.")
    if not MODEL_PATH.exists():
        pytest.skip("Local best model is not available.")
    if not VIDEO_PATH.exists():
        pytest.skip("Local sample video is not available.")

    completed = subprocess.run(
        [
            str(VTRACK_BIN),
            "demo",
            str(VIDEO_PATH),
            "--model",
            str(MODEL_PATH),
            "--no-display",
            "--device",
            "cpu",
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert "Processed" in completed.stdout
