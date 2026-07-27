"""Unit tests for MOTChallenge formatting helpers (no dataset required)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from vtrack.mot_eval import format_mot_line, resolve_sequence_dir, summarize_trackeval_results


def test_format_mot_line() -> None:
    line = format_mot_line(frame=3, track_id=9, x=1.5, y=2.5, w=10.0, h=20.0, conf=0.87)
    assert line.startswith("3,9,1.50,2.50,10.00,20.00,0.8700")


def test_resolve_sequence_dir_prefers_frcnn(tmp_path: Path) -> None:
    train = tmp_path / "train"
    (train / "MOT17-02-DPM").mkdir(parents=True)
    (train / "MOT17-02-FRCNN").mkdir(parents=True)
    resolved = resolve_sequence_dir(tmp_path, "MOT17-02")
    assert resolved.name == "MOT17-02-FRCNN"


def test_summarize_trackeval_results_combined() -> None:
    results = {
        "MotChallenge2DBox": {
            "vtrack_bytetrack": {
                "COMBINED_SEQ": {
                    "pedestrian": {
                        "HOTA": {"HOTA": np.array([0.41, 0.39])},
                        "CLEAR": {"MOTA": 0.55},
                        "Identity": {"IDF1": 0.48},
                    }
                }
            }
        }
    }
    summary = summarize_trackeval_results(results)
    assert summary["hota"] == pytest.approx(0.40)
    assert summary["mota"] == pytest.approx(0.55)
    assert summary["idf1"] == pytest.approx(0.48)


def test_resolve_sequence_dir_missing(tmp_path: Path) -> None:
    (tmp_path / "train").mkdir()
    with pytest.raises(FileNotFoundError):
        resolve_sequence_dir(tmp_path, "MOT17-99")
