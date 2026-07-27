"""MOT17 evaluation helpers: MOTChallenge preds + TrackEval metrics."""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from ultralytics import YOLO

from vtrack.tracker_presets import resolve_tracker_config

DEFAULT_MOT17_SEQUENCES = ("MOT17-02", "MOT17-04", "MOT17-09")
PERSON_CLASS_ID = 0


def format_mot_line(
    *,
    frame: int,
    track_id: int,
    x: float,
    y: float,
    w: float,
    h: float,
    conf: float,
) -> str:
    """Format one MOTChallenge prediction row."""
    return (
        f"{frame},{track_id},{x:.2f},{y:.2f},{w:.2f},{h:.2f},{conf:.4f},-1,-1,-1"
    )


def resolve_sequence_dir(dataset_root: Path, sequence: str) -> Path:
    """Resolve MOT17-02 → train/MOT17-02-FRCNN (prefer FRCNN, then SDP, then DPM)."""
    train_root = dataset_root / "train"
    if not train_root.is_dir():
        raise FileNotFoundError(f"MOT17 train split missing under {dataset_root}")

    if (train_root / sequence).is_dir():
        return train_root / sequence

    for detector in ("FRCNN", "SDP", "DPM"):
        candidate = train_root / f"{sequence}-{detector}"
        if candidate.is_dir():
            return candidate

    raise FileNotFoundError(
        f"Sequence {sequence!r} not found under {train_root} "
        "(expected MOT17-XX or MOT17-XX-FRCNN)"
    )


def write_mot_predictions(
    *,
    model: YOLO,
    sequence_dir: Path,
    output_txt: Path,
    tracker_yaml: str | Path,
    device: str | None,
    imgsz: int,
    conf: float,
    max_frames: int | None = None,
) -> int:
    """Run person tracking on a MOT17 sequence and write MOTChallenge predictions."""
    img_dir = sequence_dir / "img1"
    frames = sorted(img_dir.glob("*.jpg"))
    if not frames:
        raise FileNotFoundError(f"No frames under {img_dir}")
    if max_frames is not None:
        frames = frames[: max(0, max_frames)]

    output_txt.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    kwargs: dict[str, object] = {
        "persist": True,
        "tracker": str(tracker_yaml),
        "classes": [PERSON_CLASS_ID],
        "conf": conf,
        "imgsz": imgsz,
        "verbose": False,
    }
    if device is not None:
        kwargs["device"] = device

    for frame_idx, frame_path in enumerate(frames, start=1):
        image = cv2.imread(str(frame_path))
        if image is None:
            continue
        result = model.track(source=image, **kwargs)[0]
        boxes = result.boxes
        if boxes is None or len(boxes) == 0 or boxes.id is None:
            continue
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        ids = boxes.id.cpu().numpy()
        for i in range(len(boxes)):
            x1, y1, x2, y2 = xyxy[i].tolist()
            lines.append(
                format_mot_line(
                    frame=frame_idx,
                    track_id=int(ids[i]),
                    x=x1,
                    y=y1,
                    w=x2 - x1,
                    h=y2 - y1,
                    conf=float(confs[i]),
                )
            )

    output_txt.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return len(frames)


def _extract_metric_value(block: Any, key: str) -> float | None:
    if not isinstance(block, dict):
        return None
    value = block.get(key)
    if isinstance(value, (int, float, np.floating, np.integer)):
        return float(value)
    if isinstance(value, np.ndarray) and value.size:
        return float(np.mean(value))
    if isinstance(value, dict):
        nested = value.get(key)
        if isinstance(nested, (int, float, np.floating, np.integer)):
            return float(nested)
        if isinstance(nested, np.ndarray) and nested.size:
            return float(np.mean(nested))
        for candidate in value.values():
            if isinstance(candidate, (int, float, np.floating, np.integer)):
                return float(candidate)
            if isinstance(candidate, np.ndarray) and candidate.size:
                return float(np.mean(candidate))
    return None


def _metrics_from_seq_block(seq_block: Any) -> dict[str, float]:
    summary = {"hota": 0.0, "mota": 0.0, "idf1": 0.0}
    if not isinstance(seq_block, dict):
        return summary

    # TrackEval nests per-class under e.g. {'pedestrian': {'HOTA': ...}}
    if "HOTA" not in seq_block and any(isinstance(v, dict) for v in seq_block.values()):
        class_block = seq_block.get("pedestrian")
        if class_block is None:
            class_block = next(
                (v for v in seq_block.values() if isinstance(v, dict) and "HOTA" in v),
                None,
            )
        seq_block = class_block or seq_block

    hota = _extract_metric_value(seq_block.get("HOTA", {}), "HOTA")
    mota = _extract_metric_value(seq_block.get("CLEAR", {}), "MOTA")
    idf1 = _extract_metric_value(seq_block.get("Identity", {}), "IDF1")
    if hota is not None:
        summary["hota"] = hota
    if mota is not None:
        summary["mota"] = mota
    if idf1 is not None:
        summary["idf1"] = idf1
    return summary


def summarize_trackeval_results(results: Any) -> dict[str, float]:
    """Pull combined HOTA / MOTA / IDF1 from TrackEval's nested result dict."""
    if not results:
        return {"hota": 0.0, "mota": 0.0, "idf1": 0.0}

    try:
        dataset_block = next(iter(results.values()))
        tracker_block = next(iter(dataset_block.values()))
    except (StopIteration, AttributeError):
        return {"hota": 0.0, "mota": 0.0, "idf1": 0.0}

    if "COMBINED_SEQ" in tracker_block:
        return _metrics_from_seq_block(tracker_block["COMBINED_SEQ"])

    hota_vals: list[float] = []
    mota_vals: list[float] = []
    idf1_vals: list[float] = []
    for seq_metrics in tracker_block.values():
        summary = _metrics_from_seq_block(seq_metrics)
        if summary["hota"]:
            hota_vals.append(summary["hota"])
        if summary["mota"]:
            mota_vals.append(summary["mota"])
        if summary["idf1"]:
            idf1_vals.append(summary["idf1"])
    return {
        "hota": float(np.mean(hota_vals)) if hota_vals else 0.0,
        "mota": float(np.mean(mota_vals)) if mota_vals else 0.0,
        "idf1": float(np.mean(idf1_vals)) if idf1_vals else 0.0,
    }


def run_trackeval(
    *,
    gt_folder: Path,
    tracker_folder: Path,
    tracker_name: str,
    seq_names: Sequence[str],
    seq_lengths: dict[str, int] | None = None,
) -> dict[str, float]:
    """Evaluate MOTChallenge-format predictions with TrackEval."""
    # TrackEval still references removed NumPy aliases (np.float / np.int / …).
    for name, replacement in (
        ("float", float),
        ("int", int),
        ("bool", bool),
    ):
        if not hasattr(np, name):
            setattr(np, name, replacement)

    try:
        import trackeval
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "TrackEval missing. Install with: uv sync --extra mot"
        ) from exc

    eval_config = trackeval.Evaluator.get_default_eval_config()
    eval_config.update(
        {
            "DISPLAY_RESULTS": False,
            "PLOT_CURVES": False,
            "OUTPUT_DETAILED": False,
            "PRINT_RESULTS": False,
            "PRINT_CONFIG": False,
            "TIME_PROGRESS": False,
            "USE_PARALLEL": False,
        }
    )
    if seq_lengths:
        seq_info: dict[str, int | None] = {
            name: int(seq_lengths[name]) for name in seq_names
        }
    else:
        seq_info = {name: None for name in seq_names}
    dataset_config = trackeval.datasets.MotChallenge2DBox.get_default_dataset_config()
    dataset_config.update(
        {
            "GT_FOLDER": str(gt_folder),
            "TRACKERS_FOLDER": str(tracker_folder),
            "TRACKERS_TO_EVAL": [tracker_name],
            "CLASSES_TO_EVAL": ["pedestrian"],
            "BENCHMARK": "MOT17",
            "SPLIT_TO_EVAL": "train",
            "SKIP_SPLIT_FOL": True,
            "SEQ_INFO": seq_info,
            "GT_LOC_FORMAT": "{gt_folder}/{seq}/gt/gt.txt",
            "TRACKER_SUB_FOLDER": "data",
            "PRINT_CONFIG": False,
        }
    )
    metrics_config = {
        "METRICS": ["HOTA", "CLEAR", "Identity"],
        "THRESHOLD": 0.5,
    }

    evaluator = trackeval.Evaluator(eval_config)
    dataset_list = [trackeval.datasets.MotChallenge2DBox(dataset_config)]
    metrics_list = []
    for metric in [trackeval.metrics.HOTA, trackeval.metrics.CLEAR, trackeval.metrics.Identity]:
        if metric.get_name() in metrics_config["METRICS"]:
            metrics_list.append(metric(metrics_config))

    results, _ = evaluator.evaluate(dataset_list, metrics_list)
    return summarize_trackeval_results(results)


def evaluate_mot17(
    *,
    dataset_root: Path,
    model_path: str,
    tracker: str,
    device: str | None,
    imgsz: int,
    conf: float,
    sequences: Sequence[str],
    work_dir: Path,
    max_frames: int | None = None,
    tracker_name: str = "vtrack_bytetrack",
) -> dict[str, Any]:
    """Track selected MOT17 sequences and compute HOTA/MOTA/IDF1."""
    dataset_root = Path(dataset_root)
    work_dir = Path(work_dir)
    pred_root = work_dir / "trackers" / tracker_name / "data"
    pred_root.mkdir(parents=True, exist_ok=True)

    resolved_tracker = resolve_tracker_config(tracker)
    model = YOLO(model_path)

    seq_dirs: list[Path] = []
    seq_names: list[str] = []
    frames_processed: dict[str, int] = {}

    for sequence in sequences:
        seq_dir = resolve_sequence_dir(dataset_root, sequence)
        seq_name = seq_dir.name
        seq_dirs.append(seq_dir)
        seq_names.append(seq_name)
        pred_txt = pred_root / f"{seq_name}.txt"
        frames_processed[seq_name] = write_mot_predictions(
            model=model,
            sequence_dir=seq_dir,
            output_txt=pred_txt,
            tracker_yaml=resolved_tracker.path,
            device=device,
            imgsz=imgsz,
            conf=conf,
            max_frames=max_frames,
        )

    metrics = run_trackeval(
        gt_folder=dataset_root / "train",
        tracker_folder=work_dir / "trackers",
        tracker_name=tracker_name,
        seq_names=seq_names,
        seq_lengths=frames_processed if max_frames is not None else None,
    )

    report = {
        "dataset": "MOT17",
        "split": "train",
        "model": model_path,
        "tracker": resolved_tracker.name,
        "device": device,
        "sequences": seq_names,
        "frames_processed": frames_processed,
        "metrics": metrics,
        "notes": (
            "Pedestrian (person) tracks on MOT17 train subset with "
            "COCO-pretrained YOLO. Not a hidden-test leaderboard submission; "
            "KITTI mAP remains the vehicle detection card."
        ),
    }
    summary_path = work_dir / "summary.json"
    summary_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report
