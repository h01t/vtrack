"""Run CUDA tracker×half benches and merge a portfolio performance CSV."""

from __future__ import annotations

import argparse
import csv
import subprocess
import threading
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _sample_vram_mb(stop: threading.Event, samples: list[float], interval: float = 0.25) -> None:
    while not stop.is_set():
        try:
            completed = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used",
                    "--format=csv,noheader,nounits",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            samples.append(float(completed.stdout.strip().splitlines()[0]))
        except (OSError, subprocess.CalledProcessError, ValueError):
            pass
        stop.wait(interval)


def _run_benchmark(
    *,
    source: Path,
    model: Path,
    trackers: list[str],
    half: bool,
    device: str,
    max_frames: int,
    export_csv: Path,
) -> dict:
    export_csv.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "uv",
        "run",
        "vtrack",
        "benchmark-track",
        str(source),
        "--model",
        str(model),
        "--device",
        device,
        "--max-frames",
        str(max_frames),
        "--warmup-frames",
        "30",
        "--export-csv",
        str(export_csv),
    ]
    for tracker in trackers:
        command.extend(["--tracker", tracker])
    if half:
        command.append("--half")

    samples: list[float] = []
    stop = threading.Event()
    sampler = threading.Thread(target=_sample_vram_mb, args=(stop, samples), daemon=True)
    sampler.start()
    try:
        completed = subprocess.run(
            command,
            cwd=ROOT,
            check=True,
            capture_output=True,
            text=True,
        )
    finally:
        stop.set()
        sampler.join(timeout=2.0)

    if completed.returncode != 0:
        raise RuntimeError(completed.stderr or completed.stdout)

    if not export_csv.is_file():
        raise RuntimeError(f"Benchmark CSV missing: {export_csv}\n{completed.stdout}")

    peak_vram = max(samples) if samples else 0.0
    return {
        "export_csv": export_csv,
        "peak_vram_mb": peak_vram,
        "half": half,
        "model": str(model),
    }


def _flatten_rows(payloads: list[dict]) -> list[dict]:
    rows: list[dict] = []
    for payload in payloads:
        with open(payload["export_csv"], newline="", encoding="utf-8") as handle:
            for run in csv.DictReader(handle):
                row = dict(run)
                row["half"] = payload["half"]
                row["model"] = Path(payload["model"]).name
                row["peak_vram_mb"] = round(payload["peak_vram_mb"], 1)
                rows.append(row)
    return rows


def _write_csv(rows: list[dict], path: Path) -> None:
    fieldnames = [
        "model",
        "tracker",
        "device",
        "half",
        "imgsz",
        "frames_processed",
        "timed_frames",
        "avg_fps",
        "median_fps",
        "p95_frame_ms",
        "peak_vram_mb",
        "unique_tracks",
        "avg_track_duration_frames",
        "short_tracks_lt_5_frames",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT / "data" / "test-video.mp4")
    parser.add_argument("--model", type=Path, default=ROOT / "models" / "best.pt")
    parser.add_argument("--yolo11s", type=Path, default=ROOT / "yolo11s.pt")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-frames", type=int, default=150)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path("/srv/ai/outputs/vtrack/bench"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=ROOT / "docs" / "media" / "benchmark-cuda.csv",
    )
    args = parser.parse_args()

    if not args.source.exists():
        raise SystemExit(f"Missing source clip: {args.source}")
    if not args.model.exists():
        raise SystemExit(f"Missing fine-tuned model: {args.model}")

    trackers = ["bytetrack", "bytetrack-occlusion", "botsort"]
    payloads: list[dict] = []

    for half in (False, True):
        label = "half" if half else "fp32"
        export_csv = args.work_dir / f"best_{label}.csv"
        print(f"==> best.pt trackers half={half}")
        payloads.append(
            _run_benchmark(
                source=args.source,
                model=args.model,
                trackers=trackers,
                half=half,
                device=args.device,
                max_frames=args.max_frames,
                export_csv=export_csv,
            )
        )

    # Ensure pretrained s weights exist (Ultralytics will download on first use).
    print("==> yolo11s.pt bytetrack half=True (contrast)")
    payloads.append(
        _run_benchmark(
            source=args.source,
            model=args.yolo11s,
            trackers=["bytetrack"],
            half=True,
            device=args.device,
            max_frames=args.max_frames,
            export_csv=args.work_dir / "yolo11s_half.csv",
        )
    )

    rows = _flatten_rows(payloads)
    _write_csv(rows, args.out_csv)
    print(f"Wrote {args.out_csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
