# vtrack — Vehicle Detection & Tracking Pipeline

End-to-end vehicle detection, multi-object tracking, and analytics pipeline built on YOLOv11 and fine-tuned on the KITTI dataset.

## What This Project Does

vtrack takes video input (file, webcam, RTSP stream, or YouTube URL) and:

1. **Detects** vehicles frame-by-frame using YOLOv11
2. **Tracks** them across frames with persistent IDs via ByteTrack
3. **Visualizes** bounding boxes, track trails, and an FPS counter in real-time
4. **Analyzes** traffic patterns — line-crossing counts, zone occupancy, per-class breakdowns, and track duration statistics
5. **Exports** analytics to CSV (per-frame) and JSON (summary)

## Why This Stack

| Component | Choice | Reasoning |
|-----------|--------|-----------|
| **Detection** | YOLOv11 (Ultralytics) | State-of-the-art single-stage detector. Ultralytics provides a clean Python API with built-in training, validation, export, and tracker integration — no glue code needed. |
| **Tracking** | ByteTrack (via Ultralytics) | Lightweight, high-performance multi-object tracker that handles occlusion well. Ships with Ultralytics, so `model.track()` is a one-liner with `persist=True` for frame-to-frame ID persistence. |
| **Visualization** | supervision (Roboflow) | Purpose-built for CV annotation overlays. Provides `BoxAnnotator`, `TraceAnnotator`, `LineZone`, `PolygonZone` out of the box — significantly less boilerplate than raw OpenCV drawing. |
| **Dataset** | KITTI | Well-established autonomous driving benchmark with 7,481 annotated images and 8 vehicle-relevant classes. Auto-downloads via Ultralytics, no manual setup. |
| **Training hardware** | NVIDIA 3060 Ti (remote CUDA) | Apple Silicon MPS has known PyTorch bugs in the YOLO task assigner (`index out of bounds`). Remote CUDA training on a 3060 Ti (8GB VRAM) completed 50 epochs in ~28 minutes vs. an estimated ~18 hours on CPU. |
| **Environment** | uv + Python 3.12 | Fast dependency resolution, lockfile support, and no need for conda. |

## Model Results

**Fine-tuned YOLOv11n on KITTI (50 epochs, 3060 Ti CUDA)**

| Metric | Value |
|--------|-------|
| mAP@0.5 | **0.850** |
| mAP@0.5:0.95 | 0.602 |
| Precision | 0.854 |
| Recall | 0.791 |
| Training time | ~28 minutes |
| Model size | 5.4 MB |

**Per-class performance (mAP@0.5):**

| Class | mAP@0.5 |
|-------|---------|
| Car | 0.927 |
| Van | 0.854 |
| Truck | 0.880 |
| Pedestrian | 0.814 |
| Person sitting | 0.668 |
| Cyclist | 0.882 |
| Tram | 0.954 |
| Misc | 0.824 |

The pretrained COCO model scored mAP@0.5 of 0.022 on KITTI due to class ID mismatch — fine-tuning gave a **39x improvement**.

## Project Structure

```
object-det/
├── src/vtrack/              # Installable package
│   ├── cli.py               # Unified `vtrack` CLI
│   ├── settings.py          # Typed settings + canonical path layout
│   ├── model_profiles.py    # Checkpoint metadata / class profile resolution
│   ├── artifacts.py         # Normalized artifact bundle publishing
│   ├── workflows.py         # Shared runtime / train / eval workflows
│   ├── benchmarking.py      # Tracking benchmark report generation
│   ├── tracker_presets.py   # Built-in tracker preset resolution
│   ├── detect.py            # VehicleDetector — image/video detection
│   ├── track.py             # VehicleTracker — ByteTrack / BoT-SORT integration
│   ├── trackers/            # Repo-owned tracker YAML presets
│   ├── visualize.py         # Visualizer — boxes, trails, FPS overlay
│   ├── analytics.py         # VehicleAnalytics — counting, zones, export
│   └── pipeline.py          # VehiclePipeline — end-to-end orchestrator
├── scripts/                 # Backward-compatible wrappers around `vtrack`
├── models/                  # Local checkpoints (gitignored)
├── artifacts/               # Normalized train/eval bundles (gitignored)
├── runs/                    # Raw Ultralytics outputs (gitignored)
├── data/                    # Local datasets (gitignored)
├── tests/                   # Fast unit/CLI tests + opt-in smoke test
└── pyproject.toml
```

## Quick Start

```bash
# Clone and set up
git clone <repo-url> && cd object-det
uv sync
uv sync --extra dev   # recommended for pytest + ruff

# Run with fine-tuned KITTI model
uv run vtrack demo data/test-video.mp4 --model models/best.pt

# Enable analytics with a counting line
uv run vtrack demo data/test-video.mp4 \
    --model models/best.pt \
    --analytics \
    --line 0,400,1280,400 \
    --export-json outputs/summary.json \
    --export-csv outputs/frames.csv \
    --save outputs/annotated.mp4

# Webcam (live)
uv run vtrack demo 0 --model models/best.pt --analytics

# Single image detection
uv run vtrack detect-image data/test-image.jpg

# Compare built-in tracker presets on the same clip
uv run vtrack benchmark-track data/test-video.mp4 \
    --model models/best.pt \
    --device cpu \
    --tracker bytetrack \
    --tracker bytetrack-occlusion \
    --tracker botsort \
    --max-frames 150 \
    --export-csv outputs/benchmark.csv
```

The legacy `scripts/*.py` entrypoints still work, but they now delegate to the same installable CLI.

## Runtime Inference and Tracking

### Core Commands

```
usage: vtrack <command> [options]

commands:
  demo                  Tracking + analytics on a video source
  benchmark-track       Compare tracking presets on a shared source
  detect-image          Single-image detection
  detect-video          Detection-only video pass
  train                 Local training
  evaluate              Local evaluation and optional baseline comparison
  train-remote          Remote training + artifact sync

See `uv run vtrack <command> --help` for subcommand-specific options.
```

### Tracking Presets

- `bytetrack` — repo-owned baseline matching Ultralytics ByteTrack defaults
- `bytetrack-occlusion` — longer lost-track buffer (`track_buffer=60`) for heavier occlusion
- `botsort` — repo-owned BoT-SORT baseline with `gmc_method=sparseOptFlow` and `with_reid=False`
- `--tracker` accepts any preset alias above or an explicit YAML path

### Runtime Notes

- For tracking commands, `--track-conf` controls the detector threshold fed into the tracker, while `--confidence` controls the minimum confidence kept for overlays, analytics, and exported summaries.
- `vtrack benchmark-track` runs one or more tracker presets sequentially on the same source and prints a JSON report to stdout. If `--export-csv` is provided, it also writes one summary row per run.
- The bundled `data/test-video.mp4` clip is useful for smoke tests and CLI verification, but real tracker comparisons should be run on continuous traffic footage where occlusion and ID persistence matter.
- MPS is supported for inference via `demo`, `detect-image`, `detect-video`, and `benchmark-track`, but this project still treats Apple Silicon support as inference-only for now.

### More Examples

```bash
# Run with pretrained model (auto-downloads yolo11n.pt)
uv run vtrack demo data/test-video.mp4

# Use a repo-owned tracker preset tuned for longer occlusions
uv run vtrack demo data/test-video.mp4 \
    --model models/best.pt \
    --tracker bytetrack-occlusion \
    --track-conf 0.10

# Apple Silicon inference (training still uses remote CUDA)
uv run vtrack demo data/test-video.mp4 \
    --model models/best.pt \
    --device mps \
    --no-display

# Detection-only video pass
uv run vtrack detect-video data/test-video.mp4 --model models/best.pt --save
```

## Training

### Remote CUDA training (recommended)

```bash
# Remote host configuration
export VTRACK_REMOTE_HOST=blackbox

# Optional override. By default vtrack prefers the repo-local remote virtualenv
# at .venv/bin/python and falls back to python3 only if that virtualenv is
# missing.
# export VTRACK_REMOTE_PYTHON=python3

# Optional override. Quote ~ so your local shell does not expand it first.
# If omitted, vtrack mirrors your local checkout path relative to $HOME,
# so /Users/grmim/Dev/object-det becomes ~/Dev/object-det on the remote side.
export VTRACK_REMOTE_DIR='~/Dev/object-det'

# Optional override for Ultralytics builtin datasets like kitti.yaml.
# If omitted, vtrack uses a sibling datasets directory next to the remote checkout,
# so ~/Dev/object-det defaults to ~/Dev/datasets.
# export VTRACK_REMOTE_DATASETS_DIR='~/Dev/datasets'

# Sync code, train remotely, and pull back the normalized artifact bundle + models
uv run vtrack train-remote --epochs 50 --batch 16
```

### Local training

```bash
# CPU (slow — ~18 hours for 50 epochs)
uv run vtrack train --device cpu --epochs 50

# MPS training — not recommended (PyTorch task assigner bug)
uv run vtrack train --device mps --no-amp --epochs 50
```

### Evaluation

```bash
# Evaluate fine-tuned model
uv run vtrack evaluate --model models/best.pt --data /Users/grmim/Dev/datasets/kitti/kitti.yaml

# Compare against pretrained baseline
uv run vtrack evaluate \
    --model models/best.pt \
    --data /Users/grmim/Dev/datasets/kitti/kitti.yaml \
    --compare
```

## Artifacts

- Normalized bundles are written to `artifacts/train/<run-name>/` and `artifacts/eval/<run-name>/`.
- Each bundle includes `manifest.json`, `summary.json`, copied plots, and copied weights when relevant.
- Raw Ultralytics outputs live under `runs/` and are treated as implementation detail.
- Training also syncs/copies canonical checkpoints into `models/best.pt` and `models/last.pt`, plus named copies such as `models/vehicle_v1_best.pt`.

## Development

```bash
# Install project + dev tools
uv sync --extra dev

# Lint and tests
uv run ruff check src scripts tests
uv run pytest

# Opt-in smoke evaluation against local assets
VTRACK_RUN_SMOKE=1 uv run pytest -m smoke
```

## Use Cases

- **Traffic monitoring** — Count vehicles crossing a line or occupying a zone at intersections, highway ramps, or parking entrances
- **Autonomous driving R&D** — Validate perception models on KITTI-style driving footage
- **Parking lot management** — Monitor occupancy by vehicle type (car vs. truck vs. van)
- **Urban planning** — Collect traffic flow data (volume, composition, peak times) from existing camera infrastructure
- **Security/surveillance** — Track vehicle movement patterns, detect unusual behavior (wrong-way driving, prolonged stops)
- **Fleet management** — Count and classify vehicles entering/exiting depots or loading docks

## Future Improvements

### Near-term
- **Real continuous video testing** — Current validation is on KITTI stills stitched into clips. Test on dashcam and fixed-camera footage for real-world tracking performance.
- **Continuous tracker benchmarks** — Run the new `benchmark-track` workflow on longer traffic footage and record stable ByteTrack vs. BoT-SORT comparisons.
- **Formal MOT evaluation** — Add MOT-style ground-truth scoring (ID switches, MOTA/HOTA) on annotated video sequences.
- **FPS benchmarks** — Profile YOLOv11n vs. YOLOv11s on Apple M4 Pro using the new runtime device controls.

### Edge deployment
- **ONNX export** — For cross-platform CPU inference via ONNX Runtime.
- **CoreML export** — Optimized inference on Apple Silicon (macOS/iOS).
- **Raspberry Pi** — TFLite or NCNN for embedded deployment, continuing from prior thesis work.
- **Quantization** — INT8 quantization for 2-4x speedup on edge devices.

### Platform evolution
- **Web dashboard** — Real-time streaming with live analytics (FastAPI + React).
- **Multi-camera support** — Aggregate analytics across multiple video sources.
- **Re-identification** — Match vehicles across non-overlapping camera views.
- **Custom dataset training** — CVAT/Label Studio integration for annotation → training → deployment loop.
- **Autonomous training platform** — LLM-driven self-correcting training loop (see PLATFORM.md for the full concept).

## Dependencies

```
ultralytics>=8.3    # YOLOv11 detection + tracking
opencv-python>=4.10 # Video I/O and display
numpy>=1.26         # Array operations
supervision>=0.25   # CV visualization and zone utilities
lapx>=0.5           # Linear assignment for tracking
```

Optional: `onnx`, `onnxruntime` for model export.

## License

MIT
