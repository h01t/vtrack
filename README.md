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
├── src/vtrack/              # Core library
│   ├── config.py            # Model/class configuration (COCO + KITTI)
│   ├── detect.py            # VehicleDetector — image/video detection
│   ├── track.py             # VehicleTracker — ByteTrack integration
│   ├── visualize.py         # Visualizer — boxes, trails, FPS overlay
│   ├── analytics.py         # VehicleAnalytics — counting, zones, export
│   └── pipeline.py          # VehiclePipeline — end-to-end orchestrator
├── scripts/
│   ├── demo.py              # Full CLI demo (detection + tracking + analytics)
│   ├── train.py             # Fine-tuning script
│   ├── evaluate.py          # Model evaluation + comparison
│   ├── train_remote.sh      # Remote CUDA training wrapper
│   ├── detect_image.py      # Quick single-image test
│   └── detect_video.py      # Video detection (no tracking)
├── models/                  # Trained weights (gitignored)
│   ├── best.pt              # Best fine-tuned checkpoint
│   └── last.pt              # Final epoch checkpoint
├── configs/                 # Tracker configs
├── data/                    # Datasets (gitignored)
├── outputs/                 # Results (gitignored)
├── tasks/                   # Project tracking
│   ├── todo.md
│   └── lessons.md
└── pyproject.toml
```

## Quick Start

```bash
# Clone and set up
git clone <repo-url> && cd object-det
uv sync

# Run with pretrained model (auto-downloads yolo11n.pt)
uv run python scripts/demo.py path/to/video.mp4

# Run with fine-tuned KITTI model
uv run python scripts/demo.py path/to/video.mp4 --model models/best.pt

# Enable analytics with a counting line
uv run python scripts/demo.py path/to/video.mp4 \
    --model models/best.pt \
    --analytics \
    --line 0,400,1280,400 \
    --export-json outputs/summary.json \
    --export-csv outputs/frames.csv \
    --save outputs/annotated.mp4

# Webcam (live)
uv run python scripts/demo.py 0 --model models/best.pt --analytics

# Single image detection
uv run python scripts/detect_image.py path/to/image.jpg
```

### CLI Options

```
usage: demo.py source [options]

positional arguments:
  source                Video file, camera index (0), RTSP URL, or YouTube URL

options:
  --model MODEL         Model weights (default: yolo11n.pt)
  --confidence FLOAT    Detection threshold (default: 0.25)
  --tracker TRACKER     bytetrack.yaml or botsort.yaml
  --trace-length INT    Track trail length in frames (default: 30)
  --save PATH           Save annotated video
  --no-display          Headless mode (no OpenCV window)
  --analytics           Enable counting/stats overlay
  --line x1,y1,x2,y2   Counting line coordinates
  --zone x1,y1,...      Monitoring zone polygon (min 3 points)
  --export-csv PATH     Per-frame data export
  --export-json PATH    Summary + track details export
```

## Training

### Remote CUDA training (recommended)

```bash
# Syncs code to remote, trains, pulls weights back
./scripts/train_remote.sh --epochs 50 --batch 16
```

### Local training

```bash
# CPU (slow — ~18 hours for 50 epochs)
uv run python scripts/train.py --device cpu --epochs 50

# MPS — not recommended (PyTorch task assigner bug)
uv run python scripts/train.py --device mps --no-amp --epochs 50
```

### Evaluation

```bash
# Evaluate fine-tuned model
uv run python scripts/evaluate.py --model models/best.pt --data kitti.yaml

# Compare against pretrained baseline
uv run python scripts/evaluate.py --model models/best.pt --data kitti.yaml --compare
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
- **ByteTrack tuning** — Adjust `track_buffer` for better occlusion handling in dense traffic.
- **FPS benchmarks** — Profile YOLOv11n vs. YOLOv11s on Apple M4 Pro (MPS inference works, training doesn't).
- **BoT-SORT comparison** — Benchmark against ByteTrack for accuracy/speed tradeoff.

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
