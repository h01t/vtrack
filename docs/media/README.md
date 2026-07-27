# README Media Assets

Tracked, lightweight assets used by the public README:

- `hero-poster.png` — poster frame for the README hero
- `tracking-frame.png` — boxes + motion trails only (no tripwire)
- `analytics-frame.png` — tripwire counting line + summary panel
- `benchmark-trackers.svg` — CUDA FP16 tracker comparison (from `benchmark-cuda.csv`)
- `benchmark-cuda.csv` — CUDA latency card (trackers × half, plus pretrained `yolo11s.pt` contrast)
- `benchmark.csv` — legacy CPU snapshot (kept for historical regression experiments)
- `social-preview.png` — repository social preview image

The short demo video is generated locally as `docs/media/demo-short.mp4` and is **not** committed. Upload it as a GitHub release asset under the `media` release so the README hero link resolves at:

`https://github.com/h01t/vtrack/releases/download/media/demo-short.mp4`

## Provenance (honest)

The demo source is **continuous highway traffic** from Roboflow's supervision example [`vehicles.mp4`](https://media.roboflow.com/supervision/video-examples/vehicles.mp4), downscaled to 1280×720 at `/srv/ai/datasets/vtrack/samples/traffic-highway.mp4`. Tracking and the counting line need temporal continuity — do **not** stitch KITTI validation PNGs (independent scenes → slideshow).

The white **counting line** is a `supervision.LineZone` tripwire: when a tracked object's centroid crosses it, the analytics panel increments **in** or **out** (direction of travel across the line). Unique class counters count distinct track IDs over the clip.

Prepare / refresh the source:

```bash
# first time: download + downscale (optional if traffic-highway.mp4 already exists)
curl -L -o /srv/ai/datasets/vtrack/samples/vehicles-roboflow.mp4 \
  https://media.roboflow.com/supervision/video-examples/vehicles.mp4
ffmpeg -y -i /srv/ai/datasets/vtrack/samples/vehicles-roboflow.mp4 \
  -vf scale=1280:-2 -an -c:v libx264 -crf 23 \
  /srv/ai/datasets/vtrack/samples/traffic-highway.mp4

uv run python tasks/prepare_demo_source.py
# writes /srv/ai/datasets/vtrack/samples/demo-source.mp4 and links data/test-video.mp4
```

## CUDA performance card

```bash
# GPU free? nvidia-smi
uv run python tasks/cuda_perf_card.py \
  --source data/test-video.mp4 \
  --model models/best.pt \
  --max-frames 150
# writes docs/media/benchmark-cuda.csv
```

Render the SVG from FP16 `best.pt` rows (after regenerating the CSV):

```bash
uv run python - <<'PY'
import csv
from vtrack.readme_media import BenchmarkRow, render_benchmark_svg
rows = []
with open("docs/media/benchmark-cuda.csv", newline="", encoding="utf-8") as handle:
    for row in csv.DictReader(handle):
        if row["model"] != "best.pt" or str(row["half"]).lower() not in {"true", "1"}:
            continue
        rows.append(
            BenchmarkRow(
                tracker=row["tracker"],
                avg_fps=float(row["avg_fps"]),
                avg_track_duration_frames=float(row["avg_track_duration_frames"]),
                short_tracks_lt_5_frames=int(float(row["short_tracks_lt_5_frames"])),
            )
        )
render_benchmark_svg(
    rows,
    "docs/media/benchmark-trackers.svg",
    title="CUDA Tracker Benchmark (FP16)",
    subtitle="model=best.pt · device=cuda · half=true · frames=150 · RTX 3060 Ti",
)
PY
```

## Rebuild stills + local demo MP4

```bash
.venv/bin/python tasks/build_readme_media.py \
  --source data/test-video.mp4 \
  --model models/best.pt \
  --out-dir docs/media \
  --segment-start-sec 0 \
  --segment-duration-sec 15 \
  --scene-crop-top-ratio 1.0
```

Note: `build_readme_media.py` currently rewrites `benchmark-trackers.svg` from a CPU benchmark pass. Re-run the CUDA SVG snippet above afterward so the README chart stays CUDA-primary.

## Release + GitHub setup

1. Rebuild clip + media locally; review `docs/media/demo-short.mp4`.
2. Create or update GitHub release tagged `media`.
3. Upload `docs/media/demo-short.mp4`.
4. Verify the README hero poster opens the uploaded MP4 without authentication.
5. Set the repository social preview image to `docs/media/social-preview.png`.
