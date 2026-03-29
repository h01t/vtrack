# README Media Assets

This directory holds the lightweight, tracked assets used by the public repository README:

- `hero-poster.png` — poster frame linked from the README hero section
- `tracking-frame.png` — still image showing persistent IDs and trails
- `analytics-frame.png` — still image showing the line-counter overlay and analytics panel
- `benchmark-trackers.svg` — tracker comparison chart rendered from `benchmark.csv`
- `benchmark.csv` — snapshot data produced by `vtrack benchmark-track`
- `social-preview.png` — image intended for the repository social preview setting

The short demo video is generated locally as `docs/media/demo-short.mp4`, but it is intentionally not tracked in git. Upload that file as a GitHub release asset under the `media` release so the README hero link resolves at:

`https://github.com/h01t/vtrack/releases/download/media/demo-short.mp4`

## Current Snapshot

- Model: `models/best.pt`
- Benchmark device: `cpu`
- Benchmark frames: `150`
- Trackers: `bytetrack`, `bytetrack-occlusion`, `botsort`
- Local source clip used for the current checked-in snapshot: `data/test-video.mp4`

The checked-in stills were generated from the bundled local smoke-test clip so the README structure can be reviewed end-to-end. Before refreshing the public-facing release asset, rerun the script on an original or permissively licensed traffic clip and upload the regenerated `demo-short.mp4`.

## Rebuild Command

```bash
.venv/bin/python tasks/build_readme_media.py \
  --source data/test-video.mp4 \
  --model models/best.pt \
  --out-dir docs/media \
  --segment-start-sec 3 \
  --segment-duration-sec 12 \
  --scene-crop-top-ratio 0.67
```

The crop ratio above trims the local smoke-test clip's footer before the script composes the final 16:9 demo video and still images. For a clean public source clip, use the default crop ratio of `1.0`.

## Release + GitHub Setup

1. Run the rebuild command and review `docs/media/demo-short.mp4` locally.
2. Create or update a lightweight GitHub release tagged `media`.
3. Upload `docs/media/demo-short.mp4` to that release.
4. Verify the README hero poster opens the uploaded MP4 without authentication.
5. Set the repository social preview image to `docs/media/social-preview.png`.

## Benchmark Context

`benchmark-trackers.svg` is a reproducible snapshot generated from `benchmark.csv`, not a general leaderboard. The chart compares the three bundled tracker presets on the same short clip using the fine-tuned model and CPU inference for repeatability.
