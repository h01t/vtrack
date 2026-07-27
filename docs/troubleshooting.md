# Troubleshooting

This guide covers common local runtime failures and operational recovery paths after local hardening work.

## 1) Inference device errors

### Symptom
- CLI exits with an error similar to:
  - `MPS inference requested ... is_available() returned False`
  - `CUDA inference requested ... torch.cuda.is_available() is False`

### Cause
- Selected device is unavailable on current machine/runtime.

### Resolution
- On blackbox, prefer `--device cuda` after checking `nvidia-smi`.
- Explicitly switch to a supported device:
  - `--device cuda` (NVIDIA hosts; default for `vtrack train`)
  - `--device cpu`
  - `--device mps` (Apple Silicon inference only when available)

### Verify
```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)"
uv run vtrack train --help   # should list cuda as the device default
```

---

## 2) Source validation errors

### Symptom
- Source-related failure before runtime starts.
- Examples:
  - missing local file path
  - invalid webcam index

### Cause
- Runtime validation rejects unsupported/invalid source input.

### Resolution
- For files: verify path exists and is readable.
- For webcam: pass non-negative integer source index.
- For streams/URLs: confirm syntax and network reachability.

### Verify
```bash
uv run vtrack demo data/test-video.mp4 --no-display --device cpu
```

---

## 3) Model load failures

### Symptom
- Model initialization fails at command startup.
- Error may be wrapped as `ModelLoadError`.

### Cause
- Invalid weights path, unreadable file, incompatible model artifact, or dependency/runtime issue.

### Resolution
- Confirm `--model` points to a valid model file.
- Try baseline model:
```bash
uv run vtrack detect-image --model yolo11n.pt --device cpu --no-show
```
- If custom model fails but baseline works, re-export/retrain custom artifact.

---

## 4) Runtime pipeline failures

### Symptom
- Processing starts, then exits with wrapped runtime error (`PipelineRuntimeError`).

### Cause
- Downstream model/IO/codec/runtime exception during frame iteration.

### Resolution
- Retry with:
  - smaller `--imgsz`
  - `--device cpu`
  - local test media (`data/test-video.mp4`)
- Disable optional outputs while isolating:
  - no save/export options first
- Re-run with `VTRACK_LOG_LEVEL=DEBUG` for richer context:
```bash
VTRACK_LOG_LEVEL=DEBUG uv run vtrack detect-video data/test-video.mp4 --device cpu
```

---

## 5) GPU OOM / crowded VRAM on blackbox

### Symptom
- Training or demo fails with CUDA OOM, or runs extremely slowly while another process owns the GPU.

### Cause
- RTX 3060 Ti has 8 GiB. Heavy neighbors (NLLB, large ComfyUI graphs, vLLM) can leave too little free memory.

### Resolution
- Run `nvidia-smi` and stop/pause the other GPU owner for the train window.
- Lower `--batch` (16 → 8 → 4) before changing `--imgsz`.
- Keep AMP enabled (do not pass `--no-amp` unless debugging).

### Verify
```bash
nvidia-smi
# expect several GiB free before `vtrack train --device cuda`
```

---

## 6) Dataset path / unexpected KITTI re-download

### Symptom
- Ultralytics starts downloading `kitti.zip` even though KITTI already exists under `/srv/ai/datasets/kitti`.

### Cause
- Relative `path: kitti` in a yaml combined with the wrong Ultralytics `datasets_dir`.

### Resolution
- Train/eval with `configs/kitti.yaml` (absolute `path: /srv/ai/datasets/kitti`).
- Or set `VTRACK_KITTI_YAML` / `VTRACK_DATASETS_DIR` from `.env.example`.

### Verify
```bash
uv run python -c "from vtrack.settings import resolve_dataset_config; print(resolve_dataset_config('kitti.yaml'))"
```

---

## 7) Remote training command failures (legacy)

### Symptom
- `train-remote` fails during command execution/sync.

### Cause
- SSH connectivity, remote env mismatch, remote path mismatch, or command failure. Prefer local `vtrack train` on blackbox.

### Resolution
- Verify remote host and directory flags/env:
  - `VTRACK_REMOTE_HOST`
  - `VTRACK_REMOTE_DIR`
  - `VTRACK_REMOTE_DATASETS_DIR`
  - `VTRACK_REMOTE_PYTHON`
- Dry-run with help:
```bash
bash scripts/train_remote.sh --help
```
- Ensure remote Python and dependencies are installed.

---

## 8) Benchmark regression checks fail

### Symptom
- Regression script exits non-zero with throughput/latency threshold failures.

### Cause
- Current benchmark degraded against baseline beyond configured thresholds.

### Resolution
- Re-run benchmark with consistent environment/model/source.
- Compare CSV schema and tracker naming alignment.
- Temporarily relax thresholds only with documented rationale.

### Run checker
```bash
uv run python tasks/benchmark_regression.py \
  --current docs/media/benchmark.csv \
  --baseline docs/media/benchmark.csv
```

---

## 9) Lint/test regressions after refactors

### Symptom
- Ruff import-order failures or test import errors in CLI modules.

### Cause
- Modularized CLI moved symbols; compatibility aliases may be required.

### Resolution
- Run:
```bash
uv run ruff check src tests --fix
uv run pytest -q
```
- Ensure `src/vtrack/cli.py` keeps compatibility exports required by tests/wrappers.

---

## Logging tips

Two env vars are supported:

- `VTRACK_LOG_LEVEL` (e.g. `DEBUG`, `INFO`, `WARNING`)
- `VTRACK_LOG_FORMAT` (`text` or `json`)

Example:
```bash
VTRACK_LOG_LEVEL=DEBUG VTRACK_LOG_FORMAT=json uv run vtrack demo data/test-video.mp4 --no-display --device cpu

---

## 11) Localhost API failures

### Symptom
- `uv run vtrack serve` fails with missing FastAPI/uvicorn, or curl cannot connect.

### Cause
- API extra not installed, non-loopback bind rejected, or port already in use.

### Resolution
```bash
uv sync --extra api
uv run vtrack serve --host 127.0.0.1 --port 8000 --model models/best.pt --device cuda
curl -s http://127.0.0.1:8000/health
```
- Do not bind `0.0.0.0` — the PoC API refuses non-loopback hosts.

---

## 12) MOT17 / TrackEval failures

### Symptom
- `evaluate-mot` cannot find sequences or TrackEval import fails.

### Cause
- Dataset missing under `/srv/ai/datasets/mot17`, or `mot` extra not installed.

### Resolution
```bash
uv sync --extra mot
uv run python tasks/download_mot17.py --keep-only-poc
uv run vtrack evaluate-mot --sequences MOT17-02 --model yolo11n.pt --device cuda
```
- MOT numbers use COCO-pretrained `yolo11n.pt` on **person** tracks (MOTChallenge domain), not KITTI `best.pt`.
- Prefer the Hugging Face mirror in `tasks/download_mot17.py` when motchallenge.net is unreachable.
