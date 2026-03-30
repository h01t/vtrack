# Troubleshooting

This guide covers common local runtime failures and operational recovery paths after local hardening work.

## 1) Inference device errors

### Symptom
- CLI exits with an error similar to:
  - `MPS inference requested ... is_available() returned False`

### Cause
- Selected device is unavailable on current machine/runtime.

### Resolution
- Explicitly switch to a supported device:
  - `--device cpu`
  - `--device mps` (Apple Silicon only when available)
  - `--device cuda` (NVIDIA-only hosts)

### Verify
```bash
uv run vtrack detect-image --device cpu --help
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

## 5) Remote training command failures

### Symptom
- `train-remote` fails during command execution/sync.

### Cause
- SSH connectivity, remote env mismatch, remote path mismatch, or command failure.

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

## 6) Benchmark regression checks fail

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

## 7) Lint/test regressions after refactors

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
