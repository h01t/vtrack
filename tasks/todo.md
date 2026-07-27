# Vehicle Detection + Tracking Pipeline — TODO

## Completed foundation
- [x] Scaffolding, KITTI fine-tune on blackbox CUDA, tracking, analytics, artifacts
- [x] Blackbox-native paths (`/srv/ai`), CUDA train defaults
- [x] Portfolio polish track: CUDA perf card, ONNX export, CI, honest docs
- [x] Fixed-camera continuous demo source (Roboflow highway; not KITTI stills)
- [x] Localhost FastAPI detect/track API (`vtrack serve`)
- [x] Formal MOT17 HOTA/MOTA/IDF1 via TrackEval (MOT17-02)

## Active / portfolio polish
- [x] CUDA tracker × half latency card on 3060 Ti
- [x] ONNX export + `.pt` vs ORT compare
- [x] GitHub Actions (ruff + non-smoke pytest)
- [ ] Refresh GitHub `media` release asset after local demo rebuild (manual)
- [ ] Expand MOT17 coverage to 04/09 when HF download completes

## Deferred
- [ ] TensorRT / Raspberry Pi / INT8 edge bring-up
- [ ] Tune occlusion presets on longer multi-minute clips
