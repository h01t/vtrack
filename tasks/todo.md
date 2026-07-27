# Vehicle Detection + Tracking Pipeline — TODO

## Completed foundation
- [x] Scaffolding, KITTI fine-tune on blackbox CUDA, tracking, analytics, artifacts
- [x] Blackbox-native paths (`/srv/ai`), CUDA train defaults
- [x] Portfolio polish track: CUDA perf card, ONNX export, CI, honest docs

## Active / portfolio polish
- [x] CUDA tracker × half latency card on 3060 Ti
- [x] ONNX export + `.pt` vs ORT compare
- [x] GitHub Actions (ruff + non-smoke pytest)
- [ ] Refresh GitHub `media` release asset after local demo rebuild (manual)

## Deferred
- [ ] Dashcam / fixed-camera continuous video (demo currently uses KITTI val sequence)
- [ ] Formal MOTA/HOTA/IDF1
- [ ] Localhost inference API
- [ ] TensorRT / Raspberry Pi / INT8 edge bring-up
- [ ] Tune occlusion presets on real continuous traffic video
