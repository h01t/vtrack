# Vehicle Detection + Tracking Pipeline — TODO

## Phase 1: Scaffolding + First Detection
- [x] Initialize git repo and uv project
- [x] Create directory structure (src/vtrack, scripts, configs, data, models, outputs, tests)
- [x] Write pyproject.toml with core dependencies
- [x] Write .gitignore
- [x] Install dependencies (uv sync)
- [x] Create VehicleDetector class (src/vtrack/detect.py)
- [x] Create config module (src/vtrack/config.py)
- [x] Create smoke test script (scripts/detect_image.py)
- [x] Verify detection on sample image (bus.jpg → 1 bus at 94%)
- [x] Verify class filtering (zidane.jpg → 0 vehicles)
- [x] Initial git commit

## Phase 2: Dataset + Fine-Tuning
- [x] Choose and download vehicle dataset → KITTI (auto-download via Ultralytics, 390MB)
- [x] Baseline metrics with pretrained YOLOv11n (mAP@0.5: 0.022 — expected, COCO/KITTI class mismatch)
- [x] Create scripts/train.py + scripts/evaluate.py
- [x] Fix SSH key auth to CUDA workstation (blackbox, 192.168.1.100)
- [x] Set up remote CUDA training (3060 Ti, scripts/train_remote.sh)
- [ ] Fine-tune YOLOv11n on KITTI (50 epochs, CUDA) — **IN PROGRESS: epoch ~27/50, mAP@0.5 = 0.779**
- [ ] Pull trained weights to models/ and run final evaluation

## Phase 3: Multi-Object Tracking
- [x] Create src/vtrack/track.py (VehicleTracker with ByteTrack via Ultralytics model.track())
- [x] Create src/vtrack/visualize.py (boxes, IDs, trails, FPS counter via supervision)
- [x] Create scripts/detect_video.py
- [x] Test on KITTI clip — 18 unique vehicle tracks across 150 frames
- [x] Verify persistent track IDs across frames
- [ ] Tune track_buffer for occlusion handling (needs real continuous video)

## Phase 4: Real-Time Inference Pipeline
- [x] Create src/vtrack/pipeline.py (orchestrator)
- [x] Support video files, webcam, RTSP, YouTube
- [x] Create scripts/demo.py (CLI with argparse)
- [x] Save annotated output video
- [ ] FPS benchmarks (YOLOv11n, YOLOv11s on M4 Pro) — needs real video

## Phase 5: Analytics Layer
- [x] Create src/vtrack/analytics.py
- [x] Line-crossing counter (supervision.LineZone)
- [x] Polygon zone monitoring (supervision.PolygonZone)
- [x] Per-class vehicle counts
- [x] Track duration stats
- [x] Export to CSV/JSON
- [x] Analytics overlay on video
- [x] Integrate analytics into pipeline and demo CLI

## Phase 6: Edge Deployment (Future)
- [ ] Export to ONNX
- [ ] Export to CoreML
- [ ] Benchmark ONNX Runtime vs PyTorch MPS vs CoreML
- [ ] Test on Raspberry Pi (TFLite/NCNN)
- [ ] Profile FPS and memory on constrained hardware
