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
- [ ] Choose and download vehicle dataset (UA-DETRAC / Roboflow / COCO subset)
- [ ] Convert to YOLO format if needed
- [ ] Create configs/dataset.yaml
- [ ] Baseline metrics with pretrained YOLOv11n
- [ ] Create scripts/train.py
- [ ] Fine-tune YOLOv11n (device=mps, 50 epochs)
- [ ] Create scripts/evaluate.py
- [ ] Evaluate: target mAP@0.5 >= 0.7
- [ ] Save best weights to models/

## Phase 3: Multi-Object Tracking
- [ ] Create src/vtrack/track.py (VehicleTracker with ByteTrack)
- [ ] Create configs/bytetrack.yaml with tuned params
- [ ] Create src/vtrack/visualize.py (boxes, IDs, trails, FPS)
- [ ] Create scripts/detect_video.py
- [ ] Test on sample traffic video
- [ ] Verify persistent track IDs across frames
- [ ] Tune track_buffer for occlusion handling

## Phase 4: Real-Time Inference Pipeline
- [ ] Create src/vtrack/pipeline.py (orchestrator)
- [ ] Support video files, webcam, RTSP, YouTube
- [ ] Create scripts/demo.py (CLI with argparse)
- [ ] FPS benchmarks (YOLOv11n, YOLOv11s on M4 Pro)
- [ ] Save annotated output video

## Phase 5: Analytics Layer
- [ ] Create src/vtrack/analytics.py
- [ ] Line-crossing counter (supervision.LineZone)
- [ ] Polygon zone monitoring (supervision.PolygonZone)
- [ ] Per-class vehicle counts
- [ ] Track duration stats
- [ ] Export to CSV/JSON
- [ ] Analytics overlay on video

## Phase 6: Edge Deployment (Future)
- [ ] Export to ONNX
- [ ] Export to CoreML
- [ ] Benchmark ONNX Runtime vs PyTorch MPS vs CoreML
- [ ] Test on Raspberry Pi (TFLite/NCNN)
- [ ] Profile FPS and memory on constrained hardware
