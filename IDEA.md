### 8. Object Detection + Tracking Pipeline

**Goal:** Build an end-to-end video understanding system using a custom-collected dataset, bridging existing computer vision skills toward video and real-time tracking.

**Outline:**
- **Use case** — Pick a concrete detection scenario: pedestrian detection in hallway footage (reuse autonomous vehicle camera), vehicle counting, or a custom object (tools, food items). Collect 500–1,000 annotated frames using CVAT or Label Studio.
- **Detection model** — Fine-tune YOLOv8 or YOLOv11 on the custom dataset using Ultralytics. Validate with mAP@0.5 and mAP@0.5:0.95 metrics.
- **Tracking integration** — Add ByteTrack or BoT-SORT for multi-object tracking across frames. Assign persistent IDs and draw bounding box trails.
- **Inference pipeline** — Build a real-time inference script that processes webcam or video file input. Overlay detections + track IDs + FPS counter.
- **Edge deployment** — Export the trained model to ONNX or TorchScript. Test inference on Raspberry Pi 4 (continue from thesis work). Profile FPS on CPU vs. GPU.
- **Analytics layer** — Add a simple counting/logging module: objects per frame, track duration histograms, zone entry/exit events.
- **Demo** — Annotated demo video showing real-time detections + tracks. GitHub repo with training and inference scripts.

**Key deliverables:** Custom-trained YOLOv8 model, tracking pipeline, edge deployment benchmark, annotated demo video.

**Stack:** Python, Ultralytics YOLOv8/v11, ByteTrack/BoT-SORT, OpenCV, ONNX Runtime, CVAT/Label Studio.
