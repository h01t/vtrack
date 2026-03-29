"""Visualization overlays for detection and tracking results."""

import time

import cv2
import numpy as np
import supervision as sv

from vtrack.config import COCO_VEHICLE_NAMES


class Visualizer:
    """Draw bounding boxes, track IDs, trails, and FPS on video frames."""

    def __init__(self, trace_length: int = 30, class_names: dict[int, str] | None = None):
        self.class_names = class_names or COCO_VEHICLE_NAMES
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.label_annotator = sv.LabelAnnotator(text_scale=0.5, text_padding=5)
        self.trace_annotator = sv.TraceAnnotator(
            thickness=2, trace_length=trace_length
        )

        # FPS tracking
        self._prev_time = time.time()
        self._fps = 0.0
        self._fps_smooth = 0.8  # exponential moving average factor

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        """Draw all overlays on a frame.

        Args:
            frame: BGR image (from OpenCV).
            detections: supervision Detections object with tracker_id set.

        Returns:
            Annotated BGR frame.
        """
        frame = frame.copy()

        # Draw traces only if tracker IDs are present
        has_tracks = detections.tracker_id is not None and len(detections) > 0
        if has_tracks:
            frame = self.trace_annotator.annotate(frame, detections)

        # Draw bounding boxes
        frame = self.box_annotator.annotate(frame, detections)

        # Build labels: "car #3 0.85"
        labels = self._build_labels(detections)
        frame = self.label_annotator.annotate(frame, detections, labels=labels)

        # Draw FPS counter
        frame = self._draw_fps(frame)

        return frame

    def _build_labels(self, detections: sv.Detections) -> list[str]:
        """Build label strings from detections."""
        labels = []
        for i in range(len(detections)):
            cls_id = int(detections.class_id[i]) if detections.class_id is not None else -1
            conf = float(detections.confidence[i]) if detections.confidence is not None else 0.0
            track_id = int(detections.tracker_id[i]) if detections.tracker_id is not None else -1

            name = self.class_names.get(cls_id, f"cls_{cls_id}")
            if track_id >= 0:
                labels.append(f"{name} #{track_id} {conf:.2f}")
            else:
                labels.append(f"{name} {conf:.2f}")
        return labels

    def _draw_fps(self, frame: np.ndarray) -> np.ndarray:
        """Draw FPS counter in top-left corner."""
        now = time.time()
        dt = now - self._prev_time
        self._prev_time = now

        if dt > 0:
            instant_fps = 1.0 / dt
            self._fps = self._fps_smooth * self._fps + (1 - self._fps_smooth) * instant_fps

        text = f"FPS: {self._fps:.0f}"
        cv2.putText(
            frame, text, (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2,
        )
        return frame


def ultralytics_to_detections(result) -> sv.Detections:
    """Convert an Ultralytics result to a supervision Detections object."""
    return sv.Detections.from_ultralytics(result)


def filter_detections_by_confidence(
    detections: sv.Detections,
    min_confidence: float | None,
) -> sv.Detections:
    """Filter detections for downstream overlays and analytics."""
    if min_confidence is None or len(detections) == 0 or detections.confidence is None:
        return detections

    mask = detections.confidence >= float(min_confidence)
    return detections[mask]
