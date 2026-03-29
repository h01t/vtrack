"""End-to-end vehicle detection + tracking + visualization pipeline."""

from pathlib import Path

import cv2
import supervision as sv

from vtrack.analytics import VehicleAnalytics
from vtrack.config import DEFAULT_CONFIDENCE, DEFAULT_MODEL, DEFAULT_TRACKER
from vtrack.track import VehicleTracker
from vtrack.visualize import (
    Visualizer,
    filter_detections_by_confidence,
    ultralytics_to_detections,
)


class VehiclePipeline:
    """Orchestrate detection, tracking, visualization, analytics, and output."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        confidence: float = DEFAULT_CONFIDENCE,
        track_conf: float | None = None,
        tracker: str = DEFAULT_TRACKER,
        trace_length: int = 30,
        analytics: VehicleAnalytics | None = None,
        device: str | None = None,
        imgsz: int = 640,
        iou: float = 0.7,
        max_det: int = 300,
        half: bool = False,
        vid_stride: int = 1,
        stream_buffer: bool = False,
        agnostic_nms: bool = False,
    ):
        self.min_confidence = confidence
        self.tracker = VehicleTracker(
            model_path=model_path,
            track_conf=track_conf,
            tracker=tracker,
            device=device,
            imgsz=imgsz,
            iou=iou,
            max_det=max_det,
            half=half,
            vid_stride=vid_stride,
            stream_buffer=stream_buffer,
            agnostic_nms=agnostic_nms,
        )
        class_names = self.tracker.class_names
        self.visualizer = Visualizer(trace_length=trace_length, class_names=class_names)
        self.analytics = analytics
        self.line_zone_annotator = None
        self.polygon_zone_annotator = None
        if self.analytics:
            self.analytics.class_names = class_names
            if self.analytics.line_zone is not None:
                self.line_zone_annotator = sv.LineZoneAnnotator(thickness=2, text_scale=0.5)
            if self.analytics.polygon_zone is not None:
                self.polygon_zone_annotator = sv.PolygonZoneAnnotator(
                    zone=self.analytics.polygon_zone,
                    thickness=2,
                    text_scale=0.5,
                )

    def run(
        self,
        source: str | Path | int,
        display: bool = True,
        save_path: str | None = None,
        export_csv: str | None = None,
        export_json: str | None = None,
    ):
        """Process a video source with tracking, visualization, and analytics.

        Args:
            source: Video file, camera index (0), RTSP URL, or YouTube URL.
            display: Show live OpenCV window.
            save_path: If set, save annotated video to this path.
            export_csv: If set, export per-frame data to this CSV path.
            export_json: If set, export summary to this JSON path.
        """
        writer = None
        frame_count = 0

        tracker_describer = getattr(self.tracker, "describe_tracker", None)
        if callable(tracker_describer):
            print(f"Using tracker: {tracker_describer()}")

        try:
            for result in self.tracker.track(source):
                frame = result.orig_img
                detections = filter_detections_by_confidence(
                    ultralytics_to_detections(result),
                    self.min_confidence,
                )

                # Update analytics
                if self.analytics:
                    self.analytics.update(detections)

                # Annotate frame with tracking overlays
                annotated = self.visualizer.annotate(frame, detections)

                # Annotate frame with analytics overlay
                if self.analytics:
                    annotated = self.analytics.annotate(annotated)

                    if self.line_zone_annotator is not None:
                        annotated = self.line_zone_annotator.annotate(
                            annotated,
                            self.analytics.line_zone,
                        )
                    if self.polygon_zone_annotator is not None:
                        annotated = self.polygon_zone_annotator.annotate(annotated)

                frame_count += 1

                # Initialize video writer on first frame
                if save_path and writer is None:
                    h, w = annotated.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                    fps = self._get_source_fps(source, result)
                    writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

                if writer:
                    writer.write(annotated)

                if display:
                    cv2.imshow("Vehicle Tracking", annotated)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break

        finally:
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()

        print(f"Processed {frame_count} frames.")
        if save_path:
            print(f"Saved to {save_path}")

        # Export analytics
        if self.analytics:
            summary = self.analytics.get_summary()
            print("\n--- Analytics Summary ---")
            print(f"Unique vehicles: {summary['unique_vehicles']}")
            for cls_name, count in summary['per_class_counts'].items():
                print(f"  {cls_name}: {count}")
            if summary['line_crossings_in'] or summary['line_crossings_out']:
                print(
                    "Line crossings — "
                    f"In: {summary['line_crossings_in']}, "
                    f"Out: {summary['line_crossings_out']}"
                )
            print(f"Avg track duration: {summary['avg_track_duration_frames']:.1f} frames")

            if export_csv:
                self.analytics.export_csv(export_csv)
            if export_json:
                self.analytics.export_json(export_json)

    def _get_source_fps(self, source, result) -> float:
        """Try to get FPS from the source, default to 30."""
        if isinstance(source, (str, Path)) and Path(str(source)).exists():
            cap = cv2.VideoCapture(str(source))
            fps = cap.get(cv2.CAP_PROP_FPS)
            cap.release()
            if fps > 0:
                return fps
        return 30.0
