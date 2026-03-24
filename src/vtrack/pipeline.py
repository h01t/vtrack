"""End-to-end vehicle detection + tracking + visualization pipeline."""

from pathlib import Path

import cv2

from vtrack.analytics import VehicleAnalytics
from vtrack.config import DEFAULT_CONFIDENCE, DEFAULT_MODEL
from vtrack.track import VehicleTracker
from vtrack.visualize import Visualizer, ultralytics_to_detections


class VehiclePipeline:
    """Orchestrate detection, tracking, visualization, analytics, and output."""

    def __init__(
        self,
        model_path: str = DEFAULT_MODEL,
        confidence: float = DEFAULT_CONFIDENCE,
        tracker: str = "bytetrack.yaml",
        trace_length: int = 30,
        analytics: VehicleAnalytics | None = None,
    ):
        self.tracker = VehicleTracker(
            model_path=model_path,
            confidence=confidence,
            tracker=tracker,
        )
        self.visualizer = Visualizer(trace_length=trace_length)
        self.analytics = analytics

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

        try:
            for result in self.tracker.track(source):
                frame = result.orig_img
                detections = ultralytics_to_detections(result)

                # Update analytics
                if self.analytics:
                    self.analytics.update(detections)

                # Annotate frame with tracking overlays
                annotated = self.visualizer.annotate(frame, detections)

                # Annotate frame with analytics overlay
                if self.analytics:
                    annotated = self.analytics.annotate(annotated)

                    # Draw line/zone annotations if present
                    if self.analytics.line_zone is not None:
                        line_annotator = cv2.LINE_AA  # placeholder — use sv annotator
                        from supervision import LineZoneAnnotator
                        lza = LineZoneAnnotator(thickness=2, text_scale=0.5)
                        annotated = lza.annotate(annotated, self.analytics.line_zone)

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
            print(f"\n--- Analytics Summary ---")
            print(f"Unique vehicles: {summary['unique_vehicles']}")
            for cls_name, count in summary['per_class_counts'].items():
                print(f"  {cls_name}: {count}")
            if summary['line_crossings_in'] or summary['line_crossings_out']:
                print(f"Line crossings — In: {summary['line_crossings_in']}, Out: {summary['line_crossings_out']}")
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
