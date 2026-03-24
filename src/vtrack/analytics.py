"""Vehicle analytics: counting, zone monitoring, track statistics."""

import json
import csv
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import supervision as sv

from vtrack.config import COCO_VEHICLE_NAMES


@dataclass
class TrackRecord:
    """Record for a single tracked vehicle."""

    track_id: int
    class_id: int
    first_frame: int
    last_frame: int
    positions: list = field(default_factory=list)
    class_names: dict = field(default_factory=lambda: COCO_VEHICLE_NAMES)

    @property
    def duration_frames(self) -> int:
        return self.last_frame - self.first_frame + 1

    @property
    def class_name(self) -> str:
        return self.class_names.get(self.class_id, f"cls_{self.class_id}")


class VehicleAnalytics:
    """Track counts, durations, and zone events."""

    def __init__(
        self,
        line_zone: sv.LineZone | None = None,
        polygon_zone: sv.PolygonZone | None = None,
        class_names: dict[int, str] | None = None,
    ):
        self.class_names = class_names or COCO_VEHICLE_NAMES
        self.line_zone = line_zone
        self.polygon_zone = polygon_zone

        # Per-frame data
        self.frame_log: list[dict] = []
        self.frame_count = 0

        # Track history
        self.tracks: dict[int, TrackRecord] = {}

        # Counting
        self.class_counts = Counter()  # unique vehicles per class
        self.line_in_count = 0
        self.line_out_count = 0
        self.zone_current_count = 0

    def update(self, detections: sv.Detections):
        """Process one frame of detections. Call once per frame."""
        self.frame_count += 1

        # Update track records
        if detections.tracker_id is not None:
            for i in range(len(detections)):
                tid = int(detections.tracker_id[i])
                cls_id = int(detections.class_id[i]) if detections.class_id is not None else -1
                bbox = detections.xyxy[i].tolist()
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2

                if tid not in self.tracks:
                    self.tracks[tid] = TrackRecord(
                        track_id=tid,
                        class_id=cls_id,
                        first_frame=self.frame_count,
                        last_frame=self.frame_count,
                        class_names=self.class_names,
                    )
                    self.class_counts[cls_id] += 1

                track = self.tracks[tid]
                track.last_frame = self.frame_count
                track.positions.append((cx, cy))

        # Line crossing (requires tracker_id)
        has_tracks = detections.tracker_id is not None and len(detections) > 0
        if self.line_zone is not None and has_tracks:
            crossed_in, crossed_out = self.line_zone.trigger(detections)
            self.line_in_count = self.line_zone.in_count
            self.line_out_count = self.line_zone.out_count

        # Zone occupancy
        if self.polygon_zone is not None and has_tracks:
            in_zone = self.polygon_zone.trigger(detections)
            self.zone_current_count = int(np.sum(in_zone))

        # Log frame data
        self.frame_log.append({
            "frame": self.frame_count,
            "detections": len(detections),
            "line_in": self.line_in_count,
            "line_out": self.line_out_count,
            "zone_count": self.zone_current_count,
        })

    def get_summary(self) -> dict:
        """Get summary statistics."""
        durations = [t.duration_frames for t in self.tracks.values()]
        per_class = {}
        for cls_id, count in self.class_counts.items():
            name = self.class_names.get(cls_id, f"cls_{cls_id}")
            per_class[name] = count

        return {
            "total_frames": self.frame_count,
            "unique_vehicles": len(self.tracks),
            "per_class_counts": per_class,
            "line_crossings_in": self.line_in_count,
            "line_crossings_out": self.line_out_count,
            "avg_track_duration_frames": float(np.mean(durations)) if durations else 0,
            "max_track_duration_frames": max(durations) if durations else 0,
            "min_track_duration_frames": min(durations) if durations else 0,
        }

    def annotate(self, frame: np.ndarray) -> np.ndarray:
        """Draw analytics overlay on frame."""
        h, w = frame.shape[:2]

        # Draw stats panel in top-right corner
        panel_lines = [
            f"Vehicles: {len(self.tracks)}",
        ]

        # Per-class breakdown
        for cls_id, count in sorted(self.class_counts.items()):
            name = self.class_names.get(cls_id, f"cls_{cls_id}")
            panel_lines.append(f"  {name}: {count}")

        if self.line_zone is not None:
            panel_lines.append(f"In: {self.line_in_count} | Out: {self.line_out_count}")

        if self.polygon_zone is not None:
            panel_lines.append(f"In zone: {self.zone_current_count}")

        # Draw panel background
        line_height = 25
        panel_h = len(panel_lines) * line_height + 10
        panel_w = 200
        x0 = w - panel_w - 10
        y0 = 10

        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x0 + panel_w, y0 + panel_h), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)

        for i, line in enumerate(panel_lines):
            y = y0 + 20 + i * line_height
            cv2.putText(frame, line, (x0 + 10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame

    def export_csv(self, path: str | Path):
        """Export per-frame data to CSV."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["frame", "detections", "line_in", "line_out", "zone_count"])
            writer.writeheader()
            writer.writerows(self.frame_log)
        print(f"Frame data exported to {path}")

    def export_json(self, path: str | Path):
        """Export summary + track details to JSON."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = self.get_summary()
        data["tracks"] = [
            {
                "track_id": t.track_id,
                "class": t.class_name,
                "first_frame": t.first_frame,
                "last_frame": t.last_frame,
                "duration_frames": t.duration_frames,
            }
            for t in self.tracks.values()
        ]

        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"Summary exported to {path}")


# Need cv2 for annotate method
import cv2
