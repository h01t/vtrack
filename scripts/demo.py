"""Full pipeline demo: detect + track + visualize + analytics."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import supervision as sv
from vtrack.analytics import VehicleAnalytics
from vtrack.pipeline import VehiclePipeline


def parse_line(value: str) -> tuple[sv.Point, sv.Point]:
    """Parse 'x1,y1,x2,y2' into two supervision Points."""
    coords = [int(x) for x in value.split(",")]
    if len(coords) != 4:
        raise argparse.ArgumentTypeError("Line must be x1,y1,x2,y2")
    return sv.Point(coords[0], coords[1]), sv.Point(coords[2], coords[3])


def parse_polygon(value: str) -> np.ndarray:
    """Parse 'x1,y1,x2,y2,...' into a polygon array."""
    coords = [int(x) for x in value.split(",")]
    if len(coords) % 2 != 0 or len(coords) < 6:
        raise argparse.ArgumentTypeError("Polygon needs at least 3 points: x1,y1,x2,y2,x3,y3")
    return np.array(coords).reshape(-1, 2)


def main():
    parser = argparse.ArgumentParser(description="Vehicle detection + tracking + analytics demo")
    parser.add_argument("source", help="Video file, camera index (0), RTSP URL, or YouTube URL")

    # Model options
    parser.add_argument("--model", default="yolo11n.pt", help="Model weights path")
    parser.add_argument("--confidence", type=float, default=0.25, help="Detection confidence")
    parser.add_argument("--tracker", default="bytetrack.yaml", help="Tracker config")
    parser.add_argument("--trace-length", type=int, default=30, help="Trail length in frames")

    # Display/save
    parser.add_argument("--save", default=None, help="Output video path (e.g. outputs/demo.mp4)")
    parser.add_argument("--no-display", action="store_true", help="Disable live display window")

    # Analytics
    parser.add_argument("--analytics", action="store_true", help="Enable vehicle analytics")
    parser.add_argument("--line", type=str, default=None,
                        help="Counting line as x1,y1,x2,y2 (e.g. 0,200,640,200)")
    parser.add_argument("--zone", type=str, default=None,
                        help="Monitoring zone polygon as x1,y1,x2,y2,x3,y3,... (min 3 points)")
    parser.add_argument("--export-csv", default=None, help="Export per-frame data to CSV")
    parser.add_argument("--export-json", default=None, help="Export summary to JSON")

    args = parser.parse_args()

    # Handle camera index
    source = int(args.source) if args.source.isdigit() else args.source

    # Set up analytics if requested
    analytics = None
    if args.analytics or args.line or args.zone or args.export_csv or args.export_json:
        line_zone = None
        polygon_zone = None

        if args.line:
            start, end = parse_line(args.line)
            line_zone = sv.LineZone(start=start, end=end)

        if args.zone:
            polygon = parse_polygon(args.zone)
            polygon_zone = sv.PolygonZone(polygon=polygon)

        analytics = VehicleAnalytics(line_zone=line_zone, polygon_zone=polygon_zone)

    pipeline = VehiclePipeline(
        model_path=args.model,
        confidence=args.confidence,
        tracker=args.tracker,
        trace_length=args.trace_length,
        analytics=analytics,
    )

    pipeline.run(
        source=source,
        display=not args.no_display,
        save_path=args.save,
        export_csv=args.export_csv,
        export_json=args.export_json,
    )


if __name__ == "__main__":
    main()
