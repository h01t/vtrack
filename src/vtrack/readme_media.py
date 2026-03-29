"""Helpers for generating lightweight README media assets."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from xml.sax.saxutils import escape


@dataclass(frozen=True)
class BenchmarkRow:
    """Single tracker benchmark row used for README charts."""

    tracker: str
    avg_fps: float
    avg_track_duration_frames: float
    short_tracks_lt_5_frames: int


@dataclass(frozen=True)
class MetricSpec:
    """Metadata for one README benchmark metric panel."""

    key: str
    title: str
    formatter: Callable[[float], str]


METRICS = (
    MetricSpec("avg_fps", "Avg FPS", lambda value: f"{value:.1f}"),
    MetricSpec(
        "avg_track_duration_frames",
        "Avg Track Duration (frames)",
        lambda value: f"{value:.1f}",
    ),
    MetricSpec(
        "short_tracks_lt_5_frames",
        "Short Tracks <5 Frames (lower better)",
        lambda value: f"{int(value)}",
    ),
)

TRACKER_COLORS = {
    "bytetrack": "#2B6CB0",
    "bytetrack-occlusion": "#2F855A",
    "botsort": "#D69E2E",
}
FALLBACK_COLORS = ("#2B6CB0", "#2F855A", "#D69E2E", "#C05621")


def load_benchmark_rows(path: str | Path) -> list[BenchmarkRow]:
    """Load tracker benchmark rows from a CSV export."""
    csv_path = Path(path)
    rows: list[BenchmarkRow] = []

    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                BenchmarkRow(
                    tracker=row["tracker"],
                    avg_fps=float(row["avg_fps"]),
                    avg_track_duration_frames=float(row["avg_track_duration_frames"]),
                    short_tracks_lt_5_frames=int(float(row["short_tracks_lt_5_frames"])),
                )
            )

    if not rows:
        raise ValueError(f"No benchmark rows found in {csv_path}.")

    return rows


def render_benchmark_svg(
    rows: list[BenchmarkRow],
    output_path: str | Path,
    *,
    title: str = "Tracker Benchmark Snapshot",
    subtitle: str | None = None,
) -> Path:
    """Render a lightweight multi-panel SVG chart for README display."""
    if not rows:
        raise ValueError("At least one benchmark row is required to render the chart.")

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    width = 1200
    height = 430
    margin_x = 34
    gap = 18
    panel_width = int((width - (margin_x * 2) - (gap * (len(METRICS) - 1))) / len(METRICS))
    panel_height = 270
    panel_y = 112

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">',
        "<defs>",
        '<linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">',
        '<stop offset="0%" stop-color="#F8FAFC" />',
        '<stop offset="100%" stop-color="#EDF2F7" />',
        "</linearGradient>",
        '<filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">',
        (
            '<feDropShadow dx="0" dy="3" stdDeviation="6" '
            'flood-color="#0F172A" flood-opacity="0.10" />'
        ),
        "</filter>",
        "</defs>",
        '<rect width="100%" height="100%" fill="url(#bg)" rx="20" />',
        f'<title id="title">{escape(title)}</title>',
        f'<desc id="desc">{escape(subtitle or "Tracker benchmark comparison")}</desc>',
        f'<text x="{margin_x}" y="46" font-size="28" font-family="Arial, sans-serif" '
        'font-weight="700" fill="#0F172A">'
        f"{escape(title)}</text>",
    ]

    if subtitle:
        parts.append(
            f'<text x="{margin_x}" y="74" font-size="14" font-family="Arial, sans-serif" '
            'fill="#475569">'
            f"{escape(subtitle)}</text>"
        )

    legend_x = width - 250
    legend_y = 42
    for index, row in enumerate(rows):
        color = TRACKER_COLORS.get(row.tracker, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])
        y = legend_y + (index * 22)
        parts.extend(
            [
                (
                    f'<rect x="{legend_x}" y="{y - 10}" width="12" height="12" '
                    f'rx="3" fill="{color}" />'
                ),
                f'<text x="{legend_x + 20}" y="{y}" font-size="13" font-family="Arial, sans-serif" '
                'fill="#1E293B">'
                f"{escape(row.tracker)}</text>",
            ]
        )

    for index, metric in enumerate(METRICS):
        panel_x = margin_x + index * (panel_width + gap)
        parts.extend(
            [
                f'<rect x="{panel_x}" y="{panel_y}" width="{panel_width}" height="{panel_height}" '
                'rx="18" fill="#FFFFFF" filter="url(#shadow)" />',
                f'<text x="{panel_x + 18}" y="{panel_y + 28}" font-size="16" '
                'font-family="Arial, sans-serif" font-weight="700" fill="#0F172A">'
                f"{escape(metric.title)}</text>",
            ]
        )

        label_x = panel_x + 18
        bar_x = panel_x + 148
        plot_width = panel_width - 168
        top_y = panel_y + 58
        row_gap = 58
        values = [float(getattr(row, metric.key)) for row in rows]
        max_value = max(values) or 1.0

        for row_index, row in enumerate(rows):
            value = float(getattr(row, metric.key))
            y = top_y + row_index * row_gap
            color = TRACKER_COLORS.get(
                row.tracker,
                FALLBACK_COLORS[row_index % len(FALLBACK_COLORS)],
            )
            bar_width = max(6, int((value / max_value) * plot_width)) if value > 0 else 0
            parts.extend(
                [
                    f'<text x="{label_x}" y="{y + 17}" font-size="13" '
                    'font-family="Arial, sans-serif" fill="#334155">'
                    f"{escape(row.tracker)}</text>",
                    f'<rect x="{bar_x}" y="{y}" width="{plot_width}" height="24" rx="12" '
                    'fill="#E2E8F0" />',
                    (
                        f'<rect x="{bar_x}" y="{y}" width="{bar_width}" height="24" rx="12" '
                        f'fill="{color}" />'
                        if bar_width
                        else ""
                    ),
                    f'<text x="{bar_x + plot_width}" y="{y + 17}" font-size="12" '
                    'font-family="Arial, sans-serif" text-anchor="end" fill="#0F172A">'
                    f"{escape(metric.formatter(value))}</text>",
                ]
            )

        tick_y = panel_y + panel_height - 26
        parts.extend(
            [
                f'<text x="{bar_x}" y="{tick_y}" font-size="11" font-family="Arial, sans-serif" '
                'fill="#64748B" text-anchor="start">0</text>',
                (
                    f'<text x="{bar_x + plot_width}" y="{tick_y}" font-size="11" '
                    'font-family="Arial, sans-serif" fill="#64748B" text-anchor="end">'
                    f"{escape(metric.formatter(max_value))}</text>"
                ),
            ]
        )

    parts.append("</svg>")
    output.write_text("\n".join(part for part in parts if part), encoding="utf-8")
    return output
