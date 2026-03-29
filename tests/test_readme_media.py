from pathlib import Path

from vtrack.readme_media import BenchmarkRow, render_benchmark_svg


def test_render_benchmark_svg_writes_expected_labels(tmp_path: Path) -> None:
    output = tmp_path / "benchmark.svg"

    render_benchmark_svg(
        [
            BenchmarkRow("bytetrack", 24.3, 11.0, 3),
            BenchmarkRow("bytetrack-occlusion", 22.1, 14.5, 1),
            BenchmarkRow("botsort", 20.7, 10.2, 4),
        ],
        output,
        subtitle="model=best.pt · device=cpu · frames=150",
    )

    svg = output.read_text(encoding="utf-8")

    assert output.exists()
    assert "<svg" in svg
    assert "Tracker Benchmark Snapshot" in svg
    assert "Avg FPS" in svg
    assert "bytetrack-occlusion" in svg
    assert "frames=150" in svg
