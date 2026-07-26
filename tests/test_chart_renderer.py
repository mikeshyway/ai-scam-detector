import subprocess
from pathlib import Path

import plotly.graph_objects as go
from PIL import Image

from src.reporting import chart_renderer


def test_edge_chart_batch_splits_one_screenshot_into_ordered_images(
    monkeypatch,
) -> None:
    output_dir = Path("tmp") / "chart-renderer-unit-test" / "images"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_paths = [
        output_dir / "first.png",
        output_dir / "second.png",
    ]

    def fake_run(command: list[str], **_kwargs: object) -> subprocess.CompletedProcess:
        screenshot_argument = next(
            item for item in command if item.startswith("--screenshot=")
        )
        screenshot_path = Path(screenshot_argument.split("=", 1)[1])
        screenshot = Image.new(
            "RGB",
            (chart_renderer.CHART_WIDTH, chart_renderer.CHART_HEIGHT * 2),
            (220, 38, 38),
        )
        screenshot.paste(
            (37, 99, 235),
            (
                0,
                chart_renderer.CHART_HEIGHT,
                chart_renderer.CHART_WIDTH,
                chart_renderer.CHART_HEIGHT * 2,
            ),
        )
        screenshot.save(screenshot_path, format="PNG")
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(chart_renderer, "_browser_path", lambda: "msedge.exe")
    monkeypatch.setattr(chart_renderer.subprocess, "run", fake_run)

    chart_renderer._render_with_edge(
        [
            go.Figure(go.Bar(x=["A"], y=[1])),
            go.Figure(go.Bar(x=["B"], y=[2])),
        ],
        output_paths,
        output_dir,
    )

    with Image.open(output_paths[0]) as first:
        assert first.size == (
            chart_renderer.CHART_WIDTH,
            chart_renderer.CHART_HEIGHT,
        )
        assert first.getpixel((10, 10)) == (220, 38, 38)
    with Image.open(output_paths[1]) as second:
        assert second.size == (
            chart_renderer.CHART_WIDTH,
            chart_renderer.CHART_HEIGHT,
        )
        assert second.getpixel((10, 10)) == (37, 99, 235)
