"""Bounded subprocess entry point for batch Plotly image rendering."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import uuid
from pathlib import Path


CHART_WIDTH = 1200
CHART_HEIGHT = 700
EDGE_BATCH_SIZE = 12


def _browser_path() -> str:
    configured = os.environ.get("BROWSER_PATH", "").strip()
    if configured and Path(configured).exists():
        return configured
    candidates = [
        shutil.which("msedge"),
        shutil.which("google-chrome"),
        shutil.which("chromium"),
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    ]
    return next(
        (
            str(candidate)
            for candidate in candidates
            if candidate and Path(str(candidate)).exists()
        ),
        "",
    )


def _render_with_edge(
    figures: list[object],
    output_paths: list[Path],
    output_dir: Path,
) -> None:
    from PIL import Image
    import plotly.io as pio
    from plotly.offline.offline import get_plotlyjs

    browser_path = _browser_path()
    if not browser_path:
        raise RuntimeError("No supported browser is available for chart rendering.")

    creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    temp_parent = output_dir.parent.resolve()
    temp_path = (
        temp_parent / f"aifds_edge_charts_{uuid.uuid4().hex}"
    ).resolve()
    if temp_path.parent != temp_parent:
        raise RuntimeError("Chart-rendering workspace resolved outside its parent.")
    temp_path.mkdir(parents=True, exist_ok=False)
    try:
        for batch_start in range(0, len(figures), EDGE_BATCH_SIZE):
            batch = figures[batch_start : batch_start + EDGE_BATCH_SIZE]
            batch_outputs = output_paths[
                batch_start : batch_start + EDGE_BATCH_SIZE
            ]
            batch_number = batch_start // EDGE_BATCH_SIZE
            html_path = temp_path / f"charts-{batch_number}.html"
            screenshot_path = temp_path / f"charts-{batch_number}.png"
            profile_path = temp_path / f"edge-profile-{batch_number}"
            html_parts = [
                "<!doctype html><html><head><meta charset=\"utf-8\">",
                (
                    "<style>html,body{margin:0;padding:0;background:#fff;"
                    "overflow:hidden}.chart{width:1200px;height:700px;"
                    "overflow:hidden}</style>"
                ),
                f"<script>{get_plotlyjs()}</script></head><body>",
            ]
            for offset, figure in enumerate(batch):
                figure.update_layout(width=CHART_WIDTH, height=CHART_HEIGHT)
                html_parts.extend(
                    [
                        '<div class="chart">',
                        pio.to_html(
                            figure,
                            include_plotlyjs=False,
                            full_html=False,
                            config={
                                "staticPlot": True,
                                "displayModeBar": False,
                            },
                            default_width=f"{CHART_WIDTH}px",
                            default_height=f"{CHART_HEIGHT}px",
                            div_id=f"chart-{batch_number}-{offset}",
                        ),
                        "</div>",
                    ]
                )
            html_parts.append("</body></html>")
            html_path.write_text(
                "".join(html_parts),
                encoding="utf-8",
            )
            viewport_height = CHART_HEIGHT * len(batch)
            result = subprocess.run(
                [
                    browser_path,
                    "--headless=new",
                    "--disable-gpu",
                    "--hide-scrollbars",
                    "--no-first-run",
                    "--no-default-browser-check",
                    "--disable-background-networking",
                    "--disable-component-update",
                    "--disable-crash-reporter",
                    "--disable-breakpad",
                    "--disable-sync",
                    "--allow-file-access-from-files",
                    "--force-device-scale-factor=1",
                    f"--user-data-dir={profile_path}",
                    f"--window-size={CHART_WIDTH},{viewport_height}",
                    "--run-all-compositor-stages-before-draw",
                    "--virtual-time-budget=5000",
                    f"--screenshot={screenshot_path}",
                    html_path.as_uri(),
                ],
                capture_output=True,
                text=True,
                timeout=30,
                check=False,
                creationflags=creation_flags,
            )
            if result.returncode != 0 or not screenshot_path.exists():
                detail = result.stderr.strip()[-1000:]
                raise RuntimeError(
                    f"Headless browser chart rendering failed: {detail}"
                )
            with Image.open(screenshot_path) as screenshot:
                expected_size = (CHART_WIDTH, viewport_height)
                if screenshot.size != expected_size:
                    raise RuntimeError(
                        "Headless browser returned an unexpected chart image size: "
                        f"{screenshot.size}, expected {expected_size}."
                    )
                for offset, output_path in enumerate(batch_outputs):
                    top = offset * CHART_HEIGHT
                    chart = screenshot.crop(
                        (0, top, CHART_WIDTH, top + CHART_HEIGHT)
                    )
                    chart.save(output_path, format="PNG")
    finally:
        if (
            temp_path.exists()
            and temp_path.parent == temp_parent
            and temp_path.name.startswith("aifds_edge_charts_")
        ):
            shutil.rmtree(temp_path, ignore_errors=True)


def _render_with_kaleido(
    pio: object,
    figures: list[object],
    output_paths: list[Path],
) -> None:
    write_images = getattr(pio, "write_images", None)
    if callable(write_images):
        write_images(
            fig=figures,
            file=[str(path) for path in output_paths],
            format="png",
            width=CHART_WIDTH,
            height=CHART_HEIGHT,
            scale=1.5,
        )
        return

    for figure, output_path in zip(figures, output_paths):
        figure.write_image(
            str(output_path),
            format="png",
            width=CHART_WIDTH,
            height=CHART_HEIGHT,
            scale=1.5,
        )


def render_manifest(manifest_path: Path, output_dir: Path) -> None:
    import plotly.io as pio

    entries = json.loads(manifest_path.read_text(encoding="utf-8"))
    figures = []
    output_paths = []
    for entry in entries:
        figure_json = str(entry.get("figure_json", "")).strip()
        filename = str(entry.get("filename", "")).strip()
        if not figure_json or not filename:
            continue
        figures.append(pio.from_json(figure_json))
        output_paths.append(output_dir / filename)

    if not figures:
        return

    browser_path = _browser_path()
    if os.name == "nt" and browser_path:
        try:
            _render_with_edge(figures, output_paths, output_dir)
            return
        except (OSError, RuntimeError, subprocess.SubprocessError):
            pass

    try:
        _render_with_kaleido(pio, figures, output_paths)
    except Exception:
        if not browser_path or os.name == "nt":
            raise
        _render_with_edge(figures, output_paths, output_dir)


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit("Usage: chart_renderer <manifest.json> <output_dir>")
    manifest_path = Path(sys.argv[1]).resolve()
    output_dir = Path(sys.argv[2]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    render_manifest(manifest_path, output_dir)


if __name__ == "__main__":
    main()
