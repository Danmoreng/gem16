#!/usr/bin/env python3
"""Generate deterministic, project-owned image fixtures for Vision V19."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile


IMAGES = {
    "scene-square.png": """<svg xmlns="http://www.w3.org/2000/svg" width="800" height="800" viewBox="0 0 800 800">
<rect width="800" height="800" fill="#17233f"/>
<g fill="#fff4b0"><path d="M80 90 l10 22 24 2-18 16 6 24-22-13-22 13 6-24-18-16 24-2Z"/><path d="M230 90 l10 22 24 2-18 16 6 24-22-13-22 13 6-24-18-16 24-2Z"/><path d="M380 90 l10 22 24 2-18 16 6 24-22-13-22 13 6-24-18-16 24-2Z"/><path d="M530 90 l10 22 24 2-18 16 6 24-22-13-22 13 6-24-18-16 24-2Z"/><path d="M680 90 l10 22 24 2-18 16 6 24-22-13-22 13 6-24-18-16 24-2Z"/></g>
<rect x="255" y="380" width="300" height="42" rx="8" fill="#4d8bd6"/><rect x="285" y="422" width="28" height="190" fill="#4d8bd6"/><rect x="497" y="422" width="28" height="190" fill="#4d8bd6"/>
<g fill="#f4c542" stroke="#2a2f36" stroke-width="8"><rect x="75" y="330" width="150" height="190" rx="24"/><circle cx="150" cy="290" r="72"/></g><g fill="#222"><circle cx="125" cy="280" r="10"/><circle cx="175" cy="280" r="10"/></g><path d="M120 320 Q150 340 180 320" fill="none" stroke="#222" stroke-width="8"/>
<path d="M610 355 h120 v210 h-120 z M590 565 h160 v35 h-160 z" fill="#54ad65" stroke="#203528" stroke-width="8"/>
<circle cx="405" cy="535" r="50" fill="#dc4b47" stroke="#5d2423" stroke-width="8"/>
<g font-family="Noto Sans, sans-serif" font-weight="bold" fill="#fff"><text x="150" y="570" text-anchor="middle" font-size="30">YELLOW ROBOT</text><text x="670" y="650" text-anchor="middle" font-size="30">GREEN CHAIR</text><text x="405" y="720" text-anchor="middle" font-size="30">RED BALL UNDER TABLE</text></g>
</svg>""",
    "chart-wide.png": """<svg xmlns="http://www.w3.org/2000/svg" width="1200" height="600" viewBox="0 0 1200 600">
<rect width="1200" height="600" fill="#f7f3e8"/><text x="600" y="72" text-anchor="middle" font-family="Noto Sans, sans-serif" font-size="48" font-weight="bold" fill="#24344d">Q4 HARVEST</text>
<path d="M130 480 H1110 M130 130 V480" stroke="#26384e" stroke-width="8"/>
<g stroke="#bbb" stroke-width="2"><path d="M130 380 H1110"/><path d="M130 280 H1110"/><path d="M130 180 H1110"/></g>
<rect x="230" y="330" width="190" height="150" fill="#e49b45"/><rect x="505" y="130" width="190" height="350" fill="#4c9c68"/><rect x="780" y="230" width="190" height="250" fill="#557dc1"/>
<g font-family="Noto Sans, sans-serif" text-anchor="middle" fill="#1d2a3b"><text x="325" y="320" font-size="42" font-weight="bold">3</text><text x="600" y="120" font-size="42" font-weight="bold">7</text><text x="875" y="220" font-size="42" font-weight="bold">5</text><text x="325" y="535" font-size="34">BIRCH</text><text x="600" y="535" font-size="34">CEDAR</text><text x="875" y="535" font-size="34">MAPLE</text></g>
</svg>""",
    "document-tall.png": """<svg xmlns="http://www.w3.org/2000/svg" width="700" height="1000" viewBox="0 0 700 1000">
<rect width="700" height="1000" fill="#d9d4c9"/><rect x="70" y="55" width="560" height="890" rx="8" fill="#fff" stroke="#485264" stroke-width="4"/>
<text x="350" y="145" text-anchor="middle" font-family="Noto Serif, serif" font-size="48" font-weight="bold" fill="#22314b">NORTHSTAR REPORT</text><path d="M120 180 H580" stroke="#6882a8" stroke-width="4"/>
<g font-family="Noto Sans, sans-serif" fill="#283444"><text x="125" y="250" font-size="34" font-weight="bold">DOCUMENT CODE: NS-731</text><text x="125" y="310" font-size="30">DATE: 18 AUG 2026</text><text x="125" y="400" font-size="26">Status: Approved</text><text x="125" y="455" font-size="26">Owner: Cedar Lab</text><text x="125" y="510" font-size="26">Priority: Medium</text></g>
<g stroke="#c8ced8" stroke-width="3"><path d="M125 590 H575"/><path d="M125 645 H575"/><path d="M125 700 H575"/><path d="M125 755 H575"/></g>
<text x="350" y="875" text-anchor="middle" font-family="Noto Sans, sans-serif" font-size="18" fill="#465268">SMALL DETAIL CODE: moss-27</text>
</svg>""",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).resolve().parents[1]
        / "benchmarks/vision-v19/images",
    )
    args = parser.parse_args()
    converter = shutil.which("rsvg-convert")
    if converter is None:
        parser.error("rsvg-convert is required to reproduce the PNG fixtures")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gem16-vision-v19-") as temporary:
        for name, svg in IMAGES.items():
            source = Path(temporary) / f"{name}.svg"
            source.write_text(svg, encoding="utf-8")
            output = args.output_dir / name
            subprocess.run(
                [converter, "--format=png", "--output", str(output), str(source)],
                check=True,
            )
            print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
