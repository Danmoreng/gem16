#!/usr/bin/env python3
"""Generate the original synthetic PNG fixtures used by multimodal benchmarks."""

from __future__ import annotations

import argparse
import pathlib
import shutil
import subprocess
import tempfile


WIDTH = 896
HEIGHT = 640

IMAGES = {
    "orbit_station_47.png": """<svg xmlns="http://www.w3.org/2000/svg" width="896" height="640" viewBox="0 0 896 640">
<rect width="896" height="640" fill="#101b38"/>
<circle cx="735" cy="118" r="74" fill="#f4d35e"/><circle cx="735" cy="118" r="58" fill="#f8e69a"/>
<g fill="#d7f9ff"><circle cx="100" cy="95" r="6"/><circle cx="210" cy="150" r="5"/><circle cx="390" cy="80" r="7"/><circle cx="540" cy="155" r="5"/><circle cx="825" cy="220" r="6"/></g>
<path d="M0 510 L180 390 L310 470 L470 330 L650 455 L800 350 L896 430 L896 640 L0 640Z" fill="#273b62"/>
<rect x="190" y="310" width="480" height="220" rx="18" fill="#e87531" stroke="#fff2d0" stroke-width="8"/>
<path d="M300 310 A130 130 0 0 1 560 310Z" fill="#d9eaf2" stroke="#fff2d0" stroke-width="8"/>
<rect x="255" y="405" width="350" height="86" rx="10" fill="#fff8dc"/>
<text x="430" y="466" text-anchor="middle" font-family="Noto Sans, sans-serif" font-size="62" font-weight="bold" fill="#13213d">ORBIT 47</text>
<text x="448" y="592" text-anchor="middle" font-family="Noto Sans, sans-serif" font-size="28" fill="#d7f9ff">DESERT OBSERVATORY</text>
</svg>""",
    "cedar_greenhouse_82.png": """<svg xmlns="http://www.w3.org/2000/svg" width="896" height="640" viewBox="0 0 896 640">
<rect width="896" height="640" fill="#bfe3c0"/><rect y="440" width="896" height="200" fill="#668f55"/>
<circle cx="108" cy="100" r="62" fill="#ffd166"/>
<path d="M160 430 L260 205 L640 205 L750 430Z" fill="#d8f3ed" fill-opacity="0.85" stroke="#315c4b" stroke-width="10"/>
<path d="M450 205 V430 M260 205 L450 430 M640 205 L450 430" stroke="#6b9484" stroke-width="7"/>
<rect x="285" y="245" width="330" height="82" rx="8" fill="#173f35"/>
<text x="450" y="304" text-anchor="middle" font-family="Noto Sans, sans-serif" font-size="57" font-weight="bold" fill="#f6f1d1">CEDAR 82</text>
<g><rect x="300" y="390" width="105" height="100" rx="8" fill="#7d4e9e"/><rect x="425" y="390" width="105" height="100" rx="8" fill="#7d4e9e"/><rect x="550" y="390" width="105" height="100" rx="8" fill="#7d4e9e"/></g>
<g fill="#2b6b3f"><circle cx="352" cy="370" r="42"/><circle cx="477" cy="365" r="46"/><circle cx="602" cy="370" r="42"/></g>
<text x="448" y="575" text-anchor="middle" font-family="Noto Sans, sans-serif" font-size="29" fill="#f7fff1">THREE PURPLE PLANTERS</text>
</svg>""",
    "harbor_lighthouse_19.png": """<svg xmlns="http://www.w3.org/2000/svg" width="896" height="640" viewBox="0 0 896 640">
<rect width="896" height="390" fill="#8ed6f2"/><rect y="390" width="896" height="250" fill="#287bb5"/>
<circle cx="735" cy="102" r="62" fill="#ffe27a"/>
<path d="M110 470 L165 165 L285 165 L335 470Z" fill="#f6f0dc" stroke="#24384f" stroke-width="8"/>
<path d="M154 230 H296 L307 295 H143Z" fill="#d94841"/>
<rect x="145" y="112" width="160" height="58" fill="#25384c"/><path d="M130 112 L225 55 L320 112Z" fill="#d94841"/>
<rect x="90" y="480" width="280" height="82" rx="9" fill="#fff7d6"/>
<text x="230" y="539" text-anchor="middle" font-family="Noto Sans, sans-serif" font-size="55" font-weight="bold" fill="#173451">HARBOR 19</text>
<g fill="#e54b4b" stroke="#fff" stroke-width="4"><path d="M455 475 l55 -75 l0 75Z"/><path d="M565 525 l55 -75 l0 75Z"/><path d="M680 465 l55 -75 l0 75Z"/><path d="M770 545 l48 -67 l0 67Z"/></g>
<g fill="#53392d"><rect x="445" y="475" width="80" height="16"/><rect x="555" y="525" width="80" height="16"/><rect x="670" y="465" width="80" height="16"/><rect x="762" y="545" width="70" height="14"/></g>
<text x="620" y="605" text-anchor="middle" font-family="Noto Sans, sans-serif" font-size="29" fill="#ffffff">FOUR RED SAILBOATS</text>
</svg>""",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "benchmarks/media/images",
    )
    args = parser.parse_args()
    converter = shutil.which("rsvg-convert")
    if converter is None:
        parser.error("rsvg-convert is required to reproduce the PNG fixtures")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gem16-benchmark-images-") as temporary:
        for name, svg in IMAGES.items():
            source = pathlib.Path(temporary) / f"{name}.svg"
            source.write_text(svg, encoding="utf-8")
            subprocess.run(
                [converter, "--format=png", "--output", str(args.output_dir / name), str(source)],
                check=True,
            )
            print(args.output_dir / name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
