#!/usr/bin/env python3
"""Reproduce the checked-in public-domain LibriVox WAV benchmark excerpts."""

from __future__ import annotations

import argparse
import hashlib
import pathlib
import shutil
import subprocess
import tempfile
import urllib.request


SOURCES = (
    {
        "name": "alice_sister_bank.wav",
        "url": "https://archive.org/download/aliceinwonderland_2106_librivox/alicesadventuresinwonderland_01_carroll_64kb.mp3",
        "source_sha256": "c73d5fee88431ec60a47a8d9dc747ffcb34faf3ad43bc62cd51607ea6dca1859",
        "start": "30",
        "duration": "14",
    },
    {
        "name": "moby_lexicons_flags.wav",
        "url": "https://archive.org/download/mobydick2_2511_librivox/mobydickorthewhale_001_melville_64kb.mp3",
        "source_sha256": "f61d1d9c73704c36e2b0c1869d1cf72fd817dd04484b2296d65439bbcb47f779",
        "start": "35",
        "duration": "14",
    },
    {
        "name": "pride_wife_neighborhood.wav",
        "url": "https://archive.org/download/prideandprejudice_1005_librivox/prideandprejudice_01_austen_64kb.mp3",
        "source_sha256": "aa7339960c90bed4c57ffe144c96b7a8ef4de482d6505d19906bb50248fe6f0c",
        "start": "30",
        "duration": "14",
    },
)


def sha256(path: pathlib.Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=pathlib.Path,
        default=pathlib.Path(__file__).resolve().parents[1] / "benchmarks/media/audio",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        parser.error("ffmpeg is required to reproduce the WAV excerpts")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="gem16-benchmark-audio-") as temporary:
        temporary_dir = pathlib.Path(temporary)
        for index, source in enumerate(SOURCES):
            output = args.output_dir / source["name"]
            if output.exists() and not args.force:
                raise SystemExit(f"refusing to overwrite {output}; pass --force")
            compressed = temporary_dir / f"source-{index}.mp3"
            print(f"download {source['url']}")
            urllib.request.urlretrieve(source["url"], compressed)
            observed = sha256(compressed)
            if observed != source["source_sha256"]:
                raise SystemExit(
                    f"source checksum mismatch for {source['url']}: {observed}"
                )
            subprocess.run(
                [
                    ffmpeg,
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-ss",
                    source["start"],
                    "-i",
                    str(compressed),
                    "-t",
                    source["duration"],
                    "-ac",
                    "1",
                    "-ar",
                    "16000",
                    "-c:a",
                    "pcm_s16le",
                    str(output),
                ],
                check=True,
            )
            print(f"wrote {output} sha256={sha256(output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
