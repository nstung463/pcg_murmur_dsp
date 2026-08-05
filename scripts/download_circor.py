"""Download and extract the public CirCor training archive."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

import requests


DEFAULT_URL = "https://physionet.org/content/circor-heart-sound/get-zip/1.0.3/"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/circor-heart-sound")
    parser.add_argument("--url", default=DEFAULT_URL)
    args = parser.parse_args()
    destination = Path(args.output)
    destination.mkdir(parents=True, exist_ok=True)
    archive_path = destination / "circor-heart-sound.zip"
    with requests.get(args.url, stream=True, timeout=(30, 120)) as response:
        response.raise_for_status()
        total = int(response.headers.get("content-length", 0))
        downloaded = 0
        with archive_path.open("wb") as output:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                if chunk:
                    output.write(chunk)
                    downloaded += len(chunk)
                    if total:
                        print(f"Downloaded {downloaded / 1e6:.1f}/{total / 1e6:.1f} MB", end="\r", flush=True)
    print()
    with zipfile.ZipFile(archive_path) as archive:
        names = archive.namelist()
        archive.extractall(destination)
    archive_path.unlink()
    print(f"Extracted {len(names)} files into {destination}")


if __name__ == "__main__":
    main()
