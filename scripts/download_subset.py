"""Download a patient-wise CirCor subset directly from PhysioNet directory listing."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import requests


BASE_URL = "https://physionet.org/files/circor-heart-sound/1.0.3/training_data/"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--patients", type=int, default=100)
    parser.add_argument("--output", default="data/circor-subset/training_data")
    parser.add_argument("--base-url", default=BASE_URL)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    session = requests.Session()
    html = session.get(args.base_url, timeout=(30, 120)).text
    names = sorted(set(re.findall(r'href="([^"]+)"', html)))
    patient_ids = [Path(name).stem for name in names if name.endswith(".txt")][: args.patients]
    if not patient_ids:
        raise SystemExit("No patient metadata files found in directory listing.")
    selected = [name for name in names if any(name == f"{patient_id}.txt" or name.startswith(f"{patient_id}_") for patient_id in patient_ids)]
    for index, name in enumerate(selected, 1):
        destination = root / name
        if destination.exists() and destination.stat().st_size > 0:
            continue
        response = session.get(args.base_url + name, timeout=(30, 120))
        response.raise_for_status()
        destination.write_bytes(response.content)
        if index % 20 == 0 or index == len(selected):
            print(f"Downloaded {index}/{len(selected)} files")
    print(f"Downloaded {len(patient_ids)} patients and {len(selected)} files to {root}")


if __name__ == "__main__":
    main()
