"""Create a tiny local fixture for smoke tests without downloading clinical data."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.io import wavfile


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="artifacts/fixture/training_data")
    parser.add_argument("--patients", type=int, default=20)
    args = parser.parse_args()
    root = Path(args.output)
    root.mkdir(parents=True, exist_ok=True)
    fs = 4000
    t = np.arange(fs * 3, dtype=np.float32) / fs
    for index in range(args.patients):
        patient_id = f"{index + 1:05d}"
        label = "Present" if index % 2 else "Absent"
        lines = [
            f"{patient_id} 1 {fs}",
            f"AV {patient_id}_AV.hea {patient_id}_AV.wav {patient_id}_AV.tsv",
            f"#Murmur: {label}",
            "#Outcome: Normal",
        ]
        (root / f"{patient_id}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        frequency = 80.0 + 15.0 * (index % 4) if label == "Present" else 35.0
        noise = np.random.default_rng(index).normal(size=t.size)
        x = 0.2 * np.sin(2 * np.pi * frequency * t) + 0.03 * noise
        wavfile.write(root / f"{patient_id}_AV.wav", fs, np.asarray(np.clip(x, -1, 1) * 32767, dtype=np.int16))
        (root / f"{patient_id}_AV.tsv").write_text("0\t1\t1\n1\t2\t2\n2\t3\t3\n", encoding="utf-8")
        (root / f"{patient_id}_AV.hea").write_text(f"{patient_id}_AV 1 {fs} {len(t)}\n", encoding="utf-8")
    print(f"Created {args.patients} synthetic patients at {root}")


if __name__ == "__main__":
    main()
