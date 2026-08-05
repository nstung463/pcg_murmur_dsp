"""Audit labels, sampling rates, durations and recording locations."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import pandas as pd

from pcg_dsp.io import build_manifest, load_wav


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output-dir", default="artifacts/data_audit")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args.data_dir, include_unknown=True)
    if manifest.empty:
        raise SystemExit("No patient recordings found.")
    sample_rates = Counter()
    durations = []
    missing = []
    for path in manifest.wav_path:
        try:
            fs, signal = load_wav(path)
            sample_rates[str(fs)] += 1
            durations.append(len(signal) / fs)
        except Exception as exc:
            missing.append({"path": path, "error": str(exc)})
    summary = {
        "recordings": int(len(manifest)),
        "patients": int(manifest.patient_id.nunique()),
        "labels": manifest.groupby("label").size().to_dict(),
        "locations": manifest.groupby("location").size().to_dict(),
        "sample_rates": dict(sample_rates),
        "duration_seconds": {"min": min(durations), "mean": sum(durations) / len(durations), "max": max(durations)},
        "missing_or_invalid": missing,
    }
    manifest.to_csv(output / "manifest.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
