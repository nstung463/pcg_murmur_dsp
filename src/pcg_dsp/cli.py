"""Command-line entry points for reproducible PCG experiments."""

from __future__ import annotations

import argparse
from pathlib import Path

import yaml

from .io import build_manifest
from .pipeline import make_patient_table, train_evaluate


def load_config(path: str | Path) -> dict:
    return yaml.safe_load(Path(path).read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(description="DSP501 PCG murmur detection experiments")
    sub = parser.add_subparsers(dest="command", required=True)
    manifest_parser = sub.add_parser("prepare-manifest")
    manifest_parser.add_argument("--data-dir", required=True)
    manifest_parser.add_argument("--output", default="artifacts/manifest.csv")
    run_parser = sub.add_parser("run-experiment")
    run_parser.add_argument("--config", default="configs/default.yaml")
    run_parser.add_argument("--data-dir")
    run_parser.add_argument("--run-dir", default="artifacts/runs/smoke")
    run_parser.add_argument("--max-patients", type=int)
    args = parser.parse_args()
    if args.command == "prepare-manifest":
        manifest = build_manifest(args.data_dir)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        manifest.to_csv(args.output, index=False)
        print(f"Wrote {len(manifest)} recordings to {args.output}")
    elif args.command == "run-experiment":
        config = load_config(args.config)
        data_dir = args.data_dir or config["data"]["root"]
        table = make_patient_table(data_dir, config, args.max_patients)
        if table["label"].nunique() < 2:
            raise SystemExit("Need both Present and Absent patients; increase --max-patients.")
        result = train_evaluate(table, config, args.run_dir)
        print(result)


if __name__ == "__main__":
    main()
