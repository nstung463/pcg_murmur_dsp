"""Run sampling, quantization and noise robustness experiments for the selected DSP front-end."""

from __future__ import annotations

import argparse
import copy
from pathlib import Path

import pandas as pd
import yaml

from pcg_dsp.pipeline import make_patient_table, train_evaluate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/default.yaml")
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--output", default="artifacts/robustness_results.csv")
    parser.add_argument("--max-patients", type=int)
    args = parser.parse_args()

    base = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    base["model"]["kind"] = "svm"
    base["dsp"]["filter"] = "butterworth"
    base["features"]["mode"] = "hybrid"
    experiments: list[tuple[str, dict]] = []

    for fs in (1000, 2000, 4000):
        config = copy.deepcopy(base)
        config["dsp"]["target_fs"] = fs
        config["dsp"]["quantization_bits"] = 16
        config["noise"]["enabled"] = False
        experiments.append((f"sampling_{fs}hz", config))

    for bits in (8, 12, 16):
        config = copy.deepcopy(base)
        config["dsp"]["target_fs"] = 1000
        config["dsp"]["quantization_bits"] = bits
        config["noise"]["enabled"] = False
        experiments.append((f"quantization_{bits}bit", config))

    for kind, snr in (("white", 20), ("white", 10), ("pink", 10), ("impulse", 10)):
        config = copy.deepcopy(base)
        config["dsp"]["target_fs"] = 1000
        config["dsp"]["quantization_bits"] = 16
        config["noise"] = {"enabled": True, "kind": kind, "snr_db": snr}
        experiments.append((f"noise_{kind}_{snr}db", config))

    rows = []
    output = Path(args.output)
    for name, config in experiments:
        table = make_patient_table(args.data_dir, config, args.max_patients)
        result = train_evaluate(table, config, output.parent / "runs" / name)
        noise_enabled = bool(config["noise"].get("enabled", False))
        rows.append({"experiment": name, "target_fs": config["dsp"]["target_fs"], "quantization_bits": config["dsp"]["quantization_bits"], "noise_kind": config["noise"].get("kind", "none") if noise_enabled else "none", "snr_db": config["noise"].get("snr_db", None) if noise_enabled else None, **result})
        print(name, result)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} robustness rows to {output}")


if __name__ == "__main__":
    main()
