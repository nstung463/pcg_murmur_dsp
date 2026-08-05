"""Run a controlled filter/feature ablation matrix."""

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
    parser.add_argument("--output", default="artifacts/experiment_results.csv")
    parser.add_argument("--max-patients", type=int)
    parser.add_argument("--quick", action="store_true", help="Run one filter and two feature modes.")
    parser.add_argument("--models", default="svm,mlp", help="Comma-separated model kinds.")
    args = parser.parse_args()

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    filters = ["none", "butterworth", "fir"]
    features = ["psd", "mfcc", "stft", "hybrid"]
    if args.quick:
        filters = ["butterworth"]
        features = ["psd", "stft", "hybrid"]
        models = ["svm"]
    else:
        models = [item.strip() for item in args.models.split(",") if item.strip()]

    rows = []
    for model_name in models:
        for filter_name in filters:
            for feature_name in features:
                run_config = copy.deepcopy(config)
                run_config["model"]["kind"] = model_name
                run_config["dsp"]["filter"] = filter_name
                run_config["features"]["mode"] = feature_name
                run_name = f"{model_name}_{filter_name}_{feature_name}"
                table = make_patient_table(args.data_dir, run_config, args.max_patients)
                result = train_evaluate(table, run_config, Path(args.output).parent / "runs" / run_name)
                rows.append({"model": model_name, "filter": filter_name, "features": feature_name, **result})
                print(run_name, result)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output, index=False)
    print(f"Wrote {len(rows)} experiment rows to {output}")


if __name__ == "__main__":
    main()
