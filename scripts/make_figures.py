"""Create a compact experiment comparison figure."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="artifacts/experiment_results.csv")
    parser.add_argument("--output", default="artifacts/figures/f1_by_frontend.png")
    args = parser.parse_args()
    results = pd.read_csv(args.results)
    required = {"filter", "features", "f1_macro"}
    missing = required.difference(results.columns)
    if missing:
        raise SystemExit(f"Missing columns: {sorted(missing)}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    sns.barplot(data=results, x="features", y="f1_macro", hue="filter")
    plt.ylim(0, 1)
    plt.ylabel("Macro-F1")
    plt.xlabel("Feature representation")
    plt.title("DSP front-end ablation")
    plt.tight_layout()
    plt.savefig(output, dpi=180)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
