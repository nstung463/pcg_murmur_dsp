"""Create report figures for sampling, quantization and noise robustness."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", default="artifacts/circor_100_robustness.csv")
    parser.add_argument("--output", default="artifacts/circor_100_robustness.png")
    args = parser.parse_args()
    data = pd.read_csv(args.results)
    data["group"] = data["experiment"].str.split("_").str[0]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    sampling = data[data.group == "sampling"]
    quantization = data[data.group == "quantization"]
    noise = data[data.group == "noise"].copy()
    sns.barplot(data=sampling, x="target_fs", y="f1_macro", ax=axes[0], color="#0f766e")
    axes[0].set_title("Sampling rate")
    axes[0].set_xlabel("Target Hz")
    axes[0].set_ylabel("Macro-F1")
    sns.barplot(data=quantization, x="quantization_bits", y="f1_macro", ax=axes[1], color="#7c3aed")
    axes[1].set_title("Quantization")
    axes[1].set_xlabel("Bits")
    axes[1].set_ylabel("Macro-F1")
    noise["condition"] = noise["noise_kind"] + " " + noise["snr_db"].astype(int).astype(str) + " dB"
    sns.barplot(data=noise, x="condition", y="f1_macro", ax=axes[2], color="#ea580c")
    axes[2].set_title("Noise robustness")
    axes[2].set_xlabel("")
    axes[2].set_ylabel("Macro-F1")
    for axis in axes:
        axis.set_ylim(0, 1)
        axis.tick_params(axis="x", rotation=30)
        axis.grid(axis="y", alpha=0.2)
    fig.savefig(output, dpi=180)
    print(f"Wrote {output}")


if __name__ == "__main__":
    main()
