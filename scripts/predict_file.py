"""Run a saved classifier on one PCG WAV file."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pcg_dsp.service import analyze_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--config", help="Fallback config for legacy plain-model files.")
    args = parser.parse_args()
    result = analyze_file(args.wav, args.model, args.config)
    result = {"label": result["label"], "wav": str(Path(args.wav)), "probabilities": result["probabilities"]}
    print(json.dumps(result, ensure_ascii=True, indent=2))


if __name__ == "__main__":
    main()
