"""Run the full three-class challenge-aligned DSP matrix efficiently.

Unlike the original binary ablation runner, this script keeps Present,
Unknown and Absent, extracts the four feature families once per filtered
window, and then trains SVM/MLP on the same cached patient vectors. This makes
the 24-config matrix practical while preserving one patient-wise split.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import f1_score

from pcg_dsp.dsp import apply_filter, design_filter, feature_vector, quantize, resample_signal, segment_signal
from pcg_dsp.io import iter_patients, load_wav
from pcg_dsp.metrics import CHALLENGE_LABELS, challenge_metrics
from pcg_dsp.models import make_model
from pcg_dsp.pipeline import patient_split


FEATURES = ("psd", "mfcc", "stft", "hybrid")
FILTERS = ("none", "butterworth", "fir")
MODELS = ("svm", "mlp")


def _split_hybrid(vector: np.ndarray) -> dict[str, np.ndarray]:
    """Split the project's 46-value hybrid vector into its feature families."""

    vector = np.asarray(vector, dtype=np.float32)
    # base=5, spectral=5, FFT=6, STFT=4, MFCC=26 for n_mfcc=13.
    if vector.size < 20:
        raise ValueError(f"Unexpected hybrid feature length: {vector.size}")
    base = vector[:5]
    spectral_fft = vector[:16]
    stft = np.concatenate([base, vector[16:20]])
    mfcc = np.concatenate([base, vector[20:]])
    return {"psd": spectral_fft, "mfcc": mfcc, "stft": stft, "hybrid": vector}


def _patient_vectors(patient, filter_spec, config: dict) -> dict[str, np.ndarray]:
    dsp = config["dsp"]
    per_mode: dict[str, list[np.ndarray]] = {mode: [] for mode in FEATURES}
    for recording in patient.recordings:
        source_fs, raw = load_wav(recording.wav_path)
        x = resample_signal(raw, source_fs, int(dsp["target_fs"]))
        bits = dsp.get("quantization_bits")
        if bits is not None:
            x = quantize(x, int(bits))
        x = apply_filter(x, filter_spec)
        segments = segment_signal(x, int(dsp["target_fs"]), float(dsp["segment_seconds"]), float(dsp["segment_hop_seconds"]))
        recording_modes: dict[str, list[np.ndarray]] = {mode: [] for mode in FEATURES}
        for segment in segments:
            families = _split_hybrid(
                feature_vector(
                    segment,
                    int(dsp["target_fs"]),
                    mode="hybrid",
                    n_mfcc=int(config["features"]["n_mfcc"]),
                    n_fft=int(config["features"]["n_fft"]),
                    hop_length=int(config["features"]["hop_length"]),
                )
            )
            for mode, values in families.items():
                recording_modes[mode].append(values)
        for mode in FEATURES:
            if recording_modes[mode]:
                per_mode[mode].append(np.mean(recording_modes[mode], axis=0))
    if not all(per_mode[mode] for mode in FEATURES):
        raise ValueError(f"No readable feature vectors for patient {patient.patient_id}")
    return {mode: np.mean(per_mode[mode], axis=0).astype(np.float32) for mode in FEATURES}


def build_cached_tables(data_dir: Path, config: dict) -> dict[str, pd.DataFrame]:
    patients = [patient for patient in iter_patients(data_dir) if patient.label in CHALLENGE_LABELS]
    tables: dict[str, list[dict]] = {filt: [] for filt in FILTERS}
    for filter_name in FILTERS:
        spec = design_filter(
            int(config["dsp"]["target_fs"]),
            filter_name,
            float(config["dsp"]["low_hz"]),
            float(config["dsp"]["high_hz"]),
            int(config["dsp"]["order"]),
            int(config["dsp"]["fir_taps"]),
        )
        print(f"Extracting filter={filter_name} for {len(patients)} patients")
        for index, patient in enumerate(patients, start=1):
            try:
                vectors = _patient_vectors(patient, spec, config)
                for mode, values in vectors.items():
                    tables[filter_name].append({"patient_id": patient.patient_id, "label": patient.label, "features": values.tolist(), "mode": mode})
            except Exception as exc:
                print(f"Skipping {patient.patient_id}: {exc}")
            if index % 100 == 0:
                print(f"  {index}/{len(patients)}")
    return {
        filt: pd.DataFrame(rows)
        for filt, rows in tables.items()
    }


def _fit_evaluate(table: pd.DataFrame, filter_name: str, mode: str, model_name: str, config: dict, output_dir: Path) -> dict:
    subset = table[table["mode"] == mode].copy()
    ids_train, ids_val, ids_test = patient_split(
        subset[["patient_id", "label"]],
        seed=int(config["split"]["seed"]),
        train_size=float(config["split"]["train_size"]),
        validation_size=float(config["split"]["validation_size"]),
    )
    train = subset[subset.patient_id.isin(ids_train)]
    test = subset[subset.patient_id.isin(ids_test)]
    x_train = np.asarray(train.features.tolist(), dtype=np.float32)
    x_test = np.asarray(test.features.tolist(), dtype=np.float32)
    y_train = train.label.to_numpy()
    y_test = test.label.to_numpy()
    model = make_model(model_name, int(config["model"]["seed"]))
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    labels = list(CHALLENGE_LABELS)
    probabilities = model.predict_proba(x_test)
    result = {
        "model": model_name,
        "filter": filter_name,
        "feature": mode,
        "task": "three_class_murmur",
        "n_train": len(train),
        "n_validation": len(ids_val),
        "n_test": len(test),
        "accuracy": float(np.mean(prediction == y_test)),
        "balanced_accuracy": float(challenge_metrics(y_test, prediction, probabilities, labels)["uar"]),
        "macro_f1": float(f1_score(y_test, prediction, labels=labels, average="macro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": pd.crosstab(pd.Categorical(y_test, categories=labels), pd.Categorical(prediction, categories=labels), dropna=False).values.tolist(),
    }
    result.update(challenge_metrics(y_test, prediction, probabilities, labels))
    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "config": config}, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/challenge_aligned.yaml")
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--output-dir", default="artifacts/challenge_matrix_3class", type=Path)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    config["data"]["include_unknown"] = True
    tables = build_cached_tables(args.data_dir, config)
    rows = []
    for filter_name in FILTERS:
        for model_name in MODELS:
            for mode in FEATURES:
                run_config = copy.deepcopy(config)
                run_config["model"]["kind"] = model_name
                run_config["dsp"]["filter"] = filter_name
                run_config["features"]["mode"] = mode
                result = _fit_evaluate(tables[filter_name], filter_name, mode, model_name, run_config, args.output_dir / f"{model_name}_{filter_name}_{mode}")
                rows.append(result)
                print(result)
    pd.DataFrame(rows).to_csv(args.output_dir / "full_matrix_3class.csv", index=False)
    print(f"Wrote {len(rows)} rows to {args.output_dir / 'full_matrix_3class.csv'}")


if __name__ == "__main__":
    main()
