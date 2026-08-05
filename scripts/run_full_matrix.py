"""Run the full 3-filter x 4-feature x 2-classifier DSP matrix.

Feature extraction is cached per filter so each WAV is decoded and filtered once
per filter, then all four feature representations are computed from the same
windows. This keeps the full-cohort experiment practical on CPU while using the
same project DSP and evaluation functions as training/inference.
"""

from __future__ import annotations

import copy
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from pcg_dsp.dsp import apply_filter, design_filter, feature_vector, quantize, resample_signal, segment_signal
from pcg_dsp.io import iter_patients, load_wav
from pcg_dsp.pipeline import train_evaluate


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "circor-heart-sound" / "1.0.3" / "training_data"
OUT_DIR = ROOT / "artifacts" / "full_matrix_24"
TARGET_FS = 1000
QUANTIZATION_BITS = 16
N_JOBS = 4
FILTERS = ("none", "butterworth", "fir")
FEATURES = ("psd", "mfcc", "stft", "hybrid")
MODELS = ("svm", "mlp")


BASE_CONFIG = {
    "data": {"include_unknown": False, "skip_invalid_patients": True, "n_jobs": N_JOBS},
    "split": {"seed": 42, "train_size": 0.70, "validation_size": 0.15},
    "dsp": {
        "target_fs": TARGET_FS,
        "quantization_bits": QUANTIZATION_BITS,
        "low_hz": 25,
        "high_hz": 400,
        "order": 4,
        "fir_taps": 129,
        "wavelet_denoise": False,
        "segment_seconds": 3.0,
        "segment_hop_seconds": 1.5,
    },
    "features": {"n_mfcc": 13, "n_fft": 512, "hop_length": 128},
    "noise": {"enabled": False, "snr_db": 10, "kind": "white"},
    "model": {"seed": 42},
}


def _one_patient(patient, filter_name: str):
    cfg = BASE_CONFIG
    dsp = cfg["dsp"]
    spec = design_filter(TARGET_FS, filter_name, dsp["low_hz"], dsp["high_hz"], dsp["order"], dsp["fir_taps"])
    collected = {mode: [] for mode in FEATURES}
    for recording in patient.recordings:
        source_fs, raw = load_wav(recording.wav_path)
        x = resample_signal(raw, source_fs, TARGET_FS)
        x = quantize(x, QUANTIZATION_BITS)
        x = apply_filter(x, spec)
        segments = segment_signal(x, TARGET_FS, dsp["segment_seconds"], dsp["segment_hop_seconds"])
        recording_vectors = {mode: [] for mode in FEATURES}
        for segment in segments:
            for mode in FEATURES:
                recording_vectors[mode].append(feature_vector(segment, TARGET_FS, mode=mode, **cfg["features"]))
        for mode in FEATURES:
            if recording_vectors[mode]:
                # Match pcg_dsp.pipeline._patient_features: mean windows per
                # recording first, then mean recordings per patient.
                collected[mode].append(np.mean(np.asarray(recording_vectors[mode], dtype=np.float32), axis=0))
    if not any(collected.values()):
        raise ValueError(f"No readable recordings for patient {patient.patient_id}")
    return patient.patient_id, patient.label, {
        mode: np.mean(np.asarray(vectors, dtype=np.float32), axis=0).tolist()
        for mode, vectors in collected.items()
        if vectors
    }


def make_filter_tables(filter_name: str) -> dict[str, pd.DataFrame]:
    patients = [p for p in iter_patients(DATA_DIR) if p.label != "Unknown"]
    started = time.perf_counter()
    computed = Parallel(n_jobs=N_JOBS, backend="loky", batch_size=1)(
        delayed(_one_patient)(patient, filter_name) for patient in patients
    )
    rows = {mode: [] for mode in FEATURES}
    for patient_id, label, vectors in computed:
        for mode in FEATURES:
            if mode in vectors:
                rows[mode].append({"patient_id": patient_id, "label": label, "features": vectors[mode]})
    elapsed = time.perf_counter() - started
    print(f"filter={filter_name}: {len(patients)} patients, {elapsed:.1f}s")
    return {mode: pd.DataFrame(values) for mode, values in rows.items()}


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix_rows = []
    for filter_name in FILTERS:
        tables = make_filter_tables(filter_name)
        for feature_name in FEATURES:
            for model_name in MODELS:
                config = copy.deepcopy(BASE_CONFIG)
                config["dsp"]["filter"] = filter_name
                config["features"]["mode"] = feature_name
                config["model"]["kind"] = model_name
                run_name = f"{model_name}_{filter_name}_{feature_name}"
                result = train_evaluate(tables[feature_name], config, OUT_DIR / run_name)
                matrix_rows.append({"model": model_name, "filter": filter_name, "features": feature_name, **result})
                print(f"{run_name}: f1={result['f1_macro']:.4f}, balanced_acc={result['balanced_accuracy']:.4f}, accuracy={result['accuracy']:.4f}")
    frame = pd.DataFrame(matrix_rows).sort_values("f1_macro", ascending=False).reset_index(drop=True)
    frame.to_csv(OUT_DIR / "full_matrix.csv", index=False)
    (OUT_DIR / "summary.json").write_text(json.dumps(frame.to_dict(orient="records"), indent=2, default=float), encoding="utf-8")
    print("\nTop configurations:")
    print(frame[["model", "filter", "features", "f1_macro", "balanced_accuracy", "accuracy"]].head(10).to_string(index=False))
    print("Saved artifacts/full_matrix_24/full_matrix.csv")


if __name__ == "__main__":
    main()
