"""End-to-end patient-level experiment utilities."""

from __future__ import annotations

from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from .dsp import add_noise, apply_filter, design_filter, feature_vector, quantize, resample_signal, segment_signal, wavelet_denoise
from .io import build_manifest, iter_patients, load_wav
from .models import make_model


def patient_split(manifest: pd.DataFrame, seed: int = 42, train_size: float = 0.70, validation_size: float = 0.15):
    patients = manifest[["patient_id", "label"]].drop_duplicates().reset_index(drop=True)
    train, rest = train_test_split(patients, train_size=train_size, stratify=patients["label"], random_state=seed)
    relative_val = validation_size / (1.0 - train_size)
    validation, test = train_test_split(rest, train_size=relative_val, stratify=rest["label"], random_state=seed)
    return tuple(part["patient_id"].tolist() for part in (train, validation, test))


def _patient_features(patient, config: dict) -> np.ndarray:
    dsp = config["dsp"]
    features = []
    spec = design_filter(dsp["target_fs"], dsp["filter"], dsp["low_hz"], dsp["high_hz"], dsp["order"], dsp["fir_taps"])
    recording_errors = []
    for recording in patient.recordings:
        try:
            source_fs, raw = load_wav(recording.wav_path)
            x = resample_signal(raw, source_fs, int(dsp["target_fs"]))
            quantization_bits = dsp.get("quantization_bits")
            if quantization_bits is not None:
                x = quantize(x, int(quantization_bits))
            if config.get("noise", {}).get("enabled", False):
                x = add_noise(x, float(config["noise"]["snr_db"]), config["noise"]["kind"], int(config["split"]["seed"]))
            if dsp.get("wavelet_denoise", False):
                x = wavelet_denoise(x)
            x = apply_filter(x, spec)
            segments = segment_signal(x, int(dsp["target_fs"]), float(dsp["segment_seconds"]), float(dsp["segment_hop_seconds"]))
            vectors = [feature_vector(segment, int(dsp["target_fs"]), **config["features"]) for segment in segments]
            if vectors:
                features.append(np.mean(vectors, axis=0))
        except Exception as exc:  # malformed/incomplete public-dataset files should not abort the cohort
            recording_errors.append(f"{recording.wav_path.name}: {exc}")
    if not features:
        detail = "; ".join(recording_errors) if recording_errors else "no recording references"
        raise ValueError(f"No readable recordings for patient {patient.patient_id} ({detail})")
    return np.mean(features, axis=0).astype(np.float32)


def _patient_feature_row(patient, config: dict) -> tuple[str, str, list[float] | None, str | None]:
    """Compute one patient row, returning an error instead of failing a worker batch."""
    try:
        return patient.patient_id, patient.label, _patient_features(patient, config).tolist(), None
    except ValueError as exc:
        return patient.patient_id, patient.label, None, str(exc)


def make_patient_table(data_dir: str | Path, config: dict, max_patients: int | None = None) -> pd.DataFrame:
    rows = []
    skip_invalid = bool(config.get("data", {}).get("skip_invalid_patients", False))
    patients = []
    for patient in iter_patients(data_dir):
        if patient.label == "Unknown" and not config["data"].get("include_unknown", False):
            continue
        patients.append(patient)
        if max_patients is not None and len(patients) >= max_patients:
            break
    n_jobs = max(1, int(config.get("data", {}).get("n_jobs", 1)))
    if n_jobs == 1:
        computed = [_patient_feature_row(patient, config) for patient in patients]
    else:
        computed = Parallel(n_jobs=n_jobs, backend="loky", batch_size=1)(
            delayed(_patient_feature_row)(patient, config) for patient in patients
        )
    for patient_id, label, features, error in computed:
        if features is None:
            if not skip_invalid:
                raise ValueError(error or f"No readable recordings for patient {patient_id}")
            warnings.warn(error or f"No readable recordings for patient {patient_id}", RuntimeWarning, stacklevel=2)
            continue
        rows.append({"patient_id": patient_id, "label": label, "features": features})
    return pd.DataFrame(rows)


def train_evaluate(table: pd.DataFrame, config: dict, output_dir: str | Path) -> dict:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    ids_train, ids_val, ids_test = patient_split(table, config["split"]["seed"], config["split"]["train_size"], config["split"]["validation_size"])
    train = table[table.patient_id.isin(ids_train)]
    test = table[table.patient_id.isin(ids_test)]
    x_train = np.asarray(train.features.tolist(), dtype=np.float32)
    x_test = np.asarray(test.features.tolist(), dtype=np.float32)
    y_train = train.label.to_numpy()
    y_test = test.label.to_numpy()
    model = make_model(config["model"]["kind"], config["model"]["seed"])
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    result = {
        "n_train": int(len(train)),
        "n_validation": int(len(ids_val)),
        "n_test": int(len(test)),
        "accuracy": float(accuracy_score(y_test, prediction)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, prediction)),
        "precision_macro": float(precision_score(y_test, prediction, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, prediction, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, prediction, average="macro", zero_division=0)),
        "labels": sorted(set(y_test)),
        "confusion_matrix": confusion_matrix(y_test, prediction, labels=sorted(set(y_test))).tolist(),
    }
    joblib.dump({"model": model, "config": config}, output_dir / "model.joblib")
    (output_dir / "metrics.json").write_text(pd.Series(result).to_json(indent=2), encoding="utf-8")
    return result
