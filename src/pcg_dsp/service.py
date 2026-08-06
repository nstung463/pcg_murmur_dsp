"""Shared inference/visualization backend for the CLI and web demo."""

from __future__ import annotations

import copy
import io
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import yaml
from scipy import signal

from .dsp import apply_filter, design_filter, feature_vector, quantize, resample_signal, segment_signal, stft_matrix, wavelet_denoise
from .io import load_wav


def load_model_bundle(model_path: str | Path, config_path: str | Path | None = None) -> tuple[Any, dict]:
    """Load a saved estimator and its training-time DSP configuration."""
    saved = joblib.load(model_path)
    model = saved["model"] if isinstance(saved, dict) and "model" in saved else saved
    if isinstance(saved, dict) and isinstance(saved.get("config"), dict):
        config = copy.deepcopy(saved["config"])
    else:
        if config_path is None:
            config_path = "configs/default.yaml"
        config = yaml.safe_load(Path(config_path).read_text(encoding="utf-8"))
    return model, config


def _load_source(source: str | Path | bytes | bytearray | io.BytesIO) -> tuple[int, np.ndarray]:
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    if hasattr(source, "seek"):
        source.seek(0)
    return load_wav(source)


def _apply_dsp_overrides(config: dict, overrides: dict | None) -> dict:
    result = copy.deepcopy(config)
    if overrides:
        # Flat mappings are kept convenient for callers that only change DSP;
        # nested mappings allow the UI to toggle controlled noise experiments.
        if "dsp" in overrides or "noise" in overrides:
            result.setdefault("dsp", {}).update(overrides.get("dsp", {}))
            result.setdefault("noise", {}).update(overrides.get("noise", {}))
        else:
            result.setdefault("dsp", {}).update(overrides)
    return result


def analyze_recording(
    source: str | Path | bytes | bytearray | io.BytesIO,
    model: Any,
    config: dict,
    dsp_overrides: dict | None = None,
) -> dict[str, Any]:
    """Run one WAV through the exact training pipeline and return plot-ready arrays."""
    config = _apply_dsp_overrides(config, dsp_overrides)
    dsp = config["dsp"]
    source_fs, raw = _load_source(source)
    target_fs = int(dsp["target_fs"])
    resampled = resample_signal(raw, source_fs, target_fs)
    quantization_bits = dsp.get("quantization_bits")
    quantized = (
        quantize(resampled, int(quantization_bits))
        if quantization_bits is not None
        else np.asarray(resampled, dtype=np.float32)
    )
    processed = quantized
    if config.get("noise", {}).get("enabled", False):
        from .dsp import add_noise

        processed = add_noise(processed, float(config["noise"]["snr_db"]), config["noise"]["kind"], int(config.get("split", {}).get("seed", 42)))
    if dsp.get("wavelet_denoise", False):
        processed = wavelet_denoise(processed)
    filtered = apply_filter(processed, design_filter(target_fs, dsp["filter"], dsp["low_hz"], dsp["high_hz"], dsp["order"], dsp["fir_taps"]))
    segments = segment_signal(filtered, target_fs, float(dsp["segment_seconds"]), float(dsp["segment_hop_seconds"]))
    vectors = [feature_vector(segment, target_fs, **config["features"]) for segment in segments]
    features = np.mean(vectors, axis=0, keepdims=True)
    prediction = str(model.predict(features)[0])
    probabilities: dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        probabilities = {str(name): float(probability) for name, probability in zip(model.classes_, model.predict_proba(features)[0])}

    fft_frequency = np.fft.rfftfreq(len(filtered), 1.0 / target_fs)
    fft_magnitude = np.abs(np.fft.rfft(filtered)) / max(1, len(filtered))
    psd_frequency, psd_power = signal.welch(filtered, fs=target_fs, nperseg=min(1024, len(filtered)))
    stft_frequency, stft_time, stft_magnitude = stft_matrix(filtered, target_fs, int(config["features"]["n_fft"]), int(config["features"]["hop_length"]))
    return {
        "label": prediction,
        "probabilities": probabilities,
        "source_fs": source_fs,
        "target_fs": target_fs,
        "duration_seconds": float(len(raw) / source_fs),
        "raw_signal": raw,
        "resampled_signal": resampled,
        "quantized_signal": quantized,
        "pre_filter_signal": processed,
        "filtered_signal": filtered,
        "fft_frequency": fft_frequency,
        "fft_magnitude": fft_magnitude,
        "psd_frequency": psd_frequency,
        "psd_power": psd_power,
        "stft_frequency": stft_frequency,
        "stft_time": stft_time,
        "stft_magnitude": stft_magnitude,
        "n_segments": len(segments),
        "feature_count": int(features.shape[1]),
        "feature_matrix": features,
        "config": config,
    }


def analyze_patient_recordings(
    recording_paths: list[str | Path],
    model: Any,
    config: dict,
    dsp_overrides: dict | None = None,
    patient_id: str | None = None,
) -> dict[str, Any]:
    """Aggregate classical-model inference across all recordings of one patient.

    Training averages segment features within each recording and then averages
    those recording vectors at patient level. This helper mirrors that contract
    while retaining the first recording's signal arrays for visualization.
    """
    if not recording_paths:
        raise ValueError("No recordings were provided for patient-level inference")
    recording_results = [analyze_recording(path, model, config, dsp_overrides) for path in recording_paths]
    patient_features = np.mean(
        np.stack([np.mean(result["feature_matrix"], axis=0) for result in recording_results], axis=0),
        axis=0,
    ).reshape(1, -1)
    prediction = str(model.predict(patient_features)[0])
    probabilities: dict[str, float] = {}
    if hasattr(model, "predict_proba"):
        probabilities = {
            str(name): float(probability)
            for name, probability in zip(model.classes_, model.predict_proba(patient_features)[0])
        }
    representative = copy.deepcopy(recording_results[0])
    representative.update(
        {
            "label": prediction,
            "probabilities": probabilities,
            "n_segments": int(sum(result["n_segments"] for result in recording_results)),
            "feature_count": int(patient_features.shape[1]),
            "feature_matrix": patient_features,
            "duration_seconds": float(sum(result["duration_seconds"] for result in recording_results)),
            "recording_count": len(recording_results),
            "patient_id": patient_id,
            "aggregation_level": "patient",
        }
    )
    return representative


def analyze_file(
    wav_path: str | Path,
    model_path: str | Path,
    config_path: str | Path | None = None,
    dsp_overrides: dict | None = None,
) -> dict[str, Any]:
    model, config = load_model_bundle(model_path, config_path)
    return analyze_recording(wav_path, model, config, dsp_overrides)
