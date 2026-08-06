"""MobileNetV3 inference for the spectrogram CNN benchmark.

The checkpoint format is produced by the historical ``run_cnn_mobilenet``
experiment.  Torch is imported lazily so the classical DSP/Streamlit path can
still run in environments that do not install the optional CNN dependencies.
"""

from __future__ import annotations

import copy
import io
from pathlib import Path
from typing import Any

import numpy as np
from scipy import signal

from .dsp import add_noise, apply_filter, design_filter, quantize, resample_signal, segment_signal, stft_matrix
from .io import load_wav


CLASS_NAMES = ("Absent", "Present")


def _torch_modules():
    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as functional
        from torchvision.models import mobilenet_v3_small
    except ImportError as exc:  # pragma: no cover - depends on optional UI install
        raise RuntimeError(
            "MobileNet inference requires torch and torchvision. "
            "Install requirements-ui.txt or the optional cnn dependencies."
        ) from exc
    return torch, nn, functional, mobilenet_v3_small


def _torch_load(path: str | Path, torch):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # torch < 2.0 has no weights_only keyword
        return torch.load(path, map_location="cpu")


def load_mobilenet_bundle(model_path: str | Path) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a binary MobileNetV3-Small checkpoint and its DSP metadata."""
    torch, nn, _, mobilenet_v3_small = _torch_modules()
    saved = _torch_load(model_path, torch)
    if not isinstance(saved, dict) or "state_dict" not in saved:
        raise ValueError("MobileNet checkpoint must contain a state_dict")

    args = dict(saved.get("args", {}))
    result = dict(saved.get("result", {}))
    class_names = tuple(result.get("class_names", CLASS_NAMES))
    if len(class_names) != 2:
        raise ValueError("The Streamlit demo currently supports the binary MobileNet checkpoint only")

    model = mobilenet_v3_small(weights=None)
    in_features = model.classifier[3].in_features
    model.classifier[3] = nn.Linear(in_features, len(class_names))
    model.load_state_dict(saved["state_dict"], strict=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    target_fs = int(args.get("target_fs", result.get("target_fs", 1000)))
    filter_name = str(args.get("filter_name", result.get("filter", "fir")))
    quantization_bits = args.get("quantization_bits", result.get("quantization_bits", 16))
    image_size = int(args.get("image_size", result.get("image_size", 128)))
    metadata = {
        "model": "MobileNetV3-Small",
        "class_names": list(class_names),
        "target_fs": target_fs,
        "filter": filter_name,
        "quantization_bits": quantization_bits,
        "image_size": image_size,
        "max_windows_per_recording": int(args.get("max_windows_per_recording", result.get("max_windows_per_recording", 2))),
        "aggregation": str(args.get("aggregation", result.get("aggregation", "mean"))),
        "threshold": float(result.get("best_threshold", 0.5)),
        "device": str(device),
        "checkpoint_result": result,
    }
    bundle = {"model": model, "metadata": metadata, "torch": torch, "device": device}
    config = {
        "dsp": {
            "target_fs": target_fs,
            "quantization_bits": quantization_bits,
            "filter": filter_name,
            "low_hz": 25.0,
            "high_hz": 400.0,
            "order": 4,
            "fir_taps": 129,
            "wavelet_denoise": False,
            "segment_seconds": 3.0,
            "segment_hop_seconds": 1.5,
        },
        "features": {"mode": "log-stft", "image_size": image_size},
        "model": {"kind": "mobilenet_v3_small", "aggregation": metadata["aggregation"], "threshold": metadata["threshold"]},
        "noise": {"enabled": False, "snr_db": 10, "kind": "white"},
    }
    return bundle, config


def _spectrogram_image(x: np.ndarray, fs: int, image_size: int, torch, functional) -> Any:
    """Reproduce the benchmark's normalized 3-channel log-STFT image."""
    frequencies, _, matrix = signal.stft(
        x,
        fs,
        nperseg=min(256, len(x)),
        noverlap=min(192, max(0, len(x) - 1)),
        nfft=512,
        boundary=None,
    )
    mask = (frequencies >= 20.0) & (frequencies <= min(400.0, fs / 2.0))
    power = np.log1p(np.abs(matrix[mask])).astype(np.float32)
    if power.size == 0:
        power = np.zeros((32, 32), dtype=np.float32)
    lo, hi = np.percentile(power, [1.0, 99.0])
    image = np.clip((power - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    tensor = functional.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)
    tensor = tensor.repeat(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((tensor - mean) / std).float()


def _load_source(source: str | Path | bytes | bytearray | io.BytesIO) -> tuple[int, np.ndarray]:
    if isinstance(source, (bytes, bytearray)):
        source = io.BytesIO(source)
    if hasattr(source, "seek"):
        source.seek(0)
    return load_wav(source)


def _merge_config(config: dict[str, Any], overrides: dict[str, Any] | None) -> dict[str, Any]:
    merged = copy.deepcopy(config)
    if overrides:
        if "dsp" in overrides or "noise" in overrides:
            merged.setdefault("dsp", {}).update(overrides.get("dsp", {}))
            merged.setdefault("noise", {}).update(overrides.get("noise", {}))
        else:
            merged.setdefault("dsp", {}).update(overrides)
    return merged


def analyze_mobilenet_recording(
    source: str | Path | bytes | bytearray | io.BytesIO,
    bundle: dict[str, Any],
    config: dict[str, Any],
    dsp_overrides: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run the same DSP + log-STFT + MobileNet pipeline used in the benchmark."""
    torch = bundle["torch"]
    _, _, functional, _ = _torch_modules()
    config = _merge_config(config, dsp_overrides)
    dsp = config["dsp"]
    source_fs, raw = _load_source(source)
    target_fs = int(dsp["target_fs"])
    resampled = resample_signal(raw, source_fs, target_fs)
    bits = dsp.get("quantization_bits")
    quantized = quantize(resampled, int(bits)) if bits is not None else np.asarray(resampled, dtype=np.float32)
    processed = quantized
    if config.get("noise", {}).get("enabled", False):
        processed = add_noise(processed, float(config["noise"]["snr_db"]), config["noise"]["kind"], 42)
    filtered = apply_filter(processed, design_filter(target_fs, dsp["filter"], dsp["low_hz"], dsp["high_hz"], dsp["order"], dsp["fir_taps"]))
    segments = segment_signal(filtered, target_fs, float(dsp["segment_seconds"]), float(dsp["segment_hop_seconds"]))
    max_windows = int(bundle["metadata"]["max_windows_per_recording"])
    if max_windows and len(segments) > max_windows:
        indices = np.linspace(0, len(segments) - 1, max_windows, dtype=int)
        segments = [segments[index] for index in indices]

    images = torch.stack([_spectrogram_image(segment, target_fs, int(bundle["metadata"]["image_size"]), torch, functional) for segment in segments])
    with torch.no_grad():
        probabilities = torch.softmax(bundle["model"](images.to(bundle["device"])), dim=1).cpu().numpy()
    window_mean = np.mean(probabilities, axis=0)
    threshold = float(bundle["metadata"]["threshold"])
    prediction = "Present" if float(window_mean[1]) >= threshold else "Absent"

    fft_frequency = np.fft.rfftfreq(len(filtered), 1.0 / target_fs)
    fft_magnitude = np.abs(np.fft.rfft(filtered)) / max(1, len(filtered))
    psd_frequency, psd_power = signal.welch(filtered, fs=target_fs, nperseg=min(1024, len(filtered)))
    stft_frequency, stft_time, stft_magnitude = stft_matrix(filtered, target_fs, 512, 128)
    return {
        "label": prediction,
        "probabilities": {"Absent": float(window_mean[0]), "Present": float(window_mean[1])},
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
        "feature_count": f"log-STFT {int(bundle['metadata']['image_size'])}x{int(bundle['metadata']['image_size'])}",
        "config": config,
        "model_name": "MobileNetV3-Small",
        "cnn_windows": len(segments),
    }
