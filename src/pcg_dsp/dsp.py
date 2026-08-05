"""Signal-processing front-end used by the PCG experiments."""

from __future__ import annotations

import numpy as np
import scipy.signal as signal
from scipy.stats import kurtosis, skew


def quantize(x: np.ndarray, bits: int) -> np.ndarray:
    if bits < 2:
        raise ValueError("bits must be >= 2")
    levels = float(2**bits - 1)
    clipped = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
    return (np.round((clipped + 1.0) * levels / 2.0) * 2.0 / levels - 1.0).astype(np.float32)


def resample_signal(x: np.ndarray, source_fs: int, target_fs: int) -> np.ndarray:
    if source_fs == target_fs:
        return np.asarray(x, dtype=np.float32)
    gcd = int(np.gcd(source_fs, target_fs))
    return signal.resample_poly(x, target_fs // gcd, source_fs // gcd).astype(np.float32)


def design_filter(
    fs: int,
    kind: str,
    low_hz: float = 25.0,
    high_hz: float = 400.0,
    order: int = 4,
    fir_taps: int = 129,
):
    kind = kind.lower()
    if kind in {"none", "off"}:
        return None
    nyquist = fs / 2.0
    high_hz = min(float(high_hz), 0.9 * nyquist)
    low_hz = max(1.0, min(float(low_hz), 0.5 * high_hz))
    if kind in {"butterworth", "iir"}:
        return ("sos", signal.butter(order, [low_hz, high_hz], btype="bandpass", fs=fs, output="sos"))
    if kind in {"fir", "linear_phase_fir"}:
        taps = int(fir_taps)
        if taps % 2 == 0:
            taps += 1
        return ("fir", signal.firwin(taps, [low_hz, high_hz], pass_zero=False, fs=fs, window="hamming"))
    raise ValueError(f"Unknown filter kind: {kind}")


def apply_filter(x: np.ndarray, spec) -> np.ndarray:
    if spec is None:
        return np.asarray(x, dtype=np.float32)
    kind, coefficients = spec
    if kind == "sos":
        return signal.sosfiltfilt(coefficients, x).astype(np.float32)
    return signal.filtfilt(coefficients, [1.0], x).astype(np.float32)


def add_noise(x: np.ndarray, snr_db: float, kind: str = "white", seed: int = 42) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.asarray(x, dtype=np.float32)
    power = float(np.mean(x**2) + 1e-12)
    noise = rng.normal(size=x.size).astype(np.float32)
    if kind == "pink":
        noise = np.cumsum(noise)
        noise -= noise.mean()
    elif kind in {"impulse", "click"}:
        noise.fill(0.0)
        count = max(1, x.size // 5000)
        indices = rng.choice(x.size, size=min(count, x.size), replace=False)
        noise[indices] = rng.normal(0, 1, size=len(indices))
    noise_power = float(np.mean(noise**2) + 1e-12)
    scale = np.sqrt(power / (noise_power * 10 ** (snr_db / 10.0)))
    return (x + scale * noise).astype(np.float32)


def wavelet_denoise(x: np.ndarray, wavelet: str = "db4", level: int | None = None) -> np.ndarray:
    """Soft-threshold detail coefficients while retaining transient PCG structure."""
    try:
        import pywt
    except ImportError as exc:  # pragma: no cover - dependency is pinned in requirements
        raise RuntimeError("Install PyWavelets to use wavelet denoising") from exc
    x = np.asarray(x, dtype=np.float32)
    wavelet_obj = pywt.Wavelet(wavelet)
    max_level = pywt.dwt_max_level(len(x), wavelet_obj.dec_len)
    if max_level < 1:
        return x.copy()
    level = max(1, min(level or max_level, max_level))
    coefficients = pywt.wavedec(x, wavelet_obj, level=level)
    sigma = np.median(np.abs(coefficients[-1])) / 0.6745 + 1e-12
    threshold = sigma * np.sqrt(2.0 * np.log(len(x)))
    filtered = [coefficients[0]] + [pywt.threshold(detail, threshold, mode="soft") for detail in coefficients[1:]]
    return pywt.waverec(filtered, wavelet_obj)[: len(x)].astype(np.float32)


def stft_matrix(x: np.ndarray, fs: int, n_fft: int = 512, hop_length: int = 128) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frequencies, times, matrix = signal.stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft - hop_length, boundary=None)
    magnitude = np.abs(matrix).astype(np.float32)
    log_magnitude = np.log1p(magnitude / (magnitude.max() + 1e-12))
    return frequencies.astype(np.float32), times.astype(np.float32), log_magnitude


def _stft_features(x: np.ndarray, fs: int, n_fft: int = 512, hop_length: int = 128) -> list[float]:
    _, _, matrix = stft_matrix(x, fs, n_fft, hop_length)
    return [float(matrix.mean()), float(matrix.std()), float(np.quantile(matrix, 0.25)), float(np.quantile(matrix, 0.75))]


def _fft_features(x: np.ndarray, fs: int, n_fft: int = 512) -> list[float]:
    spectrum = np.abs(np.fft.rfft(x, n=n_fft)).astype(np.float64)
    frequencies = np.fft.rfftfreq(n_fft, 1.0 / fs)
    order = np.argsort(spectrum[1:])[-3:] + 1
    return [float(frequencies[index]) for index in order] + [float(spectrum[index] / (spectrum.max() + 1e-12)) for index in order]


def segment_signal(x: np.ndarray, fs: int, seconds: float = 3.0, hop_seconds: float = 1.5) -> list[np.ndarray]:
    size = max(1, int(seconds * fs))
    hop = max(1, int(hop_seconds * fs))
    if x.size <= size:
        return [np.pad(x, (0, size - x.size))]
    starts = range(0, x.size - size + 1, hop)
    segments = [x[start : start + size] for start in starts]
    if not segments:
        segments = [np.pad(x, (0, max(0, size - x.size)))]
    return segments


def _spectral_features(x: np.ndarray, fs: int, n_fft: int = 512) -> list[float]:
    frequencies, power = signal.welch(x, fs=fs, nperseg=min(n_fft, len(x)))
    power = np.maximum(power, 1e-12)
    total = float(power.sum())
    edges = (20.0, 100.0, 200.0, 400.0, min(800.0, fs / 2.0))
    features = [float(power[(frequencies >= lo) & (frequencies < hi)].sum() / total) for lo, hi in zip(edges, edges[1:])]
    probability = power / total
    features.append(float(-np.sum(probability * np.log(probability)) / np.log(len(probability))))
    return features


def _mfcc_features(x: np.ndarray, fs: int, n_mfcc: int = 13, n_fft: int = 512, hop_length: int = 128) -> list[float]:
    try:
        import librosa

        mfcc = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(float).tolist()
    except Exception:
        return []


def feature_vector(
    x: np.ndarray,
    fs: int,
    mode: str = "hybrid",
    n_mfcc: int = 13,
    n_fft: int = 512,
    hop_length: int = 128,
) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    rms = float(np.sqrt(np.mean(x**2) + 1e-12))
    base = [float(np.mean(x)), float(np.std(x)), rms, float(skew(x)), float(kurtosis(x))]
    spectral = _spectral_features(x, fs, n_fft)
    fft = _fft_features(x, fs, n_fft)
    stft = _stft_features(x, fs, n_fft, hop_length)
    mfcc = _mfcc_features(x, fs, n_mfcc, n_fft, hop_length)
    mode = mode.lower()
    if mode == "stats":
        values = base
    elif mode == "psd":
        values = base + spectral + fft
    elif mode == "mfcc":
        values = base + mfcc
    elif mode == "stft":
        values = base + stft
    elif mode in {"hybrid", "all"}:
        values = base + spectral + fft + stft + mfcc
    else:
        raise ValueError(f"Unknown feature mode: {mode}")
    return np.nan_to_num(np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
