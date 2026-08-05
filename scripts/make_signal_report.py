"""Render raw/filtered waveform, FFT, PSD and STFT figures for one recording."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy import signal

from pcg_dsp.dsp import apply_filter, design_filter, stft_matrix
from pcg_dsp.io import load_wav


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--wav", required=True)
    parser.add_argument("--output-dir", default="artifacts/signal_report")
    parser.add_argument("--target-fs", type=int, default=1000)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    fs, x = load_wav(args.wav)
    if fs != args.target_fs:
        from pcg_dsp.dsp import resample_signal

        x = resample_signal(x, fs, args.target_fs)
        fs = args.target_fs
    filtered = apply_filter(x, design_filter(fs, "butterworth", 25, 400))
    frequencies = np.fft.rfftfreq(len(x), 1.0 / fs)
    spectrum = np.abs(np.fft.rfft(x))
    psd_frequencies, psd = signal.welch(x, fs=fs, nperseg=min(1024, len(x)))
    stft_frequencies, times, stft = stft_matrix(x, fs)
    fig, axes = plt.subplots(4, 1, figsize=(12, 12), constrained_layout=True)
    time = np.arange(len(x)) / fs
    axes[0].plot(time, x, alpha=0.7, label="raw")
    axes[0].plot(time, filtered, alpha=0.8, label="Butterworth 25–400 Hz")
    axes[0].set(xlabel="Time (s)", ylabel="Amplitude", title="Raw and filtered PCG")
    axes[0].legend()
    axes[1].plot(frequencies, spectrum)
    axes[1].set(xlim=(0, min(800, fs / 2)), xlabel="Frequency (Hz)", ylabel="Magnitude", title="FFT magnitude")
    axes[2].semilogy(psd_frequencies, psd + 1e-12)
    axes[2].set(xlim=(0, min(800, fs / 2)), xlabel="Frequency (Hz)", ylabel="Power", title="Welch PSD")
    axes[3].pcolormesh(times, stft_frequencies, stft, shading="auto")
    axes[3].set(ylim=(0, min(800, fs / 2)), xlabel="Time (s)", ylabel="Frequency (Hz)", title="Log-STFT spectrogram")
    fig.savefig(output / "signal_report.png", dpi=180)
    print(f"Wrote {output / 'signal_report.png'}")


if __name__ == "__main__":
    main()
