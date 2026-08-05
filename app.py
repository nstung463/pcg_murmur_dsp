"""Streamlit front end for the PCG DSP inference backend."""

from __future__ import annotations

from pathlib import Path
import sys
import io

# Make `python -m streamlit run app.py` work directly from the project root,
# even when the package has not been installed in editable mode.
PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
import yaml
from scipy.io import wavfile

from pcg_dsp.service import analyze_recording, load_model_bundle


DEFAULT_MODEL = "artifacts/runs/svm_butterworth_hybrid/model.joblib"


def _plot_waveforms(result: dict):
    raw = result["raw_signal"]
    filtered = result["filtered_signal"]
    raw_t = np.arange(len(raw)) / result["source_fs"]
    filtered_t = np.arange(len(filtered)) / result["target_fs"]
    fig, axes = plt.subplots(2, 1, figsize=(11, 6), constrained_layout=True)
    axes[0].plot(raw_t, raw, linewidth=0.6, color="#64748b")
    axes[0].set_title("Raw waveform")
    axes[1].plot(filtered_t, filtered, linewidth=0.6, color="#0f766e")
    axes[1].set_title("Processed waveform")
    for axis in axes:
        axis.set_xlabel("Time (s)")
        axis.grid(alpha=0.2)
    return fig


def _plot_frequency_views(result: dict):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    axes[0].plot(result["fft_frequency"], result["fft_magnitude"], color="#7c3aed", linewidth=0.7)
    axes[0].set_xlim(0, min(800, result["target_fs"] / 2))
    axes[0].set_title("FFT magnitude")
    axes[0].set_xlabel("Frequency (Hz)")
    axes[0].grid(alpha=0.2)
    axes[1].semilogy(result["psd_frequency"], result["psd_power"] + 1e-12, color="#ea580c", linewidth=0.8)
    axes[1].set_xlim(0, min(800, result["target_fs"] / 2))
    axes[1].set_title("Welch PSD")
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].grid(alpha=0.2)
    return fig


def _plot_spectrogram(result: dict):
    fig, axis = plt.subplots(figsize=(11, 4), constrained_layout=True)
    image = axis.pcolormesh(result["stft_time"], result["stft_frequency"], result["stft_magnitude"], shading="auto", cmap="magma")
    axis.set_ylim(0, min(800, result["target_fs"] / 2))
    axis.set_xlabel("Time (s)")
    axis.set_ylabel("Frequency (Hz)")
    axis.set_title("Log-STFT spectrogram")
    fig.colorbar(image, ax=axis, label="Log magnitude")
    return fig


def _wav_bytes(signal: np.ndarray, sample_rate: int, normalize: bool = True) -> bytes:
    """Encode a float waveform as a browser-playable PCM WAV."""
    values = np.asarray(signal, dtype=np.float32)
    if normalize:
        peak = float(np.max(np.abs(values))) if values.size else 0.0
        if peak > 1e-9:
            values = values / peak
    values = np.clip(values, -1.0, 1.0)
    buffer = io.BytesIO()
    wavfile.write(buffer, int(sample_rate), (values * 32767.0).astype(np.int16))
    return buffer.getvalue()


st.set_page_config(page_title="PCG DSP Demo", page_icon="♥", layout="wide")
st.title("PCG Murmur Detection — DSP Demo")
st.caption("Educational research prototype. This is not a clinical diagnostic device.")

with st.sidebar:
    st.header("Model and DSP")
    model_path = st.text_input("Model path", DEFAULT_MODEL)
    uploaded = st.file_uploader("Upload a heart-sound WAV", type=["wav"])
    model_exists = Path(model_path).exists()
    if not model_exists:
        st.warning("Model file not found. Train a model or update the path.")

    config = None
    model = None
    if model_exists:
        try:
            model, config = load_model_bundle(model_path)
        except Exception as exc:
            st.error(f"Cannot load model: {exc}")

    if config:
        dsp = config["dsp"]
        target_fs = st.selectbox("Target sampling rate", [1000, 2000, 4000], index=[1000, 2000, 4000].index(int(dsp["target_fs"])))
        bit_options = [None, 8, 12, 16]
        bit_labels = {None: "Original / none", 8: "8-bit", 12: "12-bit", 16: "16-bit"}
        bits = st.selectbox(
            "Quantization",
            bit_options,
            index=bit_options.index(dsp.get("quantization_bits", 16)),
            format_func=lambda value: bit_labels[value],
        )
        filter_name = st.selectbox("Band-pass filter", ["none", "butterworth", "fir"], index=["none", "butterworth", "fir"].index(dsp["filter"]))
        noise_kind = st.selectbox("Noise preview", ["none", "white", "pink", "impulse"])
        snr_db = st.slider("Noise SNR (dB)", 0, 30, int(config.get("noise", {}).get("snr_db", 10)))
        analyze_clicked = st.button("Analyze recording", type="primary", disabled=uploaded is None)
    else:
        analyze_clicked = False

if analyze_clicked and uploaded is not None and model is not None and config is not None:
    overrides = {
        "dsp": {"target_fs": target_fs, "quantization_bits": bits, "filter": filter_name},
        "noise": {"enabled": noise_kind != "none", "kind": noise_kind, "snr_db": snr_db},
    }
    with st.spinner("Running DSP pipeline and inference..."):
        try:
            st.session_state["analysis"] = analyze_recording(uploaded.getvalue(), model, config, overrides)
            st.session_state["filename"] = uploaded.name
            st.session_state["audio_bytes"] = uploaded.getvalue()
        except Exception as exc:
            st.error(f"Analysis failed: {exc}")

result = st.session_state.get("analysis")
if result:
    if st.session_state.get("audio_bytes"):
        st.audio(st.session_state["audio_bytes"], format="audio/wav")
    st.subheader(f"Prediction: {result['label']}")
    metric_columns = st.columns(4)
    metric_columns[0].metric("Prediction", result["label"])
    metric_columns[1].metric("Duration", f"{result['duration_seconds']:.2f} s")
    metric_columns[2].metric("Segments", result["n_segments"])
    metric_columns[3].metric("Features", result["feature_count"])

    if result["probabilities"]:
        st.write("Class probabilities")
        st.bar_chart(result["probabilities"])

    st.subheader("Listen to DSP stages")
    st.caption("Mỗi lựa chọn là cùng một recording sau một layer DSP. Playback được normalize riêng để dễ nghe; waveform/metrics vẫn dùng giá trị xử lý thật.")
    audio_stages = {
        "Raw WAV": (result["raw_signal"], result["source_fs"], "Original PCM at source sample rate"),
        "After resample": (result["resampled_signal"], result["target_fs"], f"Resampled to {result['target_fs']} Hz"),
        "After quantization": (result["quantized_signal"], result["target_fs"], "Amplitude quantized at configured bit depth"),
        "After band-pass filter": (result["filtered_signal"], result["target_fs"], "Final waveform used for segmentation and features"),
    }
    selected_stage = st.selectbox("Audio preview", list(audio_stages), key="audio_stage")
    normalize_playback = st.checkbox("Normalize playback volume", value=True, key="normalize_playback")
    preview_signal, preview_fs, preview_description = audio_stages[selected_stage]
    st.audio(_wav_bytes(preview_signal, preview_fs, normalize_playback), format="audio/wav")
    st.caption(f"{preview_description} · {len(preview_signal) / preview_fs:.2f} s · {preview_fs} Hz")

    st.pyplot(_plot_waveforms(result), clear_figure=True)
    st.pyplot(_plot_frequency_views(result), clear_figure=True)
    st.pyplot(_plot_spectrogram(result), clear_figure=True)

    with st.expander("DSP configuration used"):
        st.code(yaml.safe_dump(result["config"], sort_keys=False), language="yaml")
else:
    st.info("Upload a WAV file, choose the DSP settings, and click Analyze recording.")
