"""Streamlit front end for the PCG DSP inference backend."""

from __future__ import annotations

from pathlib import Path
import sys
import io
import json

# Make `python -m streamlit run app.py` work directly from the project root,
# even when the package has not been installed in editable mode.
PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import yaml
from scipy import signal
from scipy.io import wavfile

from pcg_dsp.dsp import design_filter
from pcg_dsp.service import analyze_recording, load_model_bundle


DEFAULT_MODEL = "artifacts/runs/svm_butterworth_hybrid/model.joblib"
DEMO_DATA_ROOT = PROJECT_ROOT / "data" / "circor-heart-sound" / "1.0.3" / "training_data"


@st.cache_resource(show_spinner=False)
def _cached_model_bundle(model_path: str):
    """Cache the estimator so widget reruns do not reload joblib from disk."""
    return load_model_bundle(model_path)


def _find_demo_wav() -> Path | None:
    """Return one deterministic bundled recording for a no-file demo."""
    if not DEMO_DATA_ROOT.exists():
        return None
    candidates = sorted(DEMO_DATA_ROOT.glob("*.wav"))
    return candidates[0] if candidates else None


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


def _plot_filter_response(result: dict):
    """Show the selected filter's frequency response against the PCG band."""
    fs = int(result["target_fs"])
    dsp = result["config"]["dsp"]
    spec = design_filter(
        fs,
        dsp["filter"],
        dsp["low_hz"],
        dsp["high_hz"],
        dsp["order"],
        dsp["fir_taps"],
    )
    if spec is None:
        frequencies = np.linspace(0.0, fs / 2.0, 2048)
        magnitude_db = np.zeros_like(frequencies)
    elif spec[0] == "sos":
        frequencies, response = signal.sosfreqz(spec[1], worN=2048, fs=fs)
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-8))
    else:
        frequencies, response = signal.freqz(spec[1], worN=2048, fs=fs)
        magnitude_db = 20.0 * np.log10(np.maximum(np.abs(response), 1e-8))

    fig, axis = plt.subplots(figsize=(11, 3.5), constrained_layout=True)
    axis.plot(frequencies, magnitude_db, color="#7c3aed", linewidth=1.2)
    axis.axvspan(float(dsp["low_hz"]), float(dsp["high_hz"]), color="#0f766e", alpha=0.12, label="PCG band 25–400 Hz")
    axis.axhline(-3.0, color="#ea580c", linestyle="--", linewidth=0.8, label="−3 dB")
    axis.set_xlim(0, min(800, fs / 2.0))
    axis.set_ylim(max(-80.0, float(np.nanmin(magnitude_db)) - 3.0), 5.0)
    axis.set_xlabel("Frequency (Hz)")
    axis.set_ylabel("Magnitude (dB)")
    axis.set_title(f"{dsp['filter'].title()} filter frequency response")
    axis.grid(alpha=0.2)
    axis.legend(loc="lower right")
    return fig


def _top_confidence(result: dict) -> tuple[str, float]:
    probabilities = result.get("probabilities") or {}
    if not probabilities:
        return str(result["label"]), float("nan")
    label, probability = max(probabilities.items(), key=lambda item: item[1])
    return str(label), float(probability)


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


st.set_page_config(page_title="PCG DSP Demo", page_icon=":material/graphic_eq:", layout="wide")
st.title("PCG Murmur Detection — DSP Demo")
st.caption("Educational research prototype. This is not a clinical diagnostic device.")

demo_wav = _find_demo_wav()
with st.sidebar:
    st.header("Model and DSP")
    model_path = st.text_input("Model path", DEFAULT_MODEL)
    uploaded = st.file_uploader("Upload a heart-sound WAV", type=["wav"])
    if uploaded is not None:
        st.session_state["input_bytes"] = uploaded.getvalue()
        st.session_state["filename"] = uploaded.name
        st.session_state["input_source"] = "uploaded WAV"
    if demo_wav is not None and st.button("Use demo recording", icon=":material/play_circle:"):
        st.session_state["input_bytes"] = demo_wav.read_bytes()
        st.session_state["filename"] = demo_wav.name
        st.session_state["input_source"] = "bundled demo WAV"
    if st.session_state.get("input_source") == "bundled demo WAV":
        st.caption(f"Demo file: `{demo_wav.name if demo_wav else 'not found'}`")

    model_exists = Path(model_path).exists()
    if not model_exists:
        st.warning("Model file not found. Train a model or update the path.")

    config = None
    model = None
    if model_exists:
        try:
            model, config = _cached_model_bundle(model_path)
        except Exception as exc:
            st.error(f"Cannot load model: {exc}")

    if config:
        dsp = config["dsp"]
        bit_options = [None, 8, 12, 16]
        bit_labels = {None: "Original / none", 8: "8-bit", 12: "12-bit", 16: "16-bit"}
        with st.form("analysis_controls", border=False):
            target_fs = st.selectbox("Target sampling rate", [1000, 2000, 4000], index=[1000, 2000, 4000].index(int(dsp["target_fs"])))
            bits = st.selectbox(
                "Quantization",
                bit_options,
                index=bit_options.index(dsp.get("quantization_bits", 16)),
                format_func=lambda value: bit_labels[value],
            )
            filter_name = st.selectbox("Band-pass filter", ["none", "butterworth", "fir"], index=["none", "butterworth", "fir"].index(dsp["filter"]))
            noise_kind = st.selectbox("Noise preview", ["none", "white", "pink", "impulse"])
            snr_db = st.slider("Noise SNR (dB)", 0, 30, int(config.get("noise", {}).get("snr_db", 10)))
            compare_baseline = st.checkbox("Compare against training baseline", value=True)
            analyze_clicked = st.form_submit_button(
                "Analyze recording",
                type="primary",
                icon=":material/analytics:",
                disabled=st.session_state.get("input_bytes") is None,
            )
    else:
        analyze_clicked = False
        compare_baseline = False

active_bytes = st.session_state.get("input_bytes")
if analyze_clicked and active_bytes is not None and model is not None and config is not None:
    overrides = {
        "dsp": {"target_fs": target_fs, "quantization_bits": bits, "filter": filter_name},
        "noise": {"enabled": noise_kind != "none", "kind": noise_kind, "snr_db": snr_db},
    }
    baseline_dsp = config["dsp"]
    baseline_overrides = {
        "dsp": {
            "target_fs": int(baseline_dsp["target_fs"]),
            "quantization_bits": baseline_dsp.get("quantization_bits"),
            "filter": baseline_dsp["filter"],
        },
        "noise": {"enabled": False, "kind": "white", "snr_db": 10},
    }
    with st.status("Running DSP pipeline and inference...", expanded=True) as status:
        try:
            st.write("Analyzing selected DSP configuration")
            selected_result = analyze_recording(active_bytes, model, config, overrides)
            baseline_result = None
            if compare_baseline:
                st.write("Running the training-baseline comparison")
                baseline_result = analyze_recording(active_bytes, model, config, baseline_overrides)
            st.session_state["analysis"] = selected_result
            st.session_state["baseline_analysis"] = baseline_result
            st.session_state["analysis_overrides"] = overrides
            st.session_state["audio_bytes"] = active_bytes
            status.update(label="Analysis complete", state="complete")
        except Exception as exc:
            status.update(label="Analysis failed", state="error")
            st.error(f"Analysis failed: {exc}")

result = st.session_state.get("analysis")
if result:
    filename = st.session_state.get("filename", "recording.wav")
    st.caption(f"Active recording: `{filename}` · {st.session_state.get('input_source', 'selected WAV')}")
    if st.session_state.get("audio_bytes"):
        st.audio(st.session_state["audio_bytes"], format="audio/wav")

    top_label, confidence = _top_confidence(result)
    st.subheader(f"Prediction: {result['label']}")
    metric_columns = st.columns(5)
    metric_columns[0].metric("Prediction", result["label"])
    metric_columns[1].metric("Top confidence", "—" if not np.isfinite(confidence) else f"{confidence:.1%}")
    metric_columns[2].metric("Duration", f"{result['duration_seconds']:.2f} s")
    metric_columns[3].metric("Segments", result["n_segments"])
    metric_columns[4].metric("Features", result["feature_count"])
    if np.isfinite(confidence):
        if confidence < 0.65:
            st.warning(f"Prediction is uncertain ({top_label}: {confidence:.1%}). Use the signal views as evidence, not as a diagnosis.")
        else:
            st.success(f"Most likely class: {top_label} ({confidence:.1%} model probability)")

    if result["probabilities"]:
        st.write("Class probabilities")
        st.bar_chart(pd.Series(result["probabilities"], name="Probability"))

    baseline_result = st.session_state.get("baseline_analysis")
    if baseline_result:
        st.subheader("A/B comparison: training baseline vs selected DSP")
        st.caption("Baseline is the configuration used to train the SVM. Selected is an inference-time DSP preview; the model is not retrained between the two runs.")
        comparison_columns = st.columns(2)
        baseline_label, baseline_confidence = _top_confidence(baseline_result)
        with comparison_columns[0].container(border=True):
            st.markdown("**Training baseline**")
            st.metric("Prediction", baseline_label)
            st.caption(f"{baseline_result['config']['dsp']['target_fs']} Hz · {baseline_result['config']['dsp']['quantization_bits']}-bit · {baseline_result['config']['dsp']['filter']}")
            st.caption(f"Top confidence: {baseline_confidence:.1%}" if np.isfinite(baseline_confidence) else "No probability output")
        with comparison_columns[1].container(border=True):
            st.markdown("**Selected DSP preview**")
            st.metric("Prediction", result["label"])
            selected_bits = result["config"]["dsp"].get("quantization_bits")
            st.caption(f"{result['target_fs']} Hz · {selected_bits if selected_bits is not None else 'none'}-bit · {result['config']['dsp']['filter']}")
            st.caption(f"Top confidence: {confidence:.1%}" if np.isfinite(confidence) else "No probability output")
        labels = sorted(set(baseline_result["probabilities"]) | set(result["probabilities"]))
        probability_comparison = pd.DataFrame(
            {
                "Baseline": [baseline_result["probabilities"].get(label, 0.0) for label in labels],
                "Selected": [result["probabilities"].get(label, 0.0) for label in labels],
            },
            index=labels,
        )
        st.bar_chart(probability_comparison)

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
    st.pyplot(_plot_filter_response(result), clear_figure=True)
    st.pyplot(_plot_spectrogram(result), clear_figure=True)

    export_payload = {
        "recording": filename,
        "prediction": result["label"],
        "probabilities": result["probabilities"],
        "duration_seconds": result["duration_seconds"],
        "n_segments": result["n_segments"],
        "feature_count": result["feature_count"],
        "config": result["config"],
    }
    st.download_button(
        "Download analysis JSON",
        data=json.dumps(export_payload, indent=2, ensure_ascii=False, default=str),
        file_name=f"{Path(filename).stem}_dsp_analysis.json",
        mime="application/json",
        icon=":material/download:",
    )
    with st.expander("DSP configuration used"):
        st.code(yaml.safe_dump(result["config"], sort_keys=False), language="yaml")
else:
    st.info("Upload a WAV file or use the bundled demo recording, choose the DSP settings, and click Analyze recording.")
