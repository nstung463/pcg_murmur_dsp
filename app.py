"""Streamlit front end for the PCG DSP inference backend."""

from __future__ import annotations

from pathlib import Path
import sys
import io
import json
import html
import importlib

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

# Streamlit can rerun this script while retaining an older imported module in
# ``sys.modules``.  Reload the CNN module only when the patient-level helper is
# missing so a code update is picked up without requiring a full reinstall.
import pcg_dsp.cnn as _cnn
if not hasattr(_cnn, "analyze_mobilenet_patient"):
    _cnn = importlib.reload(_cnn)
analyze_mobilenet_patient = _cnn.analyze_mobilenet_patient
analyze_mobilenet_recording = _cnn.analyze_mobilenet_recording
load_mobilenet_bundle = _cnn.load_mobilenet_bundle
from pcg_dsp.dsp import design_filter, resample_signal
from pcg_dsp.io import parse_patient_file
from pcg_dsp.service import analyze_patient_recordings, analyze_recording, load_model_bundle


DEFAULT_MODEL = "artifacts/runs/svm_none_hybrid/model.joblib"
DEMO_DATA_ROOT = PROJECT_ROOT / "data" / "circor-heart-sound" / "1.0.3" / "training_data"

# Curated model bundles for the demo.  The paths are relative to the project
# root so the same dropdown works after cloning the repository on another
# machine.  We intentionally omit fixture/noise-only runs from the polished
# demo; those remain available for notebook experiments.
MODEL_PRESETS = {
    "Best benchmark · MobileNetV3-Small + FIR": "artifacts/cnn_matrix_pretrained/fir/model.pt",
    "Recommended · SVM + No filter + Hybrid": "artifacts/runs/svm_none_hybrid/model.joblib",
    "SVM · Butterworth + Hybrid (training baseline)": "artifacts/runs/svm_butterworth_hybrid/model.joblib",
    "SVM · FIR + Hybrid": "artifacts/runs/svm_fir_hybrid/model.joblib",
    "SVM · Butterworth + MFCC": "artifacts/runs/svm_butterworth_mfcc/model.joblib",
    "SVM · Butterworth + PSD": "artifacts/runs/svm_butterworth_psd/model.joblib",
    "SVM · Butterworth + STFT": "artifacts/runs/svm_butterworth_stft/model.joblib",
    "MLP · No filter + MFCC": "artifacts/runs/mlp_none_mfcc/model.joblib",
    "MLP · Butterworth + Hybrid": "artifacts/runs/mlp_butterworth_hybrid/model.joblib",
}

_ANALYSIS_STATE_KEYS = (
    "analysis",
    "baseline_analysis",
    "audio_bytes",
    "analysis_overrides",
    "analysis_model_preset",
)


def _clear_analysis_state() -> None:
    """Prevent a previous result from being shown for a new input/model."""
    for key in _ANALYSIS_STATE_KEYS:
        st.session_state.pop(key, None)


@st.cache_resource(show_spinner=False)
def _cached_model_bundle(model_path: str):
    """Cache the estimator so widget reruns do not reload joblib from disk."""
    return load_model_bundle(model_path)


@st.cache_resource(show_spinner=False)
def _cached_mobilenet_bundle(model_path: str):
    """Cache the MobileNet checkpoint and model construction across reruns."""
    return load_mobilenet_bundle(model_path)


def _find_demo_wav() -> Path | None:
    """Return one deterministic bundled recording for a no-file demo."""
    if not DEMO_DATA_ROOT.exists():
        return None
    candidates = sorted(DEMO_DATA_ROOT.glob("*.wav"))
    return candidates[0] if candidates else None


def _list_project_wavs() -> list[Path]:
    """List recordings shipped/downloaded in the project's data directory."""
    if not DEMO_DATA_ROOT.exists():
        return []
    return sorted(DEMO_DATA_ROOT.glob("*.wav"))


def _ground_truth_for_wav(wav_path: str | Path) -> str | None:
    """Read the patient-level reference label for a bundled CirCor WAV."""
    path = Path(wav_path)
    patient_id = path.name.split("_", 1)[0]
    patient_file = path.parent / f"{patient_id}.txt"
    if not patient_file.exists():
        return None
    try:
        return parse_patient_file(patient_file).label
    except (OSError, UnicodeError, ValueError):
        return None


def _patient_recordings_for_wav(wav_path: str | Path) -> tuple[str | None, list[Path]]:
    """Resolve all labelled recordings belonging to the selected patient."""
    path = Path(wav_path)
    patient_id = path.name.split("_", 1)[0]
    patient_file = path.parent / f"{patient_id}.txt"
    if not patient_file.exists():
        return None, []
    try:
        patient = parse_patient_file(patient_file)
    except (OSError, UnicodeError, ValueError):
        return None, []
    return patient.patient_id, [recording.wav_path for recording in patient.recordings]


def _render_evaluation_card(prediction: str, ground_truth: str | None, confidence: float) -> None:
    """Render an immediately readable prediction-vs-truth verdict."""
    if ground_truth in {"Absent", "Present"}:
        is_correct = prediction == ground_truth
        verdict = "CORRECT" if is_correct else "INCORRECT"
        icon = "✓" if is_correct else "✕"
        color = "#166534" if is_correct else "#b91c1c"
        background = "#f0fdf4" if is_correct else "#fef2f2"
        border = "#86efac" if is_correct else "#fca5a5"
        detail = "Prediction matches the labelled dataset." if is_correct else "Model prediction does not match the labelled dataset."
        st.markdown(
            f"""
            <div style="border:2px solid {border}; background:{background}; border-radius:12px; padding:18px 20px; margin:12px 0 18px 0;">
              <div style="font-size:0.78rem; letter-spacing:0.08em; color:{color}; font-weight:700;">PREDICTION CHECK</div>
              <div style="font-size:1.65rem; line-height:1.25; color:{color}; font-weight:800; margin:5px 0 8px 0;">{icon}&nbsp; {verdict}</div>
              <div style="font-size:1rem; color:#1f2937;">Model prediction: <b>{html.escape(prediction)}</b>&nbsp;&nbsp;|&nbsp;&nbsp; Dataset ground truth: <b>{html.escape(ground_truth)}</b></div>
              <div style="font-size:0.9rem; color:#4b5563; margin-top:6px;">{detail} Top confidence: <b>{confidence:.1%}</b>.</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    elif ground_truth == "Unknown":
        st.info("GROUND TRUTH UNKNOWN: recording is not counted as correct or incorrect.")
    else:
        st.info("GROUND TRUTH UNAVAILABLE: external uploads can be predicted, but cannot be scored without a reference label.")


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
    values = np.nan_to_num(np.asarray(signal, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    if normalize:
        peak = float(np.max(np.abs(values))) if values.size else 0.0
        if peak > 1e-9:
            values = values / peak
    values = np.clip(values, -1.0, 1.0)
    buffer = io.BytesIO()
    wavfile.write(buffer, int(sample_rate), (values * 32767.0).astype(np.int16))
    return buffer.getvalue()


def _browser_playback_signal(values: np.ndarray, sample_rate: int, playback_rate: int = 16000) -> tuple[np.ndarray, int]:
    """Convert preview audio to a browser-friendly sample rate.

    The DSP pipeline may intentionally run at 1 kHz, but some browser WAV
    decoders do not reliably play such an unusual rate. This conversion is
    only for listening; plots, features and predictions use the processing-rate arrays.
    """
    if int(sample_rate) == playback_rate:
        return np.asarray(values, dtype=np.float32), playback_rate
    return resample_signal(np.asarray(values, dtype=np.float32), int(sample_rate), playback_rate), playback_rate


st.set_page_config(page_title="PCG DSP Demo", page_icon=":material/graphic_eq:", layout="wide")
st.title("PCG Murmur Detection — DSP Demo")
st.caption("Educational research prototype. This is not a clinical diagnostic device.")

demo_wav = _find_demo_wav()
project_wavs = _list_project_wavs()
with st.sidebar:
    st.header("Model and DSP")
    available_presets = {
        label: path
        for label, path in MODEL_PRESETS.items()
        if (PROJECT_ROOT / path).exists()
    }
    if available_presets:
        preset_labels = list(available_presets)
        default_label = next((label for label in MODEL_PRESETS if label.startswith("Recommended")), next(iter(MODEL_PRESETS)))
        default_index = preset_labels.index(default_label) if default_label in preset_labels else 0
        selected_preset = st.selectbox(
            "Model preset",
            preset_labels,
            index=default_index,
            help="Choose one of the trained bundles; its DSP configuration is loaded automatically.",
        )
        model_relpath = available_presets[selected_preset]
        model_path = str(PROJECT_ROOT / model_relpath)
        st.caption(f"Bundle: `{model_relpath}`")
        if selected_preset.startswith("Recommended"):
            st.caption("Selected as the strongest classical DSP preset by balanced accuracy; no model is correct on every recording.")
    else:
        model_path = DEFAULT_MODEL
        st.error("No trained model bundles were found in `artifacts/runs`. Run the training notebook first.")
    input_modes = ["Project data", "Upload external WAV"] if project_wavs else ["Upload external WAV"]
    input_mode = st.radio(
        "Recording source",
        input_modes,
        index=0,
        horizontal=True,
        help="Choose exactly one source for the recording to analyze.",
    )
    previous_input_mode = st.session_state.get("_input_mode")
    if previous_input_mode is not None and previous_input_mode != input_mode:
        # Do not accidentally analyze a file from the previous source after
        # the user switches between project data and external upload.
        _clear_analysis_state()
        for key in ("input_bytes", "filename", "input_source", "input_path"):
            st.session_state.pop(key, None)
    st.session_state["_input_mode"] = input_mode
    if input_mode == "Upload external WAV":
        uploaded = st.file_uploader("Choose a WAV file", type=["wav"])
        if uploaded is not None:
            if (
                st.session_state.get("input_source") != "uploaded WAV"
                or st.session_state.get("filename") != uploaded.name
            ):
                _clear_analysis_state()
            st.session_state["input_bytes"] = uploaded.getvalue()
            st.session_state["filename"] = uploaded.name
            st.session_state["input_source"] = "uploaded WAV"
            st.session_state.pop("input_path", None)
    else:
        st.caption(f"Project data folder: `{DEMO_DATA_ROOT}`")
        selected_local_wav = st.selectbox(
            "Choose a recording",
            project_wavs,
            format_func=lambda path: path.name,
            help="Select a recording already stored in data/circor-heart-sound/.../training_data.",
        )
        if st.button("Use selected recording", icon=":material/folder_open:"):
            if st.session_state.get("input_path") != str(selected_local_wav):
                _clear_analysis_state()
            st.session_state["input_bytes"] = selected_local_wav.read_bytes()
            st.session_state["filename"] = selected_local_wav.name
            st.session_state["input_source"] = "project data WAV"
            st.session_state["input_path"] = str(selected_local_wav)
        if demo_wav is not None and st.button("Use bundled demo recording", icon=":material/play_circle:"):
            if st.session_state.get("input_path") != str(demo_wav):
                _clear_analysis_state()
            st.session_state["input_bytes"] = demo_wav.read_bytes()
            st.session_state["filename"] = demo_wav.name
            st.session_state["input_source"] = "bundled demo WAV"
            st.session_state["input_path"] = str(demo_wav)
    if st.session_state.get("input_source") in {"bundled demo WAV", "project data WAV"}:
        st.caption(f"Selected file: `{st.session_state.get('filename', 'not selected')}`")

    patient_id = None
    patient_recording_paths: list[Path] = []
    evaluation_scope = "recording"
    selected_input_path = st.session_state.get("input_path")
    if selected_input_path:
        patient_id, patient_recording_paths = _patient_recordings_for_wav(selected_input_path)
    if patient_recording_paths:
        evaluation_scope = st.radio(
            "Evaluation level",
            ["Patient-level (benchmark)", "Recording-level preview"],
            index=0,
            horizontal=True,
            help="Patient-level aggregates all AV/MV/PV/TV recordings to match the benchmark protocol.",
        )
        st.caption(f"Patient `{patient_id}` · {len(patient_recording_paths)} recordings will be aggregated.")

    model_exists = Path(model_path).exists()
    previous_model_path = st.session_state.get("_model_path")
    if previous_model_path is not None and previous_model_path != model_path:
        _clear_analysis_state()
    st.session_state["_model_path"] = model_path
    if not model_exists:
        st.warning("Model file not found. Train a model or update the path.")

    config = None
    model = None
    cnn_bundle = None
    model_kind = "mobilenet" if Path(model_path).suffix.lower() == ".pt" else "classical"
    if model_exists:
        try:
            if model_kind == "mobilenet":
                cnn_bundle, config = _cached_mobilenet_bundle(model_path)
            else:
                model, config = _cached_model_bundle(model_path)
        except Exception as exc:
            st.error(f"Cannot load model: {exc}")

    if config:
        dsp = config["dsp"]
        if model_kind == "mobilenet":
            st.caption("MobileNet benchmark contract: 1 kHz · 16-bit · FIR · log-STFT 128×128. Other DSP settings are inference-time ablations, not retrained CNN variants.")
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
inference_ready = (model_kind == "mobilenet" and cnn_bundle is not None) or (model_kind == "classical" and model is not None)
if analyze_clicked and active_bytes is not None and inference_ready and config is not None:
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
            use_patient_level = evaluation_scope.startswith("Patient-level") and bool(patient_recording_paths)
            if model_kind == "mobilenet" and use_patient_level:
                selected_result = analyze_mobilenet_patient(patient_recording_paths, cnn_bundle, config, overrides, patient_id)
            elif model_kind == "mobilenet":
                selected_result = analyze_mobilenet_recording(active_bytes, cnn_bundle, config, overrides)
            elif use_patient_level:
                selected_result = analyze_patient_recordings(patient_recording_paths, model, config, overrides, patient_id)
            else:
                selected_result = analyze_recording(active_bytes, model, config, overrides)
            baseline_result = None
            if compare_baseline:
                st.write("Running the training-baseline comparison")
                if model_kind == "mobilenet" and use_patient_level:
                    baseline_result = analyze_mobilenet_patient(patient_recording_paths, cnn_bundle, config, baseline_overrides, patient_id)
                elif model_kind == "mobilenet":
                    baseline_result = analyze_mobilenet_recording(active_bytes, cnn_bundle, config, baseline_overrides)
                elif use_patient_level:
                    baseline_result = analyze_patient_recordings(patient_recording_paths, model, config, baseline_overrides, patient_id)
                else:
                    baseline_result = analyze_recording(active_bytes, model, config, baseline_overrides)
            st.session_state["analysis"] = selected_result
            st.session_state["baseline_analysis"] = baseline_result
            st.session_state["analysis_overrides"] = overrides
            st.session_state["analysis_model_preset"] = selected_preset if "selected_preset" in locals() else model_path
            st.session_state["audio_bytes"] = active_bytes
            status.update(label="Analysis complete", state="complete")
        except Exception as exc:
            status.update(label="Analysis failed", state="error")
            st.error(f"Analysis failed: {exc}")

result = st.session_state.get("analysis")
if result:
    filename = st.session_state.get("filename", "recording.wav")
    if result.get("aggregation_level") == "patient":
        st.info(
            f"Patient-level prediction: aggregated {result.get('recording_count', 0)} recordings "
            f"for patient {result.get('patient_id', 'selected patient')}. The plots show the selected representative recording."
        )
    else:
        st.info(
            "Recording-level preview: this prediction uses only the selected WAV and is not directly comparable "
            "to the patient-level benchmark."
        )
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

    ground_truth = None
    input_path = st.session_state.get("input_path")
    if input_path:
        ground_truth = _ground_truth_for_wav(input_path)
    _render_evaluation_card(result["label"], ground_truth, confidence)

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
        st.caption("Baseline is the configuration used to train the selected model. Selected is an inference-time DSP preview; the model is not retrained between the two runs.")
        comparison_columns = st.columns(2)
        baseline_label, baseline_confidence = _top_confidence(baseline_result)
        with comparison_columns[0].container(border=True):
            st.markdown("**Training baseline**")
            st.metric("Prediction", baseline_label)
            st.caption(f"{baseline_result['config']['dsp']['target_fs']} Hz · {baseline_result['config']['dsp']['quantization_bits']}-bit · {baseline_result['config']['dsp']['filter']}")
            st.caption(f"Top confidence: {baseline_confidence:.1%}" if np.isfinite(baseline_confidence) else "No probability output")
            if ground_truth in {"Absent", "Present"}:
                st.caption(f"Against ground truth: {'Correct' if baseline_label == ground_truth else 'Incorrect'}")
        with comparison_columns[1].container(border=True):
            st.markdown("**Selected DSP preview**")
            st.metric("Prediction", result["label"])
            selected_bits = result["config"]["dsp"].get("quantization_bits")
            st.caption(f"{result['target_fs']} Hz · {selected_bits if selected_bits is not None else 'none'}-bit · {result['config']['dsp']['filter']}")
            st.caption(f"Top confidence: {confidence:.1%}" if np.isfinite(confidence) else "No probability output")
            if ground_truth in {"Absent", "Present"}:
                st.caption(f"Against ground truth: {'Correct' if result['label'] == ground_truth else 'Incorrect'}")
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
    st.caption("Each option is the same recording after one DSP layer. Playback is normalized separately for listening; waveforms and metrics use the actual processed values.")
    audio_stages = {
        "Raw WAV": (result["raw_signal"], result["source_fs"], "Original PCM at source sample rate"),
        "After resample": (result["resampled_signal"], result["target_fs"], f"Resampled to {result['target_fs']} Hz"),
        "After quantization": (result["quantized_signal"], result["target_fs"], "Amplitude quantized at configured bit depth"),
        "After band-pass filter": (result["filtered_signal"], result["target_fs"], "Final waveform used for segmentation and features"),
    }
    selected_stage = st.selectbox("Audio preview", list(audio_stages), key="audio_stage")
    normalize_playback = st.checkbox("Normalize playback volume", value=True, key="normalize_playback")
    preview_signal, preview_fs, preview_description = audio_stages[selected_stage]
    browser_signal, browser_fs = _browser_playback_signal(preview_signal, preview_fs)
    st.audio(_wav_bytes(browser_signal, browser_fs, normalize_playback), format="audio/wav")
    st.caption(f"{preview_description} · {len(preview_signal) / preview_fs:.2f} s · processing: {preview_fs} Hz · playback: {browser_fs} Hz")

    st.pyplot(_plot_waveforms(result), clear_figure=True)
    st.pyplot(_plot_frequency_views(result), clear_figure=True)
    st.pyplot(_plot_filter_response(result), clear_figure=True)
    st.pyplot(_plot_spectrogram(result), clear_figure=True)

    export_payload = {
        "recording": filename,
        "model": result.get("model_name", st.session_state.get("analysis_model_preset", "classical")),
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
