import numpy as np
import io
from scipy.io import wavfile

from pcg_dsp.service import analyze_file, analyze_recording


class DummyModel:
    classes_ = np.array(["Absent", "Present"])

    def predict(self, features):
        assert features.shape[0] == 1
        return np.array(["Present"])

    def predict_proba(self, features):
        return np.array([[0.25, 0.75]])


def _config():
    return {
        "split": {"seed": 42},
        "dsp": {
            "target_fs": 1000,
            "quantization_bits": 16,
            "filter": "butterworth",
            "low_hz": 25,
            "high_hz": 400,
            "order": 4,
            "fir_taps": 129,
            "wavelet_denoise": False,
            "segment_seconds": 1.0,
            "segment_hop_seconds": 0.5,
        },
        "features": {"mode": "stats", "n_mfcc": 13, "n_fft": 256, "hop_length": 64},
        "noise": {"enabled": False},
    }


def test_analyze_recording_returns_prediction_and_plot_arrays():
    fs = 1000
    x = np.sin(2 * np.pi * 80 * np.arange(3000) / fs).astype(np.float32)
    buffer = io.BytesIO()
    wavfile.write(buffer, fs, (x * 32767).astype(np.int16))
    result = analyze_recording(buffer.getvalue(), DummyModel(), _config())
    assert result["label"] == "Present"
    assert result["feature_count"] == 5
    assert result["fft_frequency"].ndim == 1


def test_analyze_file_end_to_end(tmp_path):
    fs = 1000
    x = (0.5 * np.sin(2 * np.pi * 80 * np.arange(3000) / fs) * 32767).astype(np.int16)
    wav_path = tmp_path / "sample.wav"
    wavfile.write(wav_path, fs, x)
    import joblib

    model_path = tmp_path / "model.joblib"
    joblib.dump({"model": DummyModel(), "config": _config()}, model_path)
    result = analyze_file(wav_path, model_path)
    assert result["label"] == "Present"
    assert result["probabilities"]["Present"] == 0.75
    assert result["raw_signal"].shape == result["filtered_signal"].shape
    assert result["stft_magnitude"].ndim == 2
