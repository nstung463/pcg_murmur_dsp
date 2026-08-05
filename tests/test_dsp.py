import numpy as np
import pytest

from pcg_dsp.dsp import apply_filter, design_filter, quantize, resample_signal, segment_signal, stft_matrix, wavelet_denoise


def test_quantization_is_bounded_and_deterministic():
    x = np.array([-1.2, -0.25, 0.0, 0.25, 1.2], dtype=np.float32)
    result = quantize(x, 8)
    assert np.all(result <= 1.0)
    assert np.all(result >= -1.0)
    np.testing.assert_array_equal(result, quantize(x, 8))


def test_resampling_changes_length_and_preserves_finite_values():
    x = np.sin(2 * np.pi * 100 * np.arange(4000) / 4000).astype(np.float32)
    y = resample_signal(x, 4000, 1000)
    assert 990 <= len(y) <= 1010
    assert np.isfinite(y).all()


@pytest.mark.parametrize("kind", ["butterworth", "fir", "none"])
def test_filter_modes_return_valid_signal(kind):
    fs = 1000
    x = np.random.default_rng(42).normal(size=3000).astype(np.float32)
    y = apply_filter(x, design_filter(fs, kind, 25, 400))
    assert y.shape == x.shape
    assert np.isfinite(y).all()


def test_segment_signal_has_fixed_shapes():
    segments = segment_signal(np.ones(1000, dtype=np.float32), fs=1000, seconds=2, hop_seconds=1)
    assert segments
    assert all(segment.shape == (2000,) for segment in segments)


def test_stft_and_wavelet_keep_valid_shapes():
    x = np.sin(2 * np.pi * 80 * np.arange(4000) / 1000).astype(np.float32)
    frequencies, times, matrix = stft_matrix(x, fs=1000, n_fft=256, hop_length=64)
    assert matrix.shape == (len(frequencies), len(times))
    assert np.isfinite(matrix).all()
    denoised = wavelet_denoise(x)
    assert denoised.shape == x.shape
    assert np.isfinite(denoised).all()
