# Standardized full-data benchmark

This benchmark keeps the binary DSP task and the pretrained CNN task separate
but uses the same cohort and patient split:

- 874 labeled patients (`Absent`/`Present`) from the public training release;
- patient-wise split, seed 42: 611 train / 131 validation / 132 test;
- target sampling rate 1 kHz, 16-bit quantization;
- metrics: accuracy, balanced accuracy and macro-F1.

## DSP matrix

The DSP matrix is the complete 2 × 3 × 4 experiment:

`model ∈ {SVM, MLP}` × `filter ∈ {None, Butterworth, FIR}` ×
`feature ∈ {PSD, MFCC, STFT, Hybrid}`.

| Model | Filter | Feature | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---:|---:|---:|
| MLP | None | MFCC | **0.848** | 0.657 | **0.693** |
| MLP | None | Hybrid | 0.841 | 0.625 | 0.654 |
| SVM | None | Hybrid | 0.750 | **0.664** | 0.648 |
| SVM | FIR | Hybrid | 0.765 | 0.646 | 0.644 |
| SVM | None | MFCC | 0.742 | 0.659 | 0.642 |
| SVM | FIR | MFCC | 0.758 | 0.628 | 0.628 |
| SVM | Butterworth | Hybrid | 0.742 | 0.618 | 0.615 |

The complete 24-row artifact is `artifacts/full_matrix_24/full_matrix.csv`.

## Pretrained CNN matrix

MobileNetV3-Small is evaluated separately because its representation is fixed:
each window becomes a log-STFT image; there is no PSD/MFCC/Hybrid feature
choice. The backbone uses pretrained ImageNet weights and remains frozen. The
CNN uses mean patient aggregation, threshold 0.5, two evenly spaced 3-second
windows per recording, and the same 611/131/132 split.

| Model | Filter | Feature | Accuracy | Balanced accuracy | Macro-F1 |
|---|---|---|---:|---:|---:|
| MobileNetV3-Small (pretrained, frozen) | None | log-STFT | 0.841 | 0.749 | 0.752 |
| MobileNetV3-Small (pretrained, frozen) | Butterworth | log-STFT | 0.826 | 0.698 | 0.712 |
| MobileNetV3-Small (pretrained, frozen) | FIR | log-STFT | **0.864** | **0.777** | **0.784** |

These CNN results are still binary results and must not be compared directly
with the three-class challenge-aligned weighted accuracy benchmark.
