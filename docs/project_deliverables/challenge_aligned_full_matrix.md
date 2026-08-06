# Full challenge-aligned matrix

This is now the primary comparison track for the project. It uses the three
challenge labels (`Absent`, `Present`, `Unknown`) and reports the official-style
Weighted Accuracy as the primary score, followed by UAR, AUROC, AUPRC and
Macro-F1.

Protocol:

- public CirCor training release: 942 patients;
- patient-wise split, seed 42, 65/10/25: 612 train / 94 validation / 236 test;
- target sampling rate 1 kHz, quantization 16-bit;
- DSP matrix: 24 configurations, SVM/MLP × None/Butterworth/FIR × PSD/MFCC/STFT/Hybrid;
- MobileNet: pretrained ImageNet weights, frozen backbone, log-STFT input, mean patient aggregation, threshold 0.5.

## Best DSP configurations by Weighted Accuracy

| Model | Filter | Feature | Weighted Accuracy | UAR | AUROC | AUPRC | Macro-F1 |
|---|---|---|---:|---:|---:|---:|---:|
| SVM | None | Hybrid | **0.644** | 0.596 | 0.775 | 0.587 | 0.597 |
| SVM | None | MFCC | 0.629 | 0.571 | 0.776 | 0.579 | 0.578 |
| SVM | Butterworth | Hybrid | 0.622 | **0.597** | **0.791** | 0.585 | 0.596 |
| SVM | FIR | Hybrid | 0.611 | 0.544 | 0.763 | 0.558 | 0.539 |
| SVM | Butterworth | MFCC | 0.596 | 0.533 | 0.759 | 0.541 | 0.528 |
| MLP | FIR | Hybrid | 0.564 | 0.461 | 0.793 | 0.590 | 0.479 |

The complete 24-row result is in
`artifacts/challenge_matrix_3class/full_matrix_3class.csv` and is generated
by `scripts/run_challenge_matrix.py`.

## MobileNet pretrained configurations

| Model | Filter | Feature | Weighted Accuracy | UAR | AUROC | AUPRC | Macro-F1 |
|---|---|---|---:|---:|---:|---:|---:|
| MobileNetV3-Small | Butterworth | log-STFT | **0.667** | **0.641** | **0.796** | **0.629** | **0.604** |
| MobileNetV3-Small | None | log-STFT | 0.638 | 0.620 | 0.772 | 0.596 | 0.580 |
| MobileNetV3-Small | FIR | log-STFT | 0.553 | 0.579 | 0.769 | 0.581 | 0.524 |

The CNN uses the same patient split and three-class labels, so these numbers
can be compared with the DSP rows above. They are still a public-split result,
not an official hidden-test submission.

## Interpretation

The best challenge-aligned configuration is currently pretrained MobileNet with
Butterworth preprocessing (`Weighted Accuracy = 0.667`). The best DSP-only
configuration is SVM + no filter + Hybrid (`Weighted Accuracy = 0.644`). MLP
has reasonable AUROC in some settings but poor Weighted Accuracy because it
over-predicts `Absent` and misses `Present`/`Unknown`.
