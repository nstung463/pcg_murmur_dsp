# Challenge-aligned evaluation note

The primary DSP matrix remains a binary `Absent` versus `Present` experiment.
It reports accuracy, balanced accuracy, macro precision/recall/F1 and a
patient-level confusion matrix.

For a separate public-data comparison track, the repository now includes
`configs/challenge_aligned.yaml` and the optional final cell in
`notebooks/PCG_DSP_full_end_to_end_colab.ipynb`. This track keeps all three
murmur labels (`Absent`, `Present`, `Unknown`), uses a 65/10/25 patient-wise
split, and reports:

- PhysioNet-style murmur weighted accuracy (weights 1/5/3 for Absent/Present/Unknown);
- UAR, implemented as macro recall across the three classes;
- macro one-vs-rest AUROC;
- macro AUPRC;
- the existing accuracy, balanced accuracy and macro-F1 metrics.

These metrics are emitted only when all three classes are present in the test
partition. A binary run is never labelled as challenge-comparable. The
official competition used hidden validation/test data, while recent papers
such as M2D used a separate public split and different windowing/aggregation.
Therefore this track supports a transparent research comparison, but it does
not reproduce the official hidden-test leaderboard.

The full public training release (942 patients; 612/94/236 split) with SVM +
Butterworth + hybrid features produced weighted accuracy `0.622`, UAR `0.597`,
macro AUROC `0.791`, macro AUPRC `0.585` and macro-F1 `0.596`. The complete
machine-readable result is stored in
`challenge_aligned_full_metrics.json`. These are public-split results, not
official hidden-test leaderboard scores.

| Classifier | Weighted Accuracy | UAR | Macro AUROC | Macro AUPRC | Macro-F1 |
|---|---:|---:|---:|---:|---:|
| SVM | **0.622** | **0.597** | **0.791** | **0.585** | **0.596** |
| MLP | 0.464 | 0.413 | 0.744 | 0.541 | 0.429 |

Using the identical split and DSP front-end, the MLP produced weighted accuracy
`0.464`, UAR `0.413`, macro AUROC `0.744`, macro AUPRC `0.541` and macro-F1
`0.429`. The MLP predicts the majority `Absent` class more often and is
therefore worse on the challenge-weighted objective in this configuration. Its
complete result is stored in `challenge_aligned_mlp_full_metrics.json`.
