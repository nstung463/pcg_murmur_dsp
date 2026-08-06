# CNN ablation summary

All runs use the complete 874-patient labeled cohort, patient-wise split
611/131/132, pretrained MobileNetV3-Small, 1 kHz input, no filter, two evenly
spaced 3-second windows per recording, and patient-level aggregation.

| Experiment | Accuracy | Balanced accuracy | Macro-F1 | Threshold |
|---|---:|---:|---:|---:|
| Mean + tuned threshold | 0.8409 | 0.6661 | 0.6968 | 0.56 |
| Median + tuned threshold | 0.8333 | 0.6889 | 0.7104 | 0.62 |
| Top-25% + tuned threshold | 0.7652 | 0.6598 | 0.6532 | 0.76 |
| Max + tuned threshold | 0.6667 | 0.5429 | 0.5363 | 0.74 |
| Median + last block fine-tuned | 0.8409 | 0.6799 | 0.7083 | 0.62 |

## Fixed-split seed stability

With split seed fixed at 42 and model seeds 42, 7, and 123, median aggregation
gave:

- Accuracy: `0.8333 ± 0.0152`
- Balanced Accuracy: `0.6614 ± 0.0257`
- Macro-F1: `0.6877 ± 0.0254`

The variation is expected because the test set contains only 27 `Present`
patients. The split must remain fixed when comparing random seeds; changing the
split can change the result more than changing the model initialization.

## Decision

Median aggregation is the most useful CNN setting in this experiment. Max and
top-25% aggregation overreact to isolated high-probability windows. Fine-tuning
only the last block does not improve Macro-F1 over the frozen backbone enough to
justify the extra complexity. Keep the frozen CNN as an optional model and use
the DSP/SVM/MLP matrix as the main course project result.
