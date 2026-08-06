"""Metrics for challenge-aligned patient-level murmur evaluation.

The DSP matrix in :mod:`pcg_dsp.pipeline` intentionally keeps its original
binary Absent/Present protocol.  This module provides the three-class metrics
used by the PhysioNet Challenge and recent CirCor papers so that a run can be
compared only when it also uses the same labels and split protocol.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, recall_score, roc_auc_score
from sklearn.preprocessing import label_binarize


CHALLENGE_LABELS = ("Absent", "Present", "Unknown")
CHALLENGE_WEIGHTS = {"Absent": 1.0, "Present": 5.0, "Unknown": 3.0}


def weighted_accuracy(y_true: Sequence[str], y_pred: Sequence[str]) -> float:
    """Return the three-class PhysioNet murmur weighted accuracy.

    Correct Present/Unknown/Absent patients receive weights 5/3/1.  The
    denominator uses the same weights for the ground-truth classes.
    """

    truth = np.asarray(y_true)
    pred = np.asarray(y_pred)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    denominator = sum(CHALLENGE_WEIGHTS.get(str(label), 0.0) for label in truth)
    if denominator <= 0:
        raise ValueError("y_true contains no challenge labels")
    numerator = sum(
        CHALLENGE_WEIGHTS.get(str(label), 0.0)
        for label, guess in zip(truth, pred)
        if label == guess
    )
    return float(numerator / denominator)


def uar(y_true: Sequence[str], y_pred: Sequence[str], labels: Iterable[str] = CHALLENGE_LABELS) -> float:
    """Return unweighted average recall (macro recall)."""

    return float(recall_score(y_true, y_pred, labels=list(labels), average="macro", zero_division=0))


def probability_metrics(
    y_true: Sequence[str],
    probabilities: np.ndarray,
    classes: Sequence[str],
) -> dict[str, float]:
    """Compute macro one-vs-rest AUROC and AUPRC for multiclass scores."""

    truth = np.asarray(y_true)
    scores = np.asarray(probabilities, dtype=float)
    class_names = list(classes)
    if scores.ndim != 2 or scores.shape[1] != len(class_names):
        raise ValueError("probabilities must have one column per class")
    if len(set(truth.tolist())) < 2:
        return {"auroc_macro_ovr": float("nan"), "auprc_macro_ovr": float("nan")}
    binary_truth = label_binarize(truth, classes=class_names)
    # label_binarize returns one column for a two-class target; the challenge
    # protocol is three-class, but this fallback keeps the helper general.
    if binary_truth.shape[1] != scores.shape[1]:
        binary_truth = np.column_stack([1 - binary_truth[:, 0], binary_truth[:, 0]])
    return {
        "auroc_macro_ovr": float(
            roc_auc_score(truth, scores, labels=class_names, multi_class="ovr", average="macro")
        ),
        "auprc_macro_ovr": float(average_precision_score(binary_truth, scores, average="macro")),
    }


def challenge_metrics(
    y_true: Sequence[str],
    y_pred: Sequence[str],
    probabilities: np.ndarray | None = None,
    classes: Sequence[str] = CHALLENGE_LABELS,
) -> dict[str, float]:
    """Return all challenge-aligned metrics for a patient-level prediction."""

    result = {
        "weighted_accuracy": weighted_accuracy(y_true, y_pred),
        "uar": uar(y_true, y_pred, labels=classes),
    }
    if probabilities is not None:
        result.update(probability_metrics(y_true, probabilities, classes))
    return result
