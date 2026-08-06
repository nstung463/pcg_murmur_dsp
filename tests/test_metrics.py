import numpy as np

from pcg_dsp.metrics import challenge_metrics, uar, weighted_accuracy


def test_challenge_weighted_accuracy_uses_5_3_1_weights():
    truth = ["Present", "Unknown", "Absent", "Present"]
    pred = ["Present", "Absent", "Absent", "Absent"]
    # Correct weights: 5 + 1; total: 5 + 3 + 1 + 5.
    assert weighted_accuracy(truth, pred) == 6 / 14


def test_uar_is_macro_recall_for_three_classes():
    truth = ["Present", "Unknown", "Absent"]
    pred = ["Present", "Absent", "Absent"]
    assert uar(truth, pred) == (1.0 + 0.0 + 1.0) / 3.0


def test_challenge_metrics_include_probability_metrics():
    truth = ["Absent", "Present", "Unknown", "Absent", "Present", "Unknown"]
    pred = truth.copy()
    probabilities = np.eye(3, dtype=float)[[0, 1, 2, 0, 1, 2]]
    result = challenge_metrics(truth, pred, probabilities, ["Absent", "Present", "Unknown"])
    assert result["weighted_accuracy"] == 1.0
    assert result["uar"] == 1.0
    assert result["auroc_macro_ovr"] == 1.0
    assert result["auprc_macro_ovr"] == 1.0
