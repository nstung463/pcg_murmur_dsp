"""Small, transparent classifiers for the DSP experiments."""

from __future__ import annotations

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier


def make_model(kind: str = "svm", seed: int = 42):
    kind = kind.lower()
    if kind == "svm":
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", SVC(C=2.0, kernel="rbf", probability=True, class_weight="balanced", random_state=seed)),
            ]
        )
    if kind in {"rf", "random_forest"}:
        return RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=seed, n_jobs=-1)
    if kind in {"mlp", "neural"}:
        return Pipeline(
            [
                ("scale", StandardScaler()),
                ("classifier", MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=300, early_stopping=True, random_state=seed)),
            ]
        )
    raise ValueError(f"Unknown model kind: {kind}")
