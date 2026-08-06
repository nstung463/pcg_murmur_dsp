"""Patient-wise MobileNetV3-Small baseline for the CirCor PCG data.

This is an optional deep-learning baseline inspired by the Kaggle notebook:
each 3-second DSP window is converted to a log-STFT image, classified with a
MobileNetV3-Small binary head, and window predictions are averaged back to the
patient level before reporting metrics.  No windows from a patient are shared
between train/validation/test.
"""

from __future__ import annotations

import argparse
import json
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import signal
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from pcg_dsp.dsp import apply_filter, design_filter, quantize, resample_signal, segment_signal
from pcg_dsp.io import iter_patients, load_wav
from pcg_dsp.metrics import CHALLENGE_LABELS, challenge_metrics
from pcg_dsp.pipeline import patient_split


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "circor-heart-sound" / "1.0.3" / "training_data"
DEFAULT_OUT = ROOT / "artifacts" / "cnn_mobilenet"


@dataclass
class Example:
    patient_id: str
    label: int
    segment: np.ndarray


LABEL_TO_INDEX = {label: index for index, label in enumerate(CHALLENGE_LABELS)}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def _patient_frame(data_dir: Path, three_class: bool = False) -> pd.DataFrame:
    rows = []
    for patient in iter_patients(data_dir):
        allowed = set(CHALLENGE_LABELS) if three_class else {"Absent", "Present"}
        if patient.label not in allowed:
            continue
        # A patient with no existing WAV reference cannot contribute examples.
        if patient.recordings:
            rows.append({"patient_id": patient.patient_id, "label": patient.label})
    return pd.DataFrame(rows)


def _build_examples(
    data_dir: Path,
    patient_ids: set[str],
    target_fs: int,
    filter_name: str,
    quantization_bits: int | None,
    segment_seconds: float,
    hop_seconds: float,
    max_windows_per_recording: int | None,
    three_class: bool = False,
) -> list[Example]:
    spec = design_filter(target_fs, filter_name, 25.0, 400.0, 4, 129)
    examples: list[Example] = []
    for patient in iter_patients(data_dir):
        allowed = set(CHALLENGE_LABELS) if three_class else {"Absent", "Present"}
        if patient.patient_id not in patient_ids or patient.label not in allowed:
            continue
        label = LABEL_TO_INDEX[patient.label] if three_class else int(patient.label == "Present")
        for recording in patient.recordings:
            try:
                source_fs, raw = load_wav(recording.wav_path)
                x = resample_signal(raw, source_fs, target_fs)
                if quantization_bits is not None:
                    x = quantize(x, quantization_bits)
                x = apply_filter(x, spec)
                segments = segment_signal(x, target_fs, segment_seconds, hop_seconds)
                if max_windows_per_recording is not None and len(segments) > max_windows_per_recording:
                    # Keep evenly spaced windows so short and long recordings
                    # contribute comparable temporal coverage.
                    indices = np.linspace(0, len(segments) - 1, max_windows_per_recording, dtype=int)
                    segments = [segments[index] for index in indices]
                for segment in segments:
                    examples.append(Example(patient.patient_id, label, np.asarray(segment, dtype=np.float32)))
            except Exception as exc:
                print(f"Skipping {recording.wav_path.name}: {exc}")
    return examples


def _spectrogram_image(x: np.ndarray, fs: int, image_size: int = 128) -> torch.Tensor:
    # Log-STFT in the heart-sound band, then resize to the CNN input size.
    frequencies, _, matrix = signal.stft(
        x,
        fs=fs,
        nperseg=min(256, len(x)),
        noverlap=min(192, max(0, len(x) - 1)),
        nfft=512,
        boundary=None,
    )
    mask = (frequencies >= 20.0) & (frequencies <= min(400.0, fs / 2.0))
    power = np.log1p(np.abs(matrix[mask]).astype(np.float32))
    if power.size == 0:
        power = np.zeros((32, 32), dtype=np.float32)
    lo, hi = np.percentile(power, [1.0, 99.0])
    image = np.clip((power - lo) / (hi - lo + 1e-6), 0.0, 1.0)
    tensor = torch.from_numpy(image).unsqueeze(0).unsqueeze(0)
    tensor = F.interpolate(tensor, size=(image_size, image_size), mode="bilinear", align_corners=False).squeeze(0)
    # MobileNet expects three channels. Replication retains the DSP image and
    # lets us use the standard torchvision architecture unchanged.
    tensor = tensor.repeat(3, 1, 1)
    mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
    return ((tensor - mean) / std).float()


class PCGWindowDataset(Dataset):
    def __init__(self, examples: list[Example], fs: int, image_size: int = 128):
        self.examples = examples
        self.fs = fs
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int):
        example = self.examples[index]
        return _spectrogram_image(example.segment, self.fs, self.image_size), example.label, example.patient_id


def _patient_predictions(patient_ids: list[str], labels: list[int], probabilities: list[float], aggregation: str = "mean"):
    grouped: dict[str, list[float]] = {}
    truth: dict[str, int] = {}
    for patient_id, label, probability in zip(patient_ids, labels, probabilities):
        grouped.setdefault(patient_id, []).append(float(probability))
        truth[patient_id] = int(label)
    ids = sorted(grouped)
    y_true = np.asarray([truth[patient_id] for patient_id in ids])
    values = []
    for patient_id in ids:
        current = np.asarray(grouped[patient_id], dtype=np.float32)
        if aggregation == "mean":
            values.append(float(np.mean(current)))
        elif aggregation == "median":
            values.append(float(np.median(current)))
        elif aggregation == "max":
            values.append(float(np.max(current)))
        elif aggregation == "top25":
            count = max(1, int(np.ceil(current.size * 0.25)))
            values.append(float(np.mean(np.sort(current)[-count:])))
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")
    return np.asarray([truth[patient_id] for patient_id in ids]), np.asarray(values), ids


def _metrics_from_predictions(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "n_patients": int(len(y_true)),
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist(),
    }


def _patient_multiclass_predictions(patient_ids: list[str], labels: list[int], probabilities: list[np.ndarray], aggregation: str = "mean"):
    grouped: dict[str, list[np.ndarray]] = {}
    truth: dict[str, int] = {}
    for patient_id, label, probability in zip(patient_ids, labels, probabilities):
        grouped.setdefault(patient_id, []).append(np.asarray(probability, dtype=np.float32))
        truth[patient_id] = int(label)
    ids = sorted(grouped)
    y_true = np.asarray([truth[patient_id] for patient_id in ids], dtype=np.int64)
    values = []
    for patient_id in ids:
        current = np.stack(grouped[patient_id], axis=0)
        if aggregation == "mean":
            values.append(np.mean(current, axis=0))
        elif aggregation == "median":
            values.append(np.median(current, axis=0))
        else:
            raise ValueError("Three-class CNN supports mean or median aggregation")
    return y_true, np.asarray(values), ids


def _metrics_from_multiclass_predictions(y_true: np.ndarray, probabilities: np.ndarray) -> dict:
    y_pred = np.argmax(probabilities, axis=1)
    class_names = list(CHALLENGE_LABELS)
    truth_names = [class_names[index] for index in y_true]
    pred_names = [class_names[index] for index in y_pred]
    result = {
        "n_patients": int(len(y_true)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=[0, 1, 2], average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=[0, 1, 2]).tolist(),
    }
    result.update(challenge_metrics(truth_names, pred_names, probabilities, class_names))
    return result


def _patient_metrics(patient_ids: list[str], labels: list[int], probabilities: list[float], aggregation: str = "mean", threshold: float = 0.5) -> dict:
    y_true, y_prob, _ = _patient_predictions(patient_ids, labels, probabilities, aggregation)
    return _metrics_from_predictions(y_true, y_prob, threshold)


def _tune_threshold(y_true: np.ndarray, y_prob: np.ndarray) -> tuple[float, dict]:
    candidates = np.linspace(0.20, 0.80, 61)
    scores = [f1_score(y_true, (y_prob >= threshold).astype(int), average="macro", zero_division=0) for threshold in candidates]
    index = int(np.argmax(scores))
    threshold = float(candidates[index])
    return threshold, _metrics_from_predictions(y_true, y_prob, threshold)


@torch.no_grad()
def evaluate(model, loader, device, aggregation: str = "mean", threshold: float = 0.5, return_arrays: bool = False):
    model.eval()
    ids, labels, probabilities = [], [], []
    for images, batch_labels, batch_ids in loader:
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy().tolist()
        ids.extend(list(batch_ids))
        labels.extend(batch_labels.numpy().tolist())
        probabilities.extend(probs)
    y_true, y_prob, patient_ids = _patient_predictions(ids, labels, probabilities, aggregation)
    metrics = _metrics_from_predictions(y_true, y_prob, threshold)
    if return_arrays:
        return metrics, y_true, y_prob, patient_ids
    return metrics


@torch.no_grad()
def evaluate_multiclass(model, loader, device, aggregation: str = "mean", return_arrays: bool = False):
    model.eval()
    ids, labels, probabilities = [], [], []
    for images, batch_labels, batch_ids in loader:
        logits = model(images.to(device))
        probs = torch.softmax(logits, dim=1).cpu().numpy()
        ids.extend(list(batch_ids))
        labels.extend(batch_labels.numpy().tolist())
        probabilities.extend(list(probs))
    y_true, y_prob, patient_ids = _patient_multiclass_predictions(ids, labels, probabilities, aggregation)
    metrics = _metrics_from_multiclass_predictions(y_true, y_prob)
    if return_arrays:
        return metrics, y_true, y_prob, patient_ids
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--epochs", type=int, default=6)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--split-seed", type=int, default=None, help="Keep the patient split fixed while changing model initialization.")
    parser.add_argument("--max-patients", type=int, default=None)
    parser.add_argument("--target-fs", type=int, default=1000)
    parser.add_argument("--filter", dest="filter_name", choices=["none", "butterworth", "fir"], default="none")
    parser.add_argument("--quantization-bits", type=int, default=16)
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--image-size", type=int, default=128)
    parser.add_argument("--max-windows-per-recording", type=int, default=2)
    parser.add_argument("--aggregation", choices=["mean", "median", "max", "top25"], default="mean")
    parser.add_argument("--tune-threshold", action="store_true")
    parser.add_argument("--unfreeze-last-blocks", type=int, default=0)
    parser.add_argument("--three-class", action="store_true", help="Use Absent/Present/Unknown and challenge-aligned metrics.")
    parser.add_argument("--train-size", type=float, default=0.70)
    parser.add_argument("--validation-size", type=float, default=0.15)
    args = parser.parse_args()

    seed_everything(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.three_class and args.tune_threshold:
        raise SystemExit("--tune-threshold is only supported for the binary CNN task.")
    frame = _patient_frame(args.data_dir, three_class=args.three_class)
    if args.max_patients is not None:
        frame = frame.head(args.max_patients).copy()
    split_seed = args.seed if args.split_seed is None else args.split_seed
    train_size = 0.65 if args.three_class else args.train_size
    validation_size = 0.10 if args.three_class else args.validation_size
    train_ids, val_ids, test_ids = patient_split(frame, seed=split_seed, train_size=train_size, validation_size=validation_size)
    print(f"Device: {device}; patients: train={len(train_ids)}, val={len(val_ids)}, test={len(test_ids)}")

    started = time.perf_counter()
    train_examples = _build_examples(args.data_dir, set(train_ids), args.target_fs, args.filter_name, args.quantization_bits, 3.0, 1.5, args.max_windows_per_recording, args.three_class)
    val_examples = _build_examples(args.data_dir, set(val_ids), args.target_fs, args.filter_name, args.quantization_bits, 3.0, 1.5, args.max_windows_per_recording, args.three_class)
    test_examples = _build_examples(args.data_dir, set(test_ids), args.target_fs, args.filter_name, args.quantization_bits, 3.0, 1.5, args.max_windows_per_recording, args.three_class)
    print(f"Windows: train={len(train_examples)}, val={len(val_examples)}, test={len(test_examples)}; prep={time.perf_counter()-started:.1f}s")

    train_loader = DataLoader(PCGWindowDataset(train_examples, args.target_fs, args.image_size), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(PCGWindowDataset(val_examples, args.target_fs, args.image_size), batch_size=args.batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(PCGWindowDataset(test_examples, args.target_fs, args.image_size), batch_size=args.batch_size, shuffle=False, num_workers=0)

    weights = MobileNet_V3_Small_Weights.DEFAULT if args.pretrained else None
    model = mobilenet_v3_small(weights=weights)
    n_classes = 3 if args.three_class else 2
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, n_classes)
    if args.freeze_backbone:
        for parameter in model.features.parameters():
            parameter.requires_grad = False
    if args.unfreeze_last_blocks > 0:
        for block in list(model.features.children())[-args.unfreeze_last_blocks:]:
            for parameter in block.parameters():
                parameter.requires_grad = True
    model.to(device)
    counts = np.bincount([example.label for example in train_examples], minlength=n_classes).astype(np.float32)
    class_weights = torch.tensor(counts.sum() / (n_classes * np.maximum(counts, 1.0)), dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad), lr=1e-4, weight_decay=1e-4)
    best_val = -1.0
    best_state = None
    history = []
    stale = 0
    train_started = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for images, labels, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(device)), labels.to(device))
            loss.backward()
            optimizer.step()
            losses.append(float(loss.item()))
        if args.three_class:
            val_metrics, _, _, _ = evaluate_multiclass(model, val_loader, device, args.aggregation, return_arrays=True)
            threshold = 0.5
        else:
            val_metrics, y_val, p_val, _ = evaluate(model, val_loader, device, args.aggregation, 0.5, return_arrays=True)
            threshold = 0.5
            if args.tune_threshold:
                threshold, val_metrics = _tune_threshold(y_val, p_val)
        row = {"epoch": epoch, "loss": float(np.mean(losses)), **{f"val_{k}": v for k, v in val_metrics.items() if isinstance(v, (int, float))}}
        history.append(row)
        print(f"epoch {epoch}: loss={row['loss']:.4f}, val_f1={val_metrics['f1_macro']:.4f}, val_balanced={val_metrics['balanced_accuracy']:.4f}, threshold={threshold:.2f}")
        if val_metrics["f1_macro"] > best_val:
            best_val = val_metrics["f1_macro"]
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            best_threshold = threshold
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    if "best_threshold" not in locals():
        best_threshold = 0.5
    test_metrics = evaluate_multiclass(model, test_loader, device, args.aggregation) if args.three_class else evaluate(model, test_loader, device, args.aggregation, best_threshold)
    result = {
        "model": "MobileNetV3-Small",
        "seed": args.seed,
        "pretrained": bool(args.pretrained),
        "freeze_backbone": bool(args.freeze_backbone),
        "image_size": args.image_size,
        "max_windows_per_recording": args.max_windows_per_recording,
        "aggregation": args.aggregation,
        "tune_threshold": bool(args.tune_threshold),
        "best_threshold": float(best_threshold),
        "unfreeze_last_blocks": args.unfreeze_last_blocks,
        "task": "three_class_murmur" if args.three_class else "binary_murmur",
        "class_names": list(CHALLENGE_LABELS) if args.three_class else ["Absent", "Present"],
        "train_size": train_size,
        "validation_size": validation_size,
        "split_seed": split_seed,
        "parameters": int(sum(parameter.numel() for parameter in model.parameters())),
        "device": str(device),
        "target_fs": args.target_fs,
        "filter": args.filter_name,
        "quantization_bits": args.quantization_bits,
        "n_train_windows": len(train_examples),
        "n_validation_windows": len(val_examples),
        "n_test_windows": len(test_examples),
        "n_train_patients": len(train_ids),
        "n_validation_patients": len(val_ids),
        "n_test_patients": len(test_ids),
        "prep_seconds": float(train_started - started),
        "train_seconds": float(time.perf_counter() - train_started),
        "epochs_completed": len(history),
        "test": test_metrics,
        "history": history,
    }
    torch.save({"state_dict": model.state_dict(), "args": vars(args), "result": result}, args.output_dir / "model.pt")
    (args.output_dir / "metrics.json").write_text(json.dumps(result, indent=2, default=float), encoding="utf-8")
    print(json.dumps({key: value for key, value in result.items() if key not in {"history", "test"}}, indent=2, default=float))
    print(json.dumps(result["test"], indent=2, default=float))


if __name__ == "__main__":
    main()
