"""PhysioNet CirCor file parsing and patient-wise manifests."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd
from scipy.io import wavfile


@dataclass(frozen=True)
class RecordingRef:
    location: str
    wav_path: Path
    segmentation_path: Path | None


@dataclass(frozen=True)
class PatientRecord:
    patient_id: str
    label: str
    outcome: str | None
    recordings: tuple[RecordingRef, ...]


def _value(lines: list[str], key: str) -> str | None:
    prefix = f"#{key}:"
    for line in lines:
        if line.startswith(prefix):
            value = line[len(prefix):].strip()
            return None if value.lower() in {"nan", "none", ""} else value
    return None


def parse_patient_file(path: str | Path) -> PatientRecord:
    path = Path(path)
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    patient_id = path.stem
    label = _value(lines, "Murmur") or "Unknown"
    outcome = _value(lines, "Outcome")
    recordings: list[RecordingRef] = []
    for line in lines[1:]:
        fields = line.split()
        if len(fields) < 4 or fields[0].startswith("#"):
            continue
        location, _, wav_name, *rest = fields
        wav_path = path.parent / wav_name
        segmentation = path.parent / rest[0] if rest else None
        if wav_path.exists():
            recordings.append(RecordingRef(location, wav_path, segmentation))
    return PatientRecord(patient_id, label, outcome, tuple(recordings))


def iter_patients(data_dir: str | Path) -> Iterator[PatientRecord]:
    data_dir = Path(data_dir)
    for path in sorted(data_dir.glob("*.txt")):
        yield parse_patient_file(path)


def build_manifest(data_dir: str | Path, include_unknown: bool = False) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for patient in iter_patients(data_dir):
        if patient.label == "Unknown" and not include_unknown:
            continue
        for recording in patient.recordings:
            rows.append(
                {
                    "patient_id": patient.patient_id,
                    "recording_id": recording.wav_path.stem,
                    "label": patient.label,
                    "outcome": patient.outcome,
                    "location": recording.location,
                    "wav_path": str(recording.wav_path),
                    "segmentation_path": str(recording.segmentation_path or ""),
                }
            )
    return pd.DataFrame(rows)


def load_wav(path: str | Path) -> tuple[int, np.ndarray]:
    fs, signal = wavfile.read(path)
    signal = np.asarray(signal)
    if np.issubdtype(signal.dtype, np.integer):
        info = np.iinfo(signal.dtype)
        scale = max(abs(info.min), info.max)
        signal = signal.astype(np.float32) / float(scale)
    else:
        signal = signal.astype(np.float32)
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    signal = np.nan_to_num(signal, nan=0.0, posinf=0.0, neginf=0.0)
    return int(fs), signal
