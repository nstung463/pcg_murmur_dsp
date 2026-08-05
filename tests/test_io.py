from pathlib import Path

from pcg_dsp.io import parse_patient_file


def test_parse_patient_file(tmp_path: Path):
    (tmp_path / "1234_AV.wav").write_bytes(b"")
    patient = tmp_path / "1234.txt"
    patient.write_text(
        "1234 1 4000\n"
        "AV 1234_AV.hea 1234_AV.wav 1234_AV.tsv\n"
        "#Murmur: Present\n"
        "#Outcome: Abnormal\n",
        encoding="utf-8",
    )
    parsed = parse_patient_file(patient)
    assert parsed.patient_id == "1234"
    assert parsed.label == "Present"
    assert parsed.outcome == "Abnormal"
    assert parsed.recordings[0].location == "AV"
