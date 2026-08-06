# Robust PCG Murmur Detection with DSP

DSP501 final-project implementation for studying how sampling, quantization,
digital filtering and time-frequency representations affect heart-murmur detection
under noise.

The project is based on the public CirCor DigiScope / PhysioNet Challenge 2022
training data. The dataset contains pediatric phonocardiogram recordings with
patient-level murmur labels and S1/S2 segmentation annotations. Cite the dataset
and its license according to the official PhysioNet page:
<https://physionet.org/content/circor-heart-sound/1.0.3/>.

## Research question

Which DSP front-end gives the best robustness/compute trade-off for binary
`Present` versus `Absent` murmur detection when sampling rate, quantization,
filter design and noise level are controlled?

The primary split is patient-wise. All recordings and noisy variants from one
patient stay in the same partition. This prevents the common leakage where the
same pediatric patient appears in both train and test through different
auscultation locations.

## Setup

Use Python 3.11 on Windows:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python -m pip install -r requirements.txt
.\.venv\Scripts\python -m pip install . --no-deps
$env:PYTHONPATH = "$PWD\src"
```

The non-editable install avoids a setuptools encoding issue caused by the
Vietnamese workspace path. `PYTHONPATH` keeps source edits immediately visible.

Download the public archive:

```powershell
.\.venv\Scripts\python scripts\download_circor.py
```

If the ZIP endpoint is slow, download a patient-wise real-data subset directly:

```powershell
.\.venv\Scripts\python scripts\download_subset.py `
  --patients 100 `
  --output data/circor-subset/training_data
```

Audit the downloaded recordings before training:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python scripts\data_audit.py `
  --data-dir data/circor-subset/training_data `
  --output-dir artifacts/circor_subset_audit
```

For a dependency and data-path smoke test without downloading clinical data:

```powershell
.\.venv\Scripts\python scripts\make_synthetic_fixture.py
```

Create a recording manifest:

```powershell
.\.venv\Scripts\pcg-dsp.exe prepare-manifest `
  --data-dir data/circor-heart-sound/1.0.3/training_data `
  --output artifacts/manifest.csv
```

Run a small smoke experiment first:

```powershell
.\.venv\Scripts\pcg-dsp.exe run-experiment `
  --config configs/default.yaml `
  --data-dir data/circor-heart-sound/1.0.3/training_data `
  --max-patients 40 `
  --run-dir artifacts/runs/smoke
```

Run a controlled filter/feature matrix:

```powershell
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python scripts\run_experiments.py `
  --data-dir data/circor-subset/training_data `
  --output artifacts/circor_100_experiments.csv `
  --models svm,mlp
```

Run sampling/quantization/noise robustness experiments for the selected
SVM+Butterworth+hybrid front-end:

```powershell
.\.venv\Scripts\python scripts\run_robustness.py `
  --data-dir data/circor-subset/training_data `
  --output artifacts/circor_100_robustness.csv
```

Create a report-ready comparison figure:

```powershell
.\.venv\Scripts\python scripts\make_figures.py
```

Render a signal-level DSP figure for a WAV file:

```powershell
.\.venv\Scripts\python scripts\make_signal_report.py `
  --wav data/circor-subset/training_data/13918_AV.wav
```

Run inference on one recording after a model run:

```powershell
.\.venv\Scripts\python scripts\predict_file.py `
  --wav data/circor-subset/training_data/13918_AV.wav `
  --model artifacts/runs/svm_butterworth_hybrid/model.joblib `
  --config configs/default.yaml
```

## Interactive FE demo

The backend inference service is shared by the CLI and Streamlit UI. Install
the UI dependency and launch the demo from the project root:

```powershell
.\.venv\Scripts\python -m pip install -r requirements-ui.txt
$env:PYTHONPATH = "$PWD\src"
.\.venv\Scripts\python -m streamlit run app.py
```

The UI first asks for one recording source: upload an external WAV, or select
any WAV already present in the project's `training_data` folder without
opening the browser Downloads directory. The project-data option also provides
a bundled demo recording. The user can choose a trained SVM/MLP model preset
from a dropdown, run the saved model bundle, and view the raw/processed waveform, FFT
magnitude, Welch PSD, filter frequency response, log-STFT spectrogram, class
probabilities, confidence, ground truth for labelled project recordings, and DSP
configuration used. The result card clearly marks `CORRECT` or `INCORRECT` when
a reference label is available. Sampling rate,
quantization, filter, and controlled noise preview can be changed in the
sidebar. The A/B comparison runs the training baseline next to the selected
DSP preview, and the result can be downloaded as JSON. The selected model's
feature mode remains fixed so inference stays compatible with the trained
estimator.

Run the unit tests:

```powershell
.\.venv\Scripts\pytest
```

## Experiment tracks

The controlled experiments should be run in stages rather than as one large
Cartesian product:

1. no filter versus Butterworth IIR versus linear-phase FIR;
2. PSD/statistical features versus MFCC versus STFT versus hybrid features;
3. rule/statistical baseline versus SVM versus compact MLP neural model;
4. 4 kHz, 2 kHz and 1 kHz sampling with 16/12/8-bit quantization;
5. white, pink and impulsive noise at multiple SNR values;
6. ablations removing each DSP component individually.

The current neural baseline is an `MLPClassifier`, chosen to keep the full
experiment reproducible on CPU. A spectrogram CNN can be added later without
changing the manifest or DSP interfaces.

The current real-data verification uses a direct-download subset because the
large PhysioNet ZIP endpoint can be heavily throttled. The expanded verified
subset contains 100 patients, 325 recordings, 4 kHz WAV files, and no invalid
files. Its audit, full matrix and robustness outputs are under
`artifacts/circor_100_audit/`, `artifacts/circor_100_experiments.csv` and
`artifacts/circor_100_robustness.csv`. These are subset ablation results, not
final claims for the complete CirCor cohort; repeat the same commands after a
complete archive download if the full dataset becomes available.

Generate the final Word/PDF report after the matrix artifacts exist:

\`\`\`powershell
$env:NODE_PATH = "C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\node_modules"
$node = "C:\Users\Admin\.cache\codex-runtimes\codex-primary-runtime\dependencies\node\bin\node.exe"
& $node scripts\create_final_report.js
\`\`\`

This writes \`artifacts/final_report_dsp501.docx\`; LibreOffice can export the
same document to PDF for submission.

Report macro-F1 as the primary metric, together with accuracy, precision, recall,
balanced accuracy, confusion matrices, runtime and error analysis. This is a
research prototype and must not be presented as a clinical diagnostic device.
