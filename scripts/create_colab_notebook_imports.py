from __future__ import annotations

import json
import textwrap
from pathlib import Path


def cell(kind: str, text: str) -> dict:
    base = {"cell_type": kind, "metadata": {}, "source": textwrap.dedent(text).strip("\n").splitlines(True)}
    if kind == "code":
        base.update({"execution_count": None, "outputs": []})
    return base


cells = [
    cell("markdown", r'''
    # PCG Murmur Detection — Colab end-to-end

    Notebook này **import trực tiếp code trong project**, không copy lại các hàm DSP. Cell setup sẽ clone repository GitHub rồi cài package ở chế độ editable.

    Flow: `install → import project → download full CirCor → audit → baseline → 24-run matrix → robustness → plots → model bundle → inference`.
    '''),
    cell("code", r'''
    # 1) Clone repository và cài package của project
    from pathlib import Path
    import subprocess, sys

    REPO_URL = 'https://github.com/nstung463/pcg_murmur_dsp.git'
    PROJECT_DIR = Path('/content/pcg_murmur_dsp')
    if not (PROJECT_DIR / 'src' / 'pcg_dsp').exists():
        subprocess.run(['git', 'clone', REPO_URL, str(PROJECT_DIR)], check=True)

    SRC_DIR = PROJECT_DIR / 'src'
    if not (SRC_DIR / 'pcg_dsp').exists():
        raise RuntimeError(f'Không tìm thấy source package: {SRC_DIR / "pcg_dsp"}')
    if str(SRC_DIR) not in sys.path:
        sys.path.insert(0, str(SRC_DIR))
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-e', str(PROJECT_DIR)], check=True)
    subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', 'matplotlib', 'seaborn', 'tqdm', 'requests', 'awscli'], check=True)
    print('Imported project from:', PROJECT_DIR)
    print('Python source path:', SRC_DIR)
    '''),
    cell("code", r'''
    # 2) Download full CirCor training data
    import subprocess
    DATA_DIR = Path('/content/data/circor-heart-sound/1.0.3/training_data')
    OUTPUT_DIR = Path('/content/pcg_dsp_outputs')
    DATA_DIR.mkdir(parents=True, exist_ok=True); OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # False = full cohort; True = 100 patients để test nhanh trước
    FAST_MODE = False
    MAX_PATIENTS = 100 if FAST_MODE else None
    RUN_MATRIX = True
    RUN_ROBUSTNESS = True

    if not list(DATA_DIR.glob('*.txt')):
        subprocess.run([
            'aws', 's3', 'sync', '--no-sign-request',
            's3://physionet-open/circor-heart-sound/1.0.3/training_data',
            str(DATA_DIR), '--only-show-errors'
        ], check=True)

    print('Patient files:', len(list(DATA_DIR.glob('*.txt'))))
    if not list(DATA_DIR.glob('*.txt')):
        raise RuntimeError('Download chưa tạo file *.txt; thử chạy lại cell hoặc dùng ZIP của PhysioNet.')
    '''),
    cell("code", r'''
    # 3) Import đúng các module của project
    import copy, json, yaml, joblib, sys
    if str(PROJECT_DIR / 'src') not in sys.path:
        sys.path.insert(0, str(PROJECT_DIR / 'src'))
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm.auto import tqdm
    from IPython.display import display

    from pcg_dsp.io import build_manifest, iter_patients, load_wav
    from pcg_dsp.dsp import apply_filter, design_filter, stft_matrix, quantize, resample_signal, segment_signal, feature_vector
    from pcg_dsp.pipeline import make_patient_table, train_evaluate, patient_split
    from pcg_dsp.service import analyze_recording, load_model_bundle

    BASE_CONFIG = yaml.safe_load((PROJECT_DIR / 'configs' / 'default.yaml').read_text(encoding='utf-8'))
    BASE_CONFIG['data']['root'] = str(DATA_DIR)
    BASE_CONFIG['data']['max_patients'] = MAX_PATIENTS
    print('Project package import OK')
    '''),
    cell("code", r'''
    # 4) Audit data: counts, labels, sample rates
    manifest = build_manifest(DATA_DIR, include_unknown=True)
    rates = [load_wav(path)[0] for path in tqdm(manifest.wav_path.sample(min(100, len(manifest)), random_state=42), desc='checking sample rates')]
    print(f"Patients: {manifest.patient_id.nunique():,}")
    print(f"Recordings: {len(manifest):,}")
    print('Sample rates:', pd.Series(rates).value_counts().sort_index().to_dict())
    display(manifest.groupby('label').agg(patients=('patient_id','nunique'), recordings=('recording_id','count')))
    manifest.to_csv(OUTPUT_DIR / 'manifest.csv', index=False)
    '''),
    cell("code", r'''
    # 5) Baseline đúng theo pipeline trong project
    baseline_cfg = copy.deepcopy(BASE_CONFIG)
    baseline_cfg['model']['kind'] = 'svm'
    baseline_cfg['dsp']['filter'] = 'butterworth'
    baseline_cfg['features']['mode'] = 'hybrid'
    baseline_table = make_patient_table(DATA_DIR, baseline_cfg, MAX_PATIENTS)
    baseline_result = train_evaluate(baseline_table, baseline_cfg, OUTPUT_DIR / 'baseline_model')
    print(json.dumps(baseline_result, indent=2, default=float))
    '''),
    cell("code", r'''
    # 6) Full matrix 3 filters × 4 feature modes × 2 classifiers
    matrix_rows = []
    if RUN_MATRIX:
        for model_name in ['svm', 'mlp']:
            for filter_name in ['none', 'butterworth', 'fir']:
                for feature_name in ['psd', 'mfcc', 'stft', 'hybrid']:
                    cfg = copy.deepcopy(BASE_CONFIG)
                    cfg['model']['kind'] = model_name
                    cfg['dsp']['filter'] = filter_name
                    cfg['features']['mode'] = feature_name
                    run_name = f'{model_name}_{filter_name}_{feature_name}'
                    table = make_patient_table(DATA_DIR, cfg, MAX_PATIENTS)
                    result = train_evaluate(table, cfg, OUTPUT_DIR / 'runs' / run_name)
                    matrix_rows.append({'model': model_name, 'filter': filter_name, 'features': feature_name, **result})
                    print(run_name, 'Macro-F1=', round(result['f1_macro'], 4))
        matrix_df = pd.DataFrame(matrix_rows).sort_values('f1_macro', ascending=False)
        matrix_df.to_csv(OUTPUT_DIR / 'full_matrix.csv', index=False)
        display(matrix_df[['model','filter','features','f1_macro','balanced_accuracy','accuracy']].head(10))
    else:
        matrix_df = pd.DataFrame()
    '''),
    cell("code", r'''
    # 7) Matrix heatmap
    if not matrix_df.empty:
        heat = matrix_df.pivot_table(index=['model','filter'], columns='features', values='f1_macro')
        plt.figure(figsize=(10, 5))
        sns.heatmap(heat, annot=True, fmt='.3f', cmap='YlGnBu', vmin=0, vmax=1)
        plt.title('DSP ablation matrix — Macro-F1'); plt.tight_layout(); plt.show()
    '''),
    cell("code", r'''
    # 8) Robustness đúng theo run_robustness.py của project
    robustness_rows = []
    if RUN_ROBUSTNESS:
        experiments = []
        for fs in [1000, 2000, 4000]: experiments.append((f'sampling_{fs}hz', fs, 16, False, 'white', 10))
        for bits in [8, 12, 16]: experiments.append((f'quantization_{bits}bit', 1000, bits, False, 'white', 10))
        for kind, snr in [('white',20), ('white',10), ('pink',10), ('impulse',10)]: experiments.append((f'noise_{kind}_{snr}db', 1000, 16, True, kind, snr))
        for name, fs, bits, enabled, kind, snr in experiments:
            cfg = copy.deepcopy(BASE_CONFIG)
            cfg['model']['kind'] = 'svm'; cfg['dsp']['filter'] = 'butterworth'; cfg['features']['mode'] = 'hybrid'
            cfg['dsp']['target_fs'] = fs; cfg['dsp']['quantization_bits'] = bits
            cfg['noise'] = {'enabled': enabled, 'kind': kind, 'snr_db': snr}
            table = make_patient_table(DATA_DIR, cfg, MAX_PATIENTS)
            result = train_evaluate(table, cfg, OUTPUT_DIR / 'robustness_runs' / name)
            robustness_rows.append({'experiment': name, 'target_fs': fs, 'quantization_bits': bits, 'noise_kind': kind if enabled else 'none', 'snr_db': snr if enabled else None, **result})
            print(name, 'Macro-F1=', round(result['f1_macro'], 4))
        robustness_df = pd.DataFrame(robustness_rows)
        robustness_df.to_csv(OUTPUT_DIR / 'robustness.csv', index=False)
        display(robustness_df[['experiment','f1_macro','balanced_accuracy']])
    else:
        robustness_df = pd.DataFrame()
    '''),
    cell("code", r'''
    # 9) Robustness plot
    if not robustness_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ax, prefix, title in zip(axes, ['sampling_', 'quantization_', 'noise_'], ['Sampling', 'Quantization', 'Noise']):
            part = robustness_df[robustness_df.experiment.str.startswith(prefix)]
            sns.barplot(data=part, x='experiment', y='f1_macro', ax=ax, color='#2B7A78')
            ax.set_title(title); ax.set_ylim(0, 1); ax.tick_params(axis='x', rotation=35); ax.set_xlabel(''); ax.set_ylabel('Macro-F1')
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'robustness.png', dpi=160); plt.show()
    '''),
    cell("code", r'''
    # 10) Signal report bằng đúng hàm DSP của project
    sample_wav = Path(manifest.iloc[0].wav_path)
    fs_raw, raw = load_wav(sample_wav)
    fs_target = 1000
    processed = resample_signal(raw, fs_raw, fs_target)
    processed = quantize(processed, 16)
    processed = apply_filter(processed, design_filter(fs_target, 'butterworth'))
    freqs, times, matrix = stft_matrix(processed, fs_target)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    axes[0].plot(np.arange(len(raw))/fs_raw, raw, lw=.6, color='#D96445'); axes[0].set_title('Raw PCG waveform')
    axes[1].plot(np.fft.rfftfreq(2048, 1/fs_target), np.abs(np.fft.rfft(processed, n=2048)), color='#2B7A78'); axes[1].set_title('FFT magnitude'); axes[1].set_xlim(0,500)
    from scipy import signal
    f_psd, p_psd = signal.welch(processed, fs=fs_target, nperseg=min(512,len(processed))); axes[2].semilogy(f_psd,p_psd,color='#6A5C8A'); axes[2].set_title('Welch PSD'); axes[2].set_xlim(0,500)
    axes[3].pcolormesh(times, freqs, matrix, shading='auto', cmap='viridis'); axes[3].set_title('Log-STFT spectrogram'); axes[3].set_ylim(0,500); axes[3].set_xlabel('Time (s)')
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / 'signal_report.png', dpi=160); plt.show()
    print('Sample:', sample_wav.name, '| source fs:', fs_raw)
    '''),
    cell("code", r'''
    # 11) Chọn model tốt nhất và inference WAV mới
    if not matrix_df.empty:
        best = matrix_df.iloc[0]
        best_cfg = copy.deepcopy(BASE_CONFIG)
        best_cfg['model']['kind'] = best['model']; best_cfg['dsp']['filter'] = best['filter']; best_cfg['features']['mode'] = best['features']
        best_dir = OUTPUT_DIR / 'best_model'
        best_table = make_patient_table(DATA_DIR, best_cfg, MAX_PATIENTS)
        best_result = train_evaluate(best_table, best_cfg, best_dir)
        model_bundle = load_model_bundle(best_dir / 'model.joblib')
        print('Best config:', best[['model','filter','features']].to_dict())
        print('Best result:', json.dumps(best_result, indent=2, default=float))
    else:
        model_bundle = load_model_bundle(OUTPUT_DIR / 'baseline_model' / 'model.joblib')

    # Đổi path này thành file WAV muốn test.
    def predict_wav(wav_path):
        model, model_config = model_bundle
        result = analyze_recording(wav_path, model, model_config)
        return {'label': result['label'], 'probabilities': result['probabilities']}

    # Ví dụ: predict_wav('/content/my_recording.wav')
    print('All outputs saved under:', OUTPUT_DIR)
    '''),
    cell("markdown", r'''
    ## Lưu ý khi chạy

    - `FAST_MODE = True` dùng 100 bệnh nhân để kiểm tra pipeline.
    - `FAST_MODE = False` chạy full cohort 1.568 bệnh nhân.
    - `RUN_MATRIX` và `RUN_ROBUSTNESS` có thể tắt riêng nếu Colab hết thời gian.
    - Kết quả được lưu trong `/content/pcg_dsp_outputs`.
    - Đây là research prototype, không phải công cụ chẩn đoán lâm sàng.
    '''),
]

# Mở rộng notebook thành tutorial: mỗi cell training có mục tiêu, giải thích
# và visualization để người đọc thấy dữ liệu thay đổi như thế nào.
tutorial_cells = []
for index, current in enumerate(cells):
    if index == 1:
        tutorial_cells.append(cell("markdown", r'''
        ## Step 0 — Chuẩn bị code

        Colab clone repository, thêm `src/` vào Python path và cài package `pcg_dsp`. Từ đây notebook gọi trực tiếp code project, không viết lại các hàm DSP.
        '''))
    elif index == 2:
        tutorial_cells.append(cell("markdown", r'''
        ## Step 1 — Dataset: WAV là một dãy số

        File WAV chứa các samples biên độ theo thời gian. Cell dưới tải metadata và recordings từ CirCor. `FAST_MODE=True` chỉ lấy 100 bệnh nhân để kiểm tra nhanh; `False` chạy full cohort.
        '''))
    elif index == 3:
        tutorial_cells.append(cell("markdown", r'''
        ## Step 2 — Import đúng code project

        Các module được dùng là `pcg_dsp.io`, `pcg_dsp.dsp`, `pcg_dsp.pipeline` và `pcg_dsp.service`. Đây cũng chính là code được app Streamlit dùng.
        '''))
    elif index == 4:
        tutorial_cells.append(cell("markdown", r'''
        ## Step 3 — Audit trước khi train

        Ta đếm bệnh nhân, recording, label và sample rate để biết input thật sự có gì trước khi chạy model.
        '''))
    elif index == 5:
        tutorial_cells.append(cell("markdown", r'''
        ## Step 4 — Nhìn waveform trước và sau DSP

        Một WAV đi qua resample, quantization và band-pass filter. Visualization kế tiếp cho thấy dãy số thay đổi ra sao.
        '''))
    elif index == 6:
        tutorial_cells.append(cell("markdown", r'''
        ## Step 5 — Từ waveform thành feature và patient vector

        Window 3 giây được biến thành feature vector. Nhiều window được trung bình thành recording vector; nhiều recording được trung bình thành patient vector.
        '''))
    elif index == 7:
        tutorial_cells.append(cell("markdown", r'''
        ## Step 6 — Đọc kết quả matrix

        Baseline là Butterworth + Hybrid + SVM. Split theo patient để model không nhìn thấy recording khác của cùng một bệnh nhân ở test.
        '''))
    elif index == 8:
        tutorial_cells.append(cell("markdown", r'''
        ## Step 7 — Robustness

        So sánh 3 filter × 4 feature modes × 2 classifiers để đo ảnh hưởng của từng khối DSP.
        '''))
    elif index == 9:
        tutorial_cells.append(cell("markdown", r'''
        ## Step 8 — Đọc robustness plot

        Mỗi ô là Macro-F1 trên các test patients. Ô cao hơn là cấu hình tạo ra biểu diễn tín hiệu tốt hơn cho bài toán Absent/Present.
        '''))
    elif index == 10:
        tutorial_cells.append(cell("markdown", r'''
        ## Step 9 — Robustness

        Cố ý thay sampling rate, bit depth và loại noise để xem chất lượng feature/model suy giảm thế nào.
        '''))
    elif index == 11:
        tutorial_cells.append(cell("markdown", r'''
        ## Step 10 — Signal report

        Waveform, FFT, PSD và STFT cho thấy cùng một tín hiệu được nhìn ở miền thời gian, miền tần số và time-frequency.
        '''))
    elif index == 12:
        tutorial_cells.append(cell("markdown", r'''
        ## Step 11 — Inference

        Model tốt nhất được lưu cùng config preprocessing. WAV mới phải đi qua đúng config đó trước khi dự đoán.
        '''))

    tutorial_cells.append(current)

# Visualization sau audit: raw waveform, sample numbers và filtered waveform.
tutorial_cells.insert(
    tutorial_cells.index(cells[4]) + 2,
    cell("code", r'''
    # Visualization A — file WAV trước và sau DSP
    sample_wav = Path(manifest.iloc[0].wav_path)
    fs_raw, raw = load_wav(sample_wav)
    fs_target = 1000
    resampled = resample_signal(raw, fs_raw, fs_target)
    quantized = quantize(resampled, bits=16)
    filtered = apply_filter(
        quantized,
        design_filter(fs_target, "butterworth", 25, 400, 4, 129),
    )

    print("File:", sample_wav.name)
    print("First 12 raw samples:", raw[:12])
    print("Raw fs:", fs_raw, "Hz | target fs:", fs_target, "Hz")
    print("Duration:", round(len(raw) / fs_raw, 2), "seconds")

    seconds = min(10, len(raw) / fs_raw)
    raw_n = int(seconds * fs_raw)
    filtered_n = int(seconds * fs_target)
    fig, axes = plt.subplots(2, 1, figsize=(14, 6))
    axes[0].plot(np.arange(raw_n) / fs_raw, raw[:raw_n], color="#D96445", lw=.7)
    axes[0].set_title("Raw waveform — dãy số biên độ ban đầu")
    axes[1].plot(np.arange(filtered_n) / fs_target, filtered[:filtered_n], color="#2B7A78", lw=.7)
    axes[1].set_title("Sau resample + quantization + Butterworth 25–400 Hz")
    for ax in axes:
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
    '''),
)

# Visualization sau baseline: window overlap và feature vector.
tutorial_cells.insert(
    tutorial_cells.index(cells[5]) + 2,
    cell("code", r'''
    # Visualization B — segmentation 3 giây, hop 1,5 giây
    segments = segment_signal(
        filtered,
        fs_target,
        seconds=3.0,
        hop_seconds=1.5,
    )
    feature_config = {
        key: value
        for key, value in BASE_CONFIG["features"].items()
        if key != "mode"
    }
    hybrid_vector = feature_vector(
        segments[0],
        fs_target,
        mode="hybrid",
        **feature_config,
    )
    print("Samples per window:", 3 * fs_target)
    print("Windows in sample recording:", len(segments))
    print("Hybrid feature vector length:", len(hybrid_vector))
    print("First 10 features:", hybrid_vector[:10])

    view_n = min(len(filtered), 8 * fs_target)
    t = np.arange(view_n) / fs_target
    plt.figure(figsize=(14, 3.5))
    plt.plot(t, filtered[:view_n], color="#17313B", lw=.7)
    for start in np.arange(0, max(0, view_n / fs_target - 3), 1.5):
        plt.axvspan(start, start + 3.0, color="#2B7A78", alpha=.10)
        plt.axvline(start, color="#D96445", ls="--", lw=.8)
    plt.title("Segmentation — mỗi vùng xanh là một window 3 s")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.tight_layout()
    plt.show()
    print("Patient table shape:", baseline_table.shape)
    display(baseline_table.head(3))
    '''),
)

# Visualization patient-wise split.
tutorial_cells.insert(
    tutorial_cells.index(cells[5]) + 3,
    cell("code", r'''
    # Visualization C — split theo patient, không split recording ngẫu nhiên
    train_ids, validation_ids, test_ids = patient_split(
        baseline_table,
        seed=baseline_cfg["split"]["seed"],
    )
    split_sizes = pd.Series({
        "Train patients": len(train_ids),
        "Validation patients": len(validation_ids),
        "Test patients": len(test_ids),
    })
    display(split_sizes.to_frame("count"))
    split_sizes.plot.bar(
        figsize=(6, 3),
        color=["#2B7A78", "#D96445", "#6A5C8A"],
        title="Patient-wise split",
    )
    plt.ylabel("Number of patients")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()
    '''),
)

cells = tutorial_cells

notebook = {
    "cells": cells,
    "metadata": {"colab": {"name": "PCG DSP Full End-to-End", "provenance": []}, "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parents[1] / 'notebooks' / 'PCG_DSP_full_end_to_end_colab.ipynb'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(notebook, ensure_ascii=False, indent=1), encoding='utf-8')
print(out.as_posix().encode('ascii', 'backslashreplace').decode())
