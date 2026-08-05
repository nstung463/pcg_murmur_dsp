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
    from pcg_dsp.dsp import apply_filter, design_filter, stft_matrix, quantize, resample_signal
    from pcg_dsp.pipeline import make_patient_table, train_evaluate
    from pcg_dsp.service import analyze_recording, load_model_bundle

    BASE_CONFIG = yaml.safe_load((PROJECT_DIR / 'configs' / 'default.yaml').read_text(encoding='utf-8'))
    BASE_CONFIG['data']['root'] = str(DATA_DIR)
    BASE_CONFIG['data']['max_patients'] = MAX_PATIENTS
    print('Project package import OK')
    '''),
    cell("code", r'''
    # 4) Audit data: counts, labels, sample rates
    manifest = build_manifest(DATA_DIR, include_unknown=True)
    rates = [load_wav(path)[0] for path in tqdm(manifest.wav_path.sample(min(100, len(manifest), random_state=42)), desc='checking sample rates')]
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
