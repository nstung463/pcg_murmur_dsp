from __future__ import annotations

import json
from pathlib import Path
import textwrap


def md(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": textwrap.dedent(text).strip("\n").splitlines(True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [], "source": textwrap.dedent(text).strip("\n").splitlines(True)}


cells = [
    md(r'''
    # Robust PCG Murmur Detection with DSP — Colab end-to-end

    Notebook này chạy độc lập trên Google Colab:

    `install → download full CirCor → audit → DSP features → patient-wise split → matrix → robustness → plots → model bundle → inference`

    Dataset gốc CirCor v1.0.3 có 1.568 bệnh nhân và 5.272 recordings. Notebook mặc định dùng toàn bộ training data. Nếu muốn chạy thử nhanh trước, đặt `FAST_MODE = True` ở cell cấu hình.
    '''),
    code(r'''
    # 1) Cài dependency cho Colab
    %pip -q install numpy pandas scipy scikit-learn matplotlib seaborn librosa soundfile PyWavelets tqdm joblib requests awscli
    '''),
    code(r'''
    # 2) Cấu hình và tải full dataset
    from pathlib import Path
    import os, subprocess, shutil

    DATA_DIR = Path('/content/data/circor-heart-sound/1.0.3/training_data')
    OUTPUT_DIR = Path('/content/pcg_dsp_outputs')
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # False = full cohort; True = 100 bệnh nhân đầu tiên để kiểm thử nhanh
    FAST_MODE = False
    MAX_PATIENTS = 100 if FAST_MODE else None
    RUN_MATRIX = True
    RUN_ROBUSTNESS = True

    # S3 thường ổn định hơn endpoint ZIP của PhysioNet.
    # Nếu S3 bị lỗi, đổi DOWNLOAD_METHOD thành 'zip' và chạy lại cell.
    DOWNLOAD_METHOD = 's3'
    S3_URI = 's3://physionet-open/circor-heart-sound/1.0.3/training_data'
    ZIP_URL = 'https://physionet.org/content/circor-heart-sound/get-zip/1.0.3/'

    def count_patient_files():
        return len(list(DATA_DIR.glob('*.txt')))

    if count_patient_files() == 0:
        if DOWNLOAD_METHOD == 's3':
            subprocess.run([
                'aws', 's3', 'sync', '--no-sign-request', S3_URI,
                str(DATA_DIR), '--only-show-errors'
            ], check=True)
        else:
            import zipfile, requests
            archive = Path('/content/circor-heart-sound.zip')
            with requests.get(ZIP_URL, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with archive.open('wb') as f:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
            with zipfile.ZipFile(archive) as zf:
                zf.extractall('/content/data')
            archive.unlink(missing_ok=True)

    print('DATA_DIR:', DATA_DIR)
    print('Patient metadata files:', count_patient_files())
    if count_patient_files() == 0:
        raise RuntimeError('Chưa thấy file *.txt. Hãy thử DOWNLOAD_METHOD = "zip" hoặc kiểm tra S3 access.')
    '''),
    code(r'''
    # 3) Imports và các hàm đọc dữ liệu
    from dataclasses import dataclass
    from pathlib import Path
    from typing import Iterator
    import copy, json, math, warnings
    from IPython.display import display
    import numpy as np
    import pandas as pd
    import matplotlib.pyplot as plt
    import seaborn as sns
    from tqdm.auto import tqdm
    from scipy import signal
    from scipy.io import wavfile
    from scipy.stats import kurtosis, skew
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, confusion_matrix, f1_score, precision_score, recall_score
    from sklearn.model_selection import train_test_split
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    import joblib
    warnings.filterwarnings('ignore')

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

    def _value(lines, key):
        prefix = f'#{key}:'
        for line in lines:
            if line.startswith(prefix):
                value = line[len(prefix):].strip()
                return None if value.lower() in {'nan', 'none', ''} else value
        return None

    def parse_patient_file(path):
        path = Path(path)
        lines = path.read_text(encoding='utf-8', errors='replace').splitlines()
        recordings = []
        for line in lines[1:]:
            fields = line.split()
            if len(fields) < 4 or fields[0].startswith('#'):
                continue
            location, _, wav_name, *rest = fields
            wav_path = path.parent / wav_name
            segmentation = path.parent / rest[0] if rest else None
            if wav_path.exists():
                recordings.append(RecordingRef(location, wav_path, segmentation))
        return PatientRecord(path.stem, _value(lines, 'Murmur') or 'Unknown', _value(lines, 'Outcome'), tuple(recordings))

    def iter_patients(data_dir) -> Iterator[PatientRecord]:
        for path in sorted(Path(data_dir).glob('*.txt')):
            yield parse_patient_file(path)

    def load_wav(path):
        fs, x = wavfile.read(path)
        x = np.asarray(x)
        if np.issubdtype(x.dtype, np.integer):
            info = np.iinfo(x.dtype)
            x = x.astype(np.float32) / float(max(abs(info.min), info.max))
        else:
            x = x.astype(np.float32)
        if x.ndim > 1:
            x = x.mean(axis=1)
        return int(fs), np.nan_to_num(x)

    def build_manifest(data_dir, include_unknown=True):
        rows = []
        for patient in iter_patients(data_dir):
            if patient.label == 'Unknown' and not include_unknown:
                continue
            for rec in patient.recordings:
                rows.append({'patient_id': patient.patient_id, 'recording_id': rec.wav_path.stem, 'label': patient.label, 'outcome': patient.outcome, 'location': rec.location, 'wav_path': str(rec.wav_path)})
        return pd.DataFrame(rows)
    '''),
    code(r'''
    # 4) DSP front-end — giữ cùng logic với project local
    def quantize(x, bits):
        levels = float(2**bits - 1)
        clipped = np.clip(np.asarray(x, dtype=np.float32), -1.0, 1.0)
        return (np.round((clipped + 1.0) * levels / 2.0) * 2.0 / levels - 1.0).astype(np.float32)

    def resample_signal(x, source_fs, target_fs):
        if source_fs == target_fs:
            return np.asarray(x, dtype=np.float32)
        gcd = int(np.gcd(source_fs, target_fs))
        return signal.resample_poly(x, target_fs // gcd, source_fs // gcd).astype(np.float32)

    def design_filter(fs, kind, low_hz=25.0, high_hz=400.0, order=4, fir_taps=129):
        if kind in {'none', 'off'}:
            return None
        nyquist = fs / 2.0
        high_hz = min(float(high_hz), 0.9 * nyquist)
        low_hz = max(1.0, min(float(low_hz), 0.5 * high_hz))
        if kind in {'butterworth', 'iir'}:
            return ('sos', signal.butter(order, [low_hz, high_hz], btype='bandpass', fs=fs, output='sos'))
        taps = int(fir_taps) + (int(fir_taps) % 2 == 0)
        return ('fir', signal.firwin(taps, [low_hz, high_hz], pass_zero=False, fs=fs, window='hamming'))

    def apply_filter(x, spec):
        if spec is None:
            return np.asarray(x, dtype=np.float32)
        kind, coeff = spec
        return (signal.sosfiltfilt(coeff, x) if kind == 'sos' else signal.filtfilt(coeff, [1.0], x)).astype(np.float32)

    def add_noise(x, snr_db, kind='white', seed=42):
        rng = np.random.default_rng(seed)
        x = np.asarray(x, dtype=np.float32)
        noise = rng.normal(size=x.size).astype(np.float32)
        if kind == 'pink':
            noise = np.cumsum(noise); noise -= noise.mean()
        elif kind in {'impulse', 'click'}:
            noise.fill(0.0)
            count = max(1, x.size // 5000)
            idx = rng.choice(x.size, size=min(count, x.size), replace=False)
            noise[idx] = rng.normal(size=len(idx))
        power = float(np.mean(x**2) + 1e-12)
        noise_power = float(np.mean(noise**2) + 1e-12)
        scale = np.sqrt(power / (noise_power * 10 ** (snr_db / 10.0)))
        return (x + scale * noise).astype(np.float32)

    def stft_matrix(x, fs, n_fft=512, hop_length=128):
        freqs, times, z = signal.stft(x, fs=fs, nperseg=n_fft, noverlap=n_fft-hop_length, boundary=None)
        mag = np.abs(z).astype(np.float32)
        return freqs.astype(np.float32), times.astype(np.float32), np.log1p(mag / (mag.max() + 1e-12))

    def segment_signal(x, fs, seconds=3.0, hop_seconds=1.5):
        size, hop = max(1, int(seconds*fs)), max(1, int(hop_seconds*fs))
        if x.size <= size:
            return [np.pad(x, (0, size-x.size))]
        segments = [x[start:start+size] for start in range(0, x.size-size+1, hop)]
        return segments or [np.pad(x, (0, max(0, size-x.size)))]

    def _spectral_features(x, fs, n_fft=512):
        freqs, power = signal.welch(x, fs=fs, nperseg=min(n_fft, len(x)))
        power = np.maximum(power, 1e-12); total = float(power.sum())
        edges = (20.0, 100.0, 200.0, 400.0, min(800.0, fs/2.0))
        values = [float(power[(freqs >= lo) & (freqs < hi)].sum()/total) for lo, hi in zip(edges, edges[1:])]
        p = power / total
        return values + [float(-np.sum(p*np.log(p))/np.log(len(p)))]

    def _fft_features(x, fs, n_fft=512):
        spectrum = np.abs(np.fft.rfft(x, n=n_fft)).astype(np.float64)
        freqs = np.fft.rfftfreq(n_fft, 1.0/fs)
        order = np.argsort(spectrum[1:])[-3:] + 1
        return [float(freqs[i]) for i in order] + [float(spectrum[i]/(spectrum.max()+1e-12)) for i in order]

    def _stft_features(x, fs, n_fft=512, hop_length=128):
        _, _, z = stft_matrix(x, fs, n_fft, hop_length)
        return [float(z.mean()), float(z.std()), float(np.quantile(z, .25)), float(np.quantile(z, .75))]

    def _mfcc_features(x, fs, n_mfcc=13, n_fft=512, hop_length=128):
        import librosa
        mfcc = librosa.feature.mfcc(y=x, sr=fs, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)
        return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)]).astype(float).tolist()

    def feature_vector(x, fs, mode='hybrid', n_mfcc=13, n_fft=512, hop_length=128):
        x = np.asarray(x, dtype=np.float32)
        base = [float(x.mean()), float(x.std()), float(np.sqrt(np.mean(x**2)+1e-12)), float(skew(x)), float(kurtosis(x))]
        spectral, fft, stft, mfcc = _spectral_features(x, fs, n_fft), _fft_features(x, fs, n_fft), _stft_features(x, fs, n_fft, hop_length), _mfcc_features(x, fs, n_mfcc, n_fft, hop_length)
        values = {'stats': base, 'psd': base+spectral+fft, 'mfcc': base+mfcc, 'stft': base+stft, 'hybrid': base+spectral+fft+stft+mfcc}[mode]
        return np.nan_to_num(np.asarray(values, dtype=np.float32))
    '''),
    code(r'''
    # 5) Models, config và patient-wise feature table
    BASE_CONFIG = {
        'split': {'seed': 42, 'train_size': .70, 'validation_size': .15},
        'dsp': {'target_fs': 1000, 'quantization_bits': 16, 'filter': 'butterworth', 'low_hz': 25, 'high_hz': 400, 'order': 4, 'fir_taps': 129, 'segment_seconds': 3.0, 'segment_hop_seconds': 1.5},
        'features': {'n_mfcc': 13, 'n_fft': 512, 'hop_length': 128},
        'noise': {'enabled': False, 'snr_db': 10, 'kind': 'white'},
        'model': {'kind': 'svm', 'seed': 42},
    }

    def make_model(kind='svm', seed=42):
        if kind == 'svm':
            return Pipeline([('scale', StandardScaler()), ('classifier', SVC(C=2.0, kernel='rbf', probability=True, class_weight='balanced', random_state=seed))])
        if kind == 'mlp':
            return Pipeline([('scale', StandardScaler()), ('classifier', MLPClassifier(hidden_layer_sizes=(128,64), max_iter=300, early_stopping=True, random_state=seed))])
        return RandomForestClassifier(n_estimators=200, class_weight='balanced', random_state=seed, n_jobs=-1)

    def patient_split(table, seed=42):
        patients = table[['patient_id', 'label']].drop_duplicates().reset_index(drop=True)
        train, rest = train_test_split(patients, train_size=.70, stratify=patients['label'], random_state=seed)
        val, test = train_test_split(rest, train_size=.15/.30, stratify=rest['label'], random_state=seed)
        return tuple(x.patient_id.tolist() for x in (train, val, test))

    def patient_features_by_mode(patient, cfg, modes=('psd','mfcc','stft','hybrid')):
        dsp = cfg['dsp']; fs_target = int(dsp['target_fs']); spec = design_filter(fs_target, dsp['filter'], dsp['low_hz'], dsp['high_hz'], dsp['order'], dsp['fir_taps'])
        feature_cfg = {key: value for key, value in cfg['features'].items() if key != 'mode'}
        collected = {mode: [] for mode in modes}
        for rec in patient.recordings:
            source_fs, raw = load_wav(rec.wav_path)
            x = resample_signal(raw, source_fs, fs_target)
            x = quantize(x, int(dsp['quantization_bits']))
            if cfg['noise']['enabled']:
                x = add_noise(x, cfg['noise']['snr_db'], cfg['noise']['kind'], cfg['split']['seed'])
            x = apply_filter(x, spec)
            for segment in segment_signal(x, fs_target, dsp['segment_seconds'], dsp['segment_hop_seconds']):
                for mode in modes:
                    collected[mode].append(feature_vector(segment, fs_target, mode=mode, **feature_cfg))
        return {mode: np.mean(v, axis=0).astype(np.float32) for mode, v in collected.items()}

    def make_tables_for_filter(data_dir, cfg, max_patients=None):
        rows = {mode: [] for mode in ('psd','mfcc','stft','hybrid')}
        n = 0
        patients = list(iter_patients(data_dir))
        if max_patients is not None:
            patients = patients[:max_patients]
        for patient in tqdm(patients, desc=f"features / {cfg['dsp']['filter']} / {cfg['dsp']['target_fs']} Hz"):
            if patient.label == 'Unknown':
                continue
            features = patient_features_by_mode(patient, cfg)
            for mode, vector in features.items():
                rows[mode].append({'patient_id': patient.patient_id, 'label': patient.label, 'features': vector.tolist()})
            n += 1
        return {mode: pd.DataFrame(items) for mode, items in rows.items()}

    def train_evaluate(table, cfg, save_dir=None):
        ids_train, ids_val, ids_test = patient_split(table, cfg['split']['seed'])
        train = table[table.patient_id.isin(ids_train)]; test = table[table.patient_id.isin(ids_test)]
        x_train = np.asarray(train.features.tolist(), dtype=np.float32); x_test = np.asarray(test.features.tolist(), dtype=np.float32)
        y_train, y_test = train.label.to_numpy(), test.label.to_numpy()
        model = make_model(cfg['model']['kind'], cfg['model']['seed']); model.fit(x_train, y_train); pred = model.predict(x_test)
        labels = ['Absent', 'Present']
        result = {'n_train': len(train), 'n_validation': len(ids_val), 'n_test': len(test), 'accuracy': accuracy_score(y_test,pred), 'balanced_accuracy': balanced_accuracy_score(y_test,pred), 'precision_macro': precision_score(y_test,pred,average='macro',zero_division=0), 'recall_macro': recall_score(y_test,pred,average='macro',zero_division=0), 'f1_macro': f1_score(y_test,pred,average='macro',zero_division=0), 'confusion_matrix': confusion_matrix(y_test,pred,labels=labels).tolist()}
        if save_dir is not None:
            save_dir = Path(save_dir); save_dir.mkdir(parents=True, exist_ok=True)
            joblib.dump({'model': model, 'config': cfg, 'patient_ids_train': ids_train, 'patient_ids_test': ids_test}, save_dir/'model.joblib')
            (save_dir/'metrics.json').write_text(json.dumps(result, indent=2, default=float))
        return result, model
    '''),
    code(r'''
    # 6) Audit toàn bộ dữ liệu
    manifest = build_manifest(DATA_DIR, include_unknown=True)
    print(f"Patients: {manifest.patient_id.nunique():,}")
    print(f"Recordings: {len(manifest):,}")
    print(f"Sample rates:")
    sample_rates = []
    for p in tqdm(manifest.wav_path.sample(min(100, len(manifest), random_state=42)), desc='checking sample rates'):
        sample_rates.append(load_wav(p)[0])
    print(pd.Series(sample_rates).value_counts().sort_index().to_dict())
    display(manifest.groupby('label').agg(patients=('patient_id','nunique'), recordings=('recording_id','count')))
    manifest.to_csv(OUTPUT_DIR/'full_manifest.csv', index=False)
    '''),
    code(r'''
    # 7) Baseline: Butterworth + hybrid + SVM
    baseline_cfg = copy.deepcopy(BASE_CONFIG)
    baseline_cfg['model']['kind'] = 'svm'
    baseline_cfg['dsp']['filter'] = 'butterworth'
    baseline_tables = make_tables_for_filter(DATA_DIR, baseline_cfg, MAX_PATIENTS)
    baseline_result, baseline_model = train_evaluate(baseline_tables['hybrid'], baseline_cfg, OUTPUT_DIR/'best_model')
    print(json.dumps(baseline_result, indent=2, default=float))
    '''),
    code(r'''
    # 8) Full 24-run matrix: 3 filters × 4 feature modes × 2 classifiers
    matrix_rows = []
    if RUN_MATRIX:
        for filter_name in ['none', 'butterworth', 'fir']:
            cfg = copy.deepcopy(BASE_CONFIG); cfg['dsp']['filter'] = filter_name
            tables = make_tables_for_filter(DATA_DIR, cfg, MAX_PATIENTS)
            for model_name in ['svm', 'mlp']:
                for feature_name in ['psd', 'mfcc', 'stft', 'hybrid']:
                    cfg_run = copy.deepcopy(cfg); cfg_run['model']['kind'] = model_name
                    result, _ = train_evaluate(tables[feature_name], cfg_run)
                    matrix_rows.append({'model': model_name, 'filter': filter_name, 'features': feature_name, **result})
        matrix_df = pd.DataFrame(matrix_rows).sort_values('f1_macro', ascending=False)
        matrix_df.to_csv(OUTPUT_DIR/'full_matrix.csv', index=False)
        display(matrix_df[['model','filter','features','f1_macro','balanced_accuracy','accuracy']].head(10))
    else:
        matrix_df = pd.DataFrame()
        print('RUN_MATRIX=False — chỉ chạy baseline.')
    '''),
    code(r'''
    # 9) Visualize matrix kết quả
    if not matrix_df.empty:
        heat = matrix_df.pivot_table(index=['model','filter'], columns='features', values='f1_macro')
        plt.figure(figsize=(10, 5))
        sns.heatmap(heat, annot=True, fmt='.3f', cmap='YlGnBu', vmin=0, vmax=1)
        plt.title('DSP ablation matrix — Macro-F1'); plt.tight_layout(); plt.show()
    '''),
    code(r'''
    # 10) Robustness: sampling, quantization và noise
    robustness_rows = []
    if RUN_ROBUSTNESS:
        specs = []
        for fs in [1000, 2000, 4000]: specs.append((f'sampling_{fs}hz', fs, 16, False, 'white', 10))
        for bits in [8, 12, 16]: specs.append((f'quantization_{bits}bit', 1000, bits, False, 'white', 10))
        for kind, snr in [('white',20),('white',10),('pink',10),('impulse',10)]: specs.append((f'noise_{kind}_{snr}db',1000,16,True,kind,snr))
        for name, fs, bits, enabled, kind, snr in specs:
            cfg = copy.deepcopy(BASE_CONFIG); cfg['model']['kind']='svm'; cfg['dsp'].update({'filter':'butterworth','target_fs':fs,'quantization_bits':bits}); cfg['noise']={'enabled':enabled,'kind':kind,'snr_db':snr}
            tables = make_tables_for_filter(DATA_DIR, cfg, MAX_PATIENTS)
            result, _ = train_evaluate(tables['hybrid'], cfg)
            robustness_rows.append({'experiment':name,'target_fs':fs,'quantization_bits':bits,'noise_kind':kind if enabled else 'none','snr_db':snr if enabled else None, **result})
        robustness_df = pd.DataFrame(robustness_rows); robustness_df.to_csv(OUTPUT_DIR/'robustness.csv', index=False); display(robustness_df[['experiment','f1_macro','balanced_accuracy']])
    else:
        robustness_df = pd.DataFrame(); print('RUN_ROBUSTNESS=False')
    '''),
    code(r'''
    # 11) Plot robustness
    if not robustness_df.empty:
        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        for ax, pattern, title in zip(axes, ['sampling_', 'quantization_', 'noise_'], ['Sampling', 'Quantization', 'Noise']):
            part = robustness_df[robustness_df.experiment.str.startswith(pattern)]
            sns.barplot(data=part, x='experiment', y='f1_macro', ax=ax, color='#2B7A78')
            ax.set_title(title); ax.set_ylim(0,1); ax.tick_params(axis='x', rotation=35); ax.set_xlabel(''); ax.set_ylabel('Macro-F1')
        plt.tight_layout(); plt.savefig(OUTPUT_DIR/'robustness.png', dpi=160); plt.show()
    '''),
    code(r'''
    # 12) Signal-level DSP visualization trên một recording
    sample_wav = Path(manifest.iloc[0].wav_path)
    fs_raw, raw = load_wav(sample_wav)
    cfg = copy.deepcopy(BASE_CONFIG)
    processed = apply_filter(quantize(resample_signal(raw, fs_raw, 1000), 16), design_filter(1000, 'butterworth'))
    freqs, times, z = stft_matrix(processed, 1000)
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    axes[0].plot(np.arange(len(raw))/fs_raw, raw, lw=.6, color='#D96445'); axes[0].set_title('Raw PCG waveform')
    axes[1].plot(np.fft.rfftfreq(2048, 1/1000), np.abs(np.fft.rfft(processed, n=2048)), color='#2B7A78'); axes[1].set_title('FFT magnitude'); axes[1].set_xlim(0,500)
    f_psd, p_psd = signal.welch(processed, fs=1000, nperseg=min(512,len(processed))); axes[2].semilogy(f_psd,p_psd,color='#6A5C8A'); axes[2].set_title('Welch PSD'); axes[2].set_xlim(0,500)
    axes[3].pcolormesh(times, freqs, z, shading='auto', cmap='viridis'); axes[3].set_title('Log-STFT spectrogram'); axes[3].set_ylim(0,500); axes[3].set_xlabel('Time (s)')
    plt.tight_layout(); plt.savefig(OUTPUT_DIR/'signal_report.png', dpi=160); plt.show()
    print('Sample:', sample_wav.name, '| source fs:', fs_raw)
    '''),
    code(r'''
    # 13) Lưu model bundle và hàm inference cho WAV mới
    # Nếu matrix đã chạy, chọn run tốt nhất; nếu không thì dùng baseline.
    if not matrix_df.empty:
        best = matrix_df.iloc[0]
        best_cfg = copy.deepcopy(BASE_CONFIG); best_cfg['dsp']['filter'] = best['filter']; best_cfg['features']['mode'] = best['features']; best_cfg['model']['kind'] = best['model']
        best_tables = make_tables_for_filter(DATA_DIR, best_cfg, MAX_PATIENTS)
        best_result, best_model = train_evaluate(best_tables[best['features']], best_cfg, OUTPUT_DIR/'best_model')
        print('Saved best model:', best.to_dict())
    else:
        best_cfg, best_model = baseline_cfg, baseline_model

    def predict_wav(wav_path, model=best_model, cfg=best_cfg):
        patient = PatientRecord('inference', 'Unknown', None, (RecordingRef('input', Path(wav_path), None),))
        vector = patient_features_by_mode(patient, cfg, modes=(cfg['features'].get('mode','hybrid'),))[cfg['features'].get('mode','hybrid')].reshape(1,-1)
        label = model.predict(vector)[0]
        proba = model.predict_proba(vector)[0] if hasattr(model, 'predict_proba') else None
        return {'label': label, 'probabilities': proba}

    # Ví dụ: predict_wav('/content/my_recording.wav')
    print('Outputs saved under:', OUTPUT_DIR)
    '''),
    md(r'''
    ## Diễn giải kết quả

    - `full_matrix.csv`: 24 cấu hình filter × feature × classifier.
    - `robustness.csv`: sampling rate, quantization và noise.
    - `best_model/model.joblib`: model bundle để inference.
    - `signal_report.png`: waveform, FFT, PSD, STFT.

    Khi báo cáo, ghi rõ đây là bài toán research prototype, không phải thiết bị chẩn đoán lâm sàng. Nếu Colab hết thời gian, chạy `FAST_MODE = True` để kiểm tra pipeline rồi chạy lại full cohort theo từng cell.
    '''),
]

nb = {
    "cells": cells,
    "metadata": {
        "colab": {"name": "PCG DSP Full End-to-End", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

out = Path(__file__).resolve().parents[1] / 'notebooks' / 'PCG_DSP_full_end_to_end_colab.ipynb'
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(nb, ensure_ascii=False, indent=1), encoding='utf-8')
print(out.as_posix().encode('ascii', 'backslashreplace').decode())
