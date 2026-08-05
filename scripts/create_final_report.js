const fs = require('fs');
const path = require('path');
const {
  AlignmentType,
  BorderStyle,
  Document,
  Footer,
  HeadingLevel,
  ImageRun,
  LevelFormat,
  PageBreak,
  PageNumber,
  Paragraph,
  ShadingType,
  Table,
  TableCell,
  TableRow,
  TextRun,
  WidthType,
  Header,
  ExternalHyperlink,
  Packer,
} = require('docx');

const root = path.resolve(__dirname, '..');
const artifacts = path.join(root, 'artifacts');
const output = path.join(artifacts, 'final_report_dsp501.docx');

function readImage(name) {
  const file = path.join(artifacts, name);
  if (!fs.existsSync(file)) throw new Error(`Missing report image: ${file}`);
  return fs.readFileSync(file);
}

const colors = {
  navy: '17324D',
  teal: '0F766E',
  orange: 'EA580C',
  light: 'EAF1F5',
  grid: 'CBD5E1',
};

function run(text, options = {}) {
  return new TextRun({ text, font: options.font || 'Aptos', size: options.size || 22, bold: options.bold || false, color: options.color, italics: options.italics || false });
}

function para(text = '', options = {}) {
  const children = Array.isArray(text) ? text : [run(text, options)];
  return new Paragraph({ children, alignment: options.alignment, style: options.style, spacing: options.spacing || { after: 140, line: 276 }, keepNext: options.keepNext });
}

function heading(text, level) {
  return new Paragraph({ text, heading: level, spacing: { before: 260, after: 120 }, keepNext: true });
}

function bullet(text) {
  return new Paragraph({ children: [run(text)], bullet: { level: 0 }, spacing: { after: 70, line: 276 } });
}

function caption(text) {
  return para(text, { alignment: AlignmentType.CENTER, italics: true, size: 19, color: '475569', spacing: { before: 40, after: 180 } });
}

function cell(text, options = {}) {
  const paragraph = new Paragraph({ children: [run(String(text), { size: options.size || 19, bold: options.bold || false, color: options.color })], alignment: options.alignment || AlignmentType.LEFT, spacing: { after: 0, line: 240 } });
  return new TableCell({ children: [paragraph], width: { size: options.width, type: WidthType.DXA }, shading: options.header ? { fill: colors.navy, type: ShadingType.CLEAR } : options.shading ? { fill: options.shading, type: ShadingType.CLEAR } : undefined, margins: { top: 90, bottom: 90, left: 100, right: 100 } });
}

function table(headers, rows, widths) {
  const headerRow = new TableRow({ children: headers.map((value, i) => cell(value, { width: widths[i], header: true, bold: true, color: 'FFFFFF', alignment: AlignmentType.CENTER })) });
  const bodyRows = rows.map((row, r) => new TableRow({ children: row.map((value, i) => cell(value, { width: widths[i], shading: r % 2 ? 'F8FAFC' : undefined, alignment: i === 0 ? AlignmentType.LEFT : AlignmentType.CENTER })) }));
  return new Table({ rows: [headerRow, ...bodyRows], width: { size: widths.reduce((a, b) => a + b, 0), type: WidthType.DXA }, columnWidths: widths, borders: { top: { style: BorderStyle.SINGLE, size: 4, color: colors.grid }, bottom: { style: BorderStyle.SINGLE, size: 4, color: colors.grid }, insideHorizontal: { style: BorderStyle.SINGLE, size: 2, color: colors.grid }, insideVertical: { style: BorderStyle.SINGLE, size: 2, color: colors.grid } } });
}

function image(name, width, height) {
  return new Paragraph({ alignment: AlignmentType.CENTER, children: [new ImageRun({ type: 'png', data: readImage(name), transformation: { width, height } })], spacing: { before: 100, after: 80 } });
}

function externalLink(label, url) {
  return new Paragraph({ children: [new ExternalHyperlink({ link: url, children: [run(label, { color: '0563C1', italics: true })] })], spacing: { after: 120 } });
}

const doc = new Document({
  creator: 'DSP501 project team',
  title: 'Robust PCG Murmur Detection with Digital Signal Processing',
  description: 'DSP501 final project report',
  styles: {
    default: { document: { run: { font: 'Aptos', size: 22, color: '1E293B' }, paragraph: { spacing: { line: 276, after: 140 } } } },
    title: { run: { font: 'Aptos Display', size: 40, bold: true, color: colors.navy }, paragraph: { alignment: AlignmentType.CENTER, spacing: { after: 240 } } },
    subtitle: { run: { font: 'Aptos', size: 26, color: colors.teal }, paragraph: { alignment: AlignmentType.CENTER, spacing: { after: 180 } } },
    heading1: { run: { font: 'Aptos Display', size: 30, bold: true, color: colors.navy }, paragraph: { outlineLevel: 0 } },
    heading2: { run: { font: 'Aptos Display', size: 25, bold: true, color: colors.teal }, paragraph: { outlineLevel: 1 } },
  },
  numbering: { config: [{ reference: 'bullet-list', levels: [{ level: 0, format: LevelFormat.BULLET, text: '•', alignment: AlignmentType.LEFT, style: { paragraph: { indent: { left: 720, hanging: 360 } } } }] }] },
  sections: [{
    properties: { page: { margin: { top: 900, right: 900, bottom: 900, left: 1000 } } },
    headers: { default: new Header({ children: [para('DSP501 · Digital Signal and Image Processing', { alignment: AlignmentType.RIGHT, size: 17, color: '64748B', spacing: { after: 0 } })] }) },
    footers: { default: new Footer({ children: [new Paragraph({ alignment: AlignmentType.CENTER, children: [run('Trang ', { size: 17, color: '64748B' }), new TextRun({ children: [PageNumber.CURRENT], size: 17, color: '64748B' })] })] }) },
    children: [
      new Paragraph({ text: 'ĐỒ ÁN CUỐI KỲ', style: 'title', spacing: { before: 1000, after: 180 } }),
      new Paragraph({ text: 'DIGITAL SIGNAL AND IMAGE PROCESSING (DSP501)', style: 'subtitle' }),
      new Paragraph({ text: 'Robust Phonocardiogram Murmur Detection', style: 'title', spacing: { before: 400, after: 100 } }),
      new Paragraph({ text: 'with Digital Signal Processing', style: 'title', spacing: { after: 500 } }),
      para('Báo cáo nghiên cứu và triển khai hệ thống DSP/ML trên tín hiệu tim PCG', { alignment: AlignmentType.CENTER, size: 23, color: '475569', spacing: { after: 700 } }),
      table(['Thông tin', 'Nội dung'], [['Môn học', 'Digital Signal and Image Processing (DSP501)'], ['Dữ liệu', 'CirCor DigiScope / PhysioNet Challenge 2022'], ['Phạm vi', 'Audit dữ liệu, DSP ablation, robustness và FE demo'], ['Sinh viên', '____________________________'], ['Ngày', '05/08/2026']], [2200, 7100]),
      para('', { spacing: { after: 420 } }),
      para('Lưu ý: hệ thống là research prototype phục vụ mục tiêu học tập và phân tích DSP; không phải thiết bị chẩn đoán lâm sàng.', { alignment: AlignmentType.CENTER, size: 18, italics: true, color: colors.orange }),
      new Paragraph({ children: [new PageBreak()] }),

      heading('Tóm tắt', HeadingLevel.HEADING_1),
      para('Báo cáo trình bày một pipeline xử lý và phân loại tín hiệu phonocardiogram (PCG) nhằm nghiên cứu ảnh hưởng của các khối DSP đến bài toán phát hiện heart murmur. Hệ thống thực hiện đọc WAV, chuyển mono, chuẩn hóa, resampling, quantization, lọc dải thông, chia đoạn và trích xuất các biểu diễn thời gian–tần số gồm thống kê thời gian, Welch PSD, FFT, STFT và MFCC. Các vector đặc trưng được gộp theo bệnh nhân để tránh rò rỉ dữ liệu giữa các tập train/test.'),
      para('Trên subset tải trực tiếp gồm 100 bệnh nhân và 325 bản ghi hợp lệ, nghiên cứu chạy matrix 24 cấu hình (3 loại filter × 4 nhóm feature × SVM/MLP) và 10 thí nghiệm robustness cho sampling rate, quantization và noise. Cấu hình tốt nhất là SVM với Butterworth hoặc FIR và hybrid features, đạt accuracy 0.933, balanced accuracy 0.950 và macro-F1 0.928 trên tập test 15 bệnh nhân. Kết quả cho thấy quantization 8-bit và pink/impulse noise làm suy giảm hiệu năng rõ hơn thay đổi sampling rate trong phạm vi 1–4 kHz.'),
      para('Từ khóa: phonocardiogram, PCG, heart murmur, sampling, quantization, band-pass filter, FFT, PSD, STFT, SVM.'),

      heading('Mục lục', HeadingLevel.HEADING_1),
      para('1. Giới thiệu — 3'),
      para('2. Dữ liệu và thiết kế thí nghiệm — 3'),
      para('3. Cơ sở DSP và pipeline — 4'),
      para('4. Matrix thực nghiệm — 4'),
      para('5. Robustness theo sampling, quantization và noise — 5'),
      para('6. Signal-level analysis và FE demo — 6'),
      para('7. Đánh giá và hạn chế — 7'),
      para('8. Kết luận và hướng phát triển — 7'),
      para('Tài liệu tham khảo — 8'),
      new Paragraph({ children: [new PageBreak()] }),

      heading('1. Giới thiệu', HeadingLevel.HEADING_1),
      heading('1.1. Bối cảnh bài toán', HeadingLevel.HEADING_2),
      para('Phonocardiogram là tín hiệu âm thanh thu được khi nghe tim bằng cảm biến hoặc ống nghe điện tử. Murmur có thể biểu hiện dưới dạng năng lượng bất thường hoặc thay đổi cấu trúc phổ trong chu kỳ tim. Do đó, PCG là một bài toán phù hợp để minh họa các khái niệm DSP: lấy mẫu, lượng tử hóa, lọc, phân tích phổ, biểu diễn thời gian–tần số và đánh giá độ bền với nhiễu.'),
      heading('1.2. Câu hỏi nghiên cứu', HeadingLevel.HEADING_2),
      bullet('Front-end DSP nào đạt cân bằng tốt giữa hiệu năng và độ phức tạp cho phân loại Absent/Present?'),
      bullet('Sampling rate và quantization ảnh hưởng như thế nào đến đặc trưng PCG?'),
      bullet('Các dạng nhiễu white, pink và impulse làm thay đổi macro-F1 ra sao?'),
      bullet('Patient-wise split có thể giữ nguyên tính tái lập và tránh leakage như thế nào?'),
      heading('1.3. Đóng góp của đồ án', HeadingLevel.HEADING_2),
      bullet('Xây dựng pipeline DSP có cấu hình hóa, tái lập và dùng chung cho training, CLI inference và Streamlit FE.'),
      bullet('So sánh định lượng filter, feature representation và classifier bằng macro-F1, balanced accuracy và confusion matrix.'),
      bullet('Bổ sung robustness matrix cho sampling, bit depth và noise.'),
      bullet('Tạo demo trực quan cho waveform, FFT, PSD, STFT và xác suất phân loại.'),

      heading('2. Dữ liệu và thiết kế thí nghiệm', HeadingLevel.HEADING_1),
      heading('2.1. Dataset', HeadingLevel.HEADING_2),
      para('Nguồn dữ liệu là CirCor DigiScope được công bố trên PhysioNet. Mỗi bệnh nhân có file metadata .txt, một hoặc nhiều recording WAV tại các vị trí nghe tim, nhãn murmur và tùy chọn file segmentation. Trong triển khai này, metadata được parse theo patient_id; các recording cùng bệnh nhân luôn được giữ chung một partition.'),
      externalLink('PhysioNet CirCor Heart Sound Dataset', 'https://physionet.org/content/circor-heart-sound/1.0.3/'),
      table(['Đặc tính', 'Giá trị'], [['Bệnh nhân tải được', '100'], ['Recording WAV hợp lệ', '325'], ['Absent / Present / Unknown recordings', '199 / 109 / 17'], ['Bệnh nhân Absent / Present / Unknown', '60 / 32 / 8'], ['Sample rate nguồn', '4 kHz (325/325)'], ['Thời lượng', '6.592–46.496 s'], ['Invalid/missing', '0']], [4300, 5000]),
      caption('Bảng 1. Thống kê audit subset 100 bệnh nhân.'),
      para('Archive ZIP toàn bộ bị throttled trong môi trường thực thi, nên nghiên cứu sử dụng patient-wise direct-download subset. Đây là giới hạn về phạm vi dữ liệu và được giữ rõ trong phần kết luận; các metric không được trình bày như kết quả trên toàn bộ cohort.'),
      heading('2.2. Chia dữ liệu', HeadingLevel.HEADING_2),
      para('Sau khi loại Unknown khỏi supervised learning, 92 bệnh nhân có nhãn được chia stratified theo patient_id với seed 42: 64 train, 13 validation và 15 test. Các location AV/MV/PV/TV của cùng bệnh nhân không bao giờ bị tách sang partition khác.'),

      heading('3. Cơ sở DSP và pipeline', HeadingLevel.HEADING_1),
      heading('3.1. Chuỗi xử lý', HeadingLevel.HEADING_2),
      para('Với mỗi WAV, hệ thống thực hiện: (1) đọc PCM và chuyển mono; (2) resample về target_fs; (3) lượng tử hóa về bit depth được chọn; (4) thêm noise có kiểm soát nếu bật robustness test; (5) wavelet denoise tùy chọn; (6) band-pass filter 25–400 Hz; (7) chia cửa sổ 3 giây, hop 1.5 giây; (8) trích xuất feature; (9) trung bình theo window và recording để tạo một vector patient-level.'),
      heading('3.2. Feature representation', HeadingLevel.HEADING_2),
      table(['Nhóm', 'Thành phần'], [['Stats', 'Mean, standard deviation, RMS, skewness, kurtosis'], ['PSD', 'Welch band-power ratios và spectral entropy'], ['FFT', 'Ba peak frequency và magnitude chuẩn hóa'], ['STFT', 'Mean, std và quantiles của log magnitude spectrogram'], ['MFCC', 'Mean/std của 13 MFCC'], ['Hybrid', 'Kết hợp toàn bộ nhóm trên']], [2100, 7200]),
      caption('Bảng 2. Các biểu diễn được dùng trong matrix.'),
      heading('3.3. Classifier', HeadingLevel.HEADING_2),
      para('Hai baseline được so sánh: SVM RBF với StandardScaler, probability=True và class_weight=balanced; MLPClassifier với hai hidden layers (128, 64), early stopping và seed cố định. SVM được chọn làm model demo vì ổn định hơn MLP trên subset nhỏ và có xác suất lớp để hiển thị trong FE.'),

      heading('4. Matrix thực nghiệm', HeadingLevel.HEADING_1),
      para('Matrix chính gồm 24 cấu hình: filter ∈ {none, Butterworth, FIR}, feature ∈ {PSD, MFCC, STFT, hybrid}, model ∈ {SVM, MLP}. Metric chính là macro-F1 vì hai lớp không cân bằng hoàn toàn; balanced accuracy, precision, recall và confusion matrix được lưu kèm.'),
      image('circor_100_f1.png', 700, 390),
      caption('Hình 1. So sánh macro-F1 giữa các front-end DSP và classifier.'),
      table(['Cấu hình', 'Accuracy', 'Balanced acc.', 'Macro-F1'], [['SVM + Butterworth + hybrid', '0.933', '0.950', '0.928'], ['SVM + FIR + hybrid', '0.933', '0.950', '0.928'], ['SVM + FIR + PSD', '0.933', '0.900', '0.921'], ['SVM + no filter + PSD', '0.933', '0.900', '0.921'], ['MLP + no filter + hybrid', '0.867', '0.800', '0.830']], [4600, 1500, 1600, 1600]),
      caption('Bảng 3. Các cấu hình có macro-F1 cao nhất trong 24-run matrix.'),
      para('Hybrid không luôn vượt PSD ở mọi classifier, nhưng SVM hybrid sau Butterworth/FIR đạt balanced accuracy và macro-F1 cao nhất. MLP có thể đạt kết quả khá với hybrid không lọc nhưng dao động lớn hơn giữa các front-end, phù hợp với nhận định rằng subset patient-level còn nhỏ.'),

      heading('5. Robustness theo sampling, quantization và noise', HeadingLevel.HEADING_1),
      para('Để tách ảnh hưởng của từng yếu tố DSP, hệ thống cố định SVM + Butterworth + hybrid và chạy thêm 10 điều kiện. Mỗi điều kiện dùng cùng patient split, seed và metric với matrix chính.'),
      image('circor_100_robustness.png', 720, 205),
      caption('Hình 2. Macro-F1 theo sampling rate, quantization và noise.'),
      table(['Điều kiện', 'Macro-F1'], [['Sampling 1 / 2 / 4 kHz', '0.928 / 0.928 / 0.928'], ['Quantization 8 / 12 / 16 bit', '0.732 / 0.928 / 0.928'], ['White noise, 20 dB', '0.850'], ['White noise, 10 dB', '0.921'], ['Pink noise, 10 dB', '0.796'], ['Impulse noise, 10 dB', '0.796']], [6500, 2800]),
      caption('Bảng 4. Robustness matrix.'),
      para('Trong subset này, giảm sampling từ 4 kHz xuống 1 kHz không làm thay đổi metric của cấu hình đã chọn, cho thấy dải 25–400 Hz vẫn được bảo toàn. Ngược lại, 8-bit quantization làm giảm macro-F1 rõ rệt. Pink và impulse noise gây suy giảm nhiều hơn white noise ở cùng mức SNR, gợi ý rằng nhiễu có cấu trúc hoặc nhiễu xung ảnh hưởng mạnh đến phổ và transient của PCG.'),

      heading('6. Signal-level analysis và FE demo', HeadingLevel.HEADING_1),
      heading('6.1. Signal report', HeadingLevel.HEADING_2),
      para('Signal report minh họa một recording qua raw waveform, processed waveform, FFT, Welch PSD và log-STFT. Các biểu đồ giúp người xem liên hệ trực tiếp giữa thao tác DSP và output model thay vì chỉ xem một con số accuracy.'),
      image('circor_100_signal_report/signal_report.png', 720, 410),
      caption('Hình 3. Ví dụ signal-level report trên một WAV CirCor thật.'),
      heading('6.2. Streamlit demo', HeadingLevel.HEADING_2),
      para('FE nằm trong app.py và gọi cùng service inference với CLI. Người dùng upload WAV, chọn sampling rate, quantization, filter và noise preview; sau đó xem audio player, waveform, FFT, PSD, STFT, class probabilities và cấu hình DSP được dùng. Model mặc định là svm_butterworth_hybrid/model.joblib.'),
      para('Lệnh chạy demo:'),
      para('.\\.venv\\Scripts\\python.exe -m streamlit run app.py', { size: 20, color: colors.navy }),
      para('Khi trình bày, demo nên đi theo flow: upload một WAV → giải thích raw/filtered waveform → chỉ ra năng lượng trên FFT/PSD → xem STFT → đổi 16-bit sang 8-bit hoặc bật pink noise → phân tích thay đổi probability. Đây là minh họa robustness, không phải phép chẩn đoán.'),

      heading('7. Đánh giá và hạn chế', HeadingLevel.HEADING_1),
      bullet('Test set chỉ có 15 bệnh nhân; khoảng tin cậy và độ ổn định theo seed chưa thể đại diện cho toàn cohort.'),
      bullet('Subset 100 bệnh nhân không phải toàn bộ CirCor archive; cần tải đủ dataset và rerun toàn bộ matrix nếu yêu cầu kết quả chính thức.'),
      bullet('Segmentation annotations được parse và lưu trong manifest nhưng pipeline hiện tại chưa dùng để cắt riêng S1, systole, S2 và diastole.'),
      bullet('Kết quả là proof-of-concept cho môn DSP, không được diễn giải thành hiệu năng chẩn đoán y khoa.'),
      bullet('MLP và SVM là baseline CPU; nghiên cứu chưa đánh giá CNN hoặc calibration trên cohort lớn.'),

      heading('8. Kết luận và hướng phát triển', HeadingLevel.HEADING_1),
      para('Đồ án đã xây dựng và kiểm chứng một pipeline PCG end-to-end có thể tái lập từ dữ liệu WAV tới prediction và FE visualization. Matrix 24-run cho thấy SVM kết hợp hybrid features sau Butterworth hoặc FIR là lựa chọn tốt nhất trong subset hiện tại, với macro-F1 0.928. Robustness matrix cho thấy hệ thống ít nhạy với sampling 1–4 kHz nhưng nhạy hơn với quantization 8-bit và noise có cấu trúc/xung.'),
      para('Các bước phát triển tiếp theo gồm: tải toàn bộ CirCor archive, lặp lại đánh giá với nhiều patient-wise seeds, sử dụng segmentation để tạo cycle-level feature, bổ sung confidence interval/calibration, và chỉ khi cần mới so sánh với spectrogram CNN.'),

      heading('Tài liệu tham khảo', HeadingLevel.HEADING_1),
      para('[1] PhysioNet, “CirCor DigiScope Dataset,” version 1.0.3. '),
      externalLink('https://physionet.org/content/circor-heart-sound/1.0.3/', 'https://physionet.org/content/circor-heart-sound/1.0.3/'),
      para('[2] SciPy Signal Processing documentation: resampling, filtering, Welch PSD and STFT.'),
      para('[3] Scikit-learn documentation: SVC, MLPClassifier, patient-wise evaluation utilities.'),
      para('[4] PyWavelets documentation: discrete wavelet transform and soft-threshold denoising.'),
    ],
  }],
});

Packer.toBuffer(doc).then((buffer) => {
  fs.mkdirSync(artifacts, { recursive: true });
  fs.writeFileSync(output, buffer);
  console.log(`Wrote ${output}`);
});
