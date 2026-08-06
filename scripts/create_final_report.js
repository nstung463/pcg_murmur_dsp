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
      para('Bản chính sử dụng toàn bộ public training release: 942 bệnh nhân và 3.163 WAV, trong đó 874 bệnh nhân có nhãn Absent/Present được dùng cho supervised learning. Patient-wise split gồm 611 train, 131 validation và 132 test. Full matrix 24 cấu hình cho thấy MLP + không filter + MFCC đạt macro-F1 cao nhất 0,693, còn SVM + không filter + hybrid có balanced accuracy cao nhất 0,664 trong cùng split. Một MobileNetV3-Small pretrained frozen được thêm như baseline phụ trên cùng split; FIR đạt accuracy 0,864, balanced accuracy 0,777 và macro-F1 0,784. Benchmark preprocess cố định SVM + Butterworth + hybrid cho thấy 1 kHz + 16-bit là baseline nhanh nhất (85,8 s; macro-F1 0,615), còn 4 kHz + không quantization thêm đạt macro-F1 0,637 nhưng tốn 126,9 s. Matrix và robustness trên subset 100 patient được giữ như pilot để minh họa xu hướng khác biệt với kết quả full cohort.'),
      para('Từ khóa: phonocardiogram, PCG, heart murmur, sampling, quantization, band-pass filter, FFT, PSD, STFT, SVM.'),

      heading('Mục lục', HeadingLevel.HEADING_1),
      para('1. Giới thiệu — 3'),
      para('2. Dữ liệu và thiết kế thí nghiệm — 3'),
      para('3. Cơ sở DSP và pipeline — 4'),
      para('4. Matrix thực nghiệm — 4'),
      para('4.1. Full-cohort benchmark sampling/quantization — 5'),
      para('4.2. Full-data matrix model/filter/feature — 6'),
      para('4.3. Pretrained MobileNet baseline — 7'),
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
      table(['Đặc tính', 'Giá trị'], [['Bệnh nhân trong metadata', '942'], ['Recording WAV hợp lệ', '3.163'], ['Absent / Present / Unknown patients', '695 / 179 / 68'], ['Bệnh nhân dùng supervised', '874 (loại 68 Unknown)'], ['Sample rate nguồn', '4 kHz (3.163/3.163)'], ['Split patient-wise', '611 / 131 / 132'], ['Invalid/missing', '0']], [4300, 5000]),
      caption('Bảng 1. Audit toàn bộ public training release CirCor 1.0.3.'),
      para('Đây là toàn bộ public training release tải được, không phải toàn bộ 1.568 subjects của mô tả dataset ban đầu. Nhãn Unknown được giữ trong audit nhưng loại khỏi supervised learning; các đường ghi âm còn lại đều có sample rate 4 kHz và đọc được thành công.'),
      heading('2.2. Chia dữ liệu', HeadingLevel.HEADING_2),
      para('Sau khi loại Unknown khỏi supervised learning, 874 bệnh nhân có nhãn được chia stratified theo patient_id với seed 42: 611 train, 131 validation và 132 test. Các location AV/MV/PV/TV của cùng bệnh nhân không bao giờ bị tách sang partition khác.'),

      heading('3. Cơ sở DSP và pipeline', HeadingLevel.HEADING_1),
      heading('3.1. Chuỗi xử lý', HeadingLevel.HEADING_2),
      para('Với mỗi WAV, hệ thống thực hiện: (1) đọc PCM và chuyển mono; (2) resample về target_fs; (3) lượng tử hóa về bit depth được chọn; (4) thêm noise có kiểm soát nếu bật robustness test; (5) wavelet denoise tùy chọn; (6) band-pass filter 25–400 Hz; (7) chia cửa sổ 3 giây, hop 1.5 giây; (8) trích xuất feature; (9) trung bình theo window và recording để tạo một vector patient-level.'),
      heading('3.2. Feature representation', HeadingLevel.HEADING_2),
      table(['Nhóm', 'Thành phần'], [['Stats', 'Mean, standard deviation, RMS, skewness, kurtosis'], ['PSD', 'Welch band-power ratios và spectral entropy'], ['FFT', 'Ba peak frequency và magnitude chuẩn hóa'], ['STFT', 'Mean, std và quantiles của log magnitude spectrogram'], ['MFCC', 'Mean/std của 13 MFCC'], ['Hybrid', 'Kết hợp toàn bộ nhóm trên']], [2100, 7200]),
      caption('Bảng 2. Các biểu diễn được dùng trong matrix.'),
      heading('3.3. Classifier', HeadingLevel.HEADING_2),
      para('Hai baseline cổ điển được so sánh: SVM RBF với StandardScaler, probability=True và class_weight=balanced; MLPClassifier với hai hidden layers (128, 64), early stopping và seed cố định. Ngoài ra, MobileNetV3-Small pretrained với backbone frozen được chạy như baseline spectrogram nhẹ và hiện có thể chọn trực tiếp trong FE khi checkpoint tồn tại.'),

      heading('4. Matrix thực nghiệm', HeadingLevel.HEADING_1),
      para('Matrix chính gồm 24 cấu hình: filter ∈ {none, Butterworth, FIR}, feature ∈ {PSD, MFCC, STFT, hybrid}, model ∈ {SVM, MLP}. Metric chính là macro-F1 vì hai lớp không cân bằng hoàn toàn; balanced accuracy, precision, recall và confusion matrix được lưu kèm.'),
      image('circor_100_f1.png', 700, 390),
      caption('Hình 1. So sánh macro-F1 giữa các front-end DSP và classifier.'),
      table(['Cấu hình', 'Accuracy', 'Balanced acc.', 'Macro-F1'], [['SVM + Butterworth + hybrid', '0.933', '0.950', '0.928'], ['SVM + FIR + hybrid', '0.933', '0.950', '0.928'], ['SVM + FIR + PSD', '0.933', '0.900', '0.921'], ['SVM + no filter + PSD', '0.933', '0.900', '0.921'], ['MLP + no filter + hybrid', '0.867', '0.800', '0.830']], [4600, 1500, 1600, 1600]),
      caption('Bảng 3. Các cấu hình có macro-F1 cao nhất trong 24-run matrix.'),
      para('Hybrid không luôn vượt PSD ở mọi classifier, nhưng SVM hybrid sau Butterworth/FIR đạt balanced accuracy và macro-F1 cao nhất. MLP có thể đạt kết quả khá với hybrid không lọc nhưng dao động lớn hơn giữa các front-end, phù hợp với nhận định rằng subset patient-level còn nhỏ.'),

      new Paragraph({ children: [new PageBreak()] }),
      heading('4.1. Full-cohort benchmark: sampling và quantization', HeadingLevel.HEADING_2),
      para('Để trả lời trực tiếp câu hỏi có cần hạ sample rate hoặc quantize lại hay không, benchmark đầy đủ giữ nguyên SVM + Butterworth + hybrid, cùng seed 42 và cùng patient-wise split, chỉ thay target sampling rate và bit depth. Mỗi thời gian dưới đây là thời gian feature extraction + train trên CPU với 4 workers.'),
      table(['Cấu hình', 'Thời gian', 'So với 1 kHz', 'Accuracy', 'Macro-F1'], [['1 kHz + 16-bit', '85,8 s', '1,00×', '0,742', '0,615'], ['2 kHz + 16-bit', '98,4 s', '1,15×', '0,758', '0,628'], ['4 kHz + 16-bit', '122,8 s', '1,43×', '0,773', '0,630'], ['4 kHz + none', '126,9 s', '1,48×', '0,780', '0,637'], ['4 kHz + 8-bit', '128,7 s', '1,50×', '0,765', '0,634'], ['4 kHz + 12-bit', '126,3 s', '1,47×', '0,773', '0,630']], [3000, 1400, 1500, 1400, 1400]),
      caption('Bảng 4. Benchmark toàn bộ 874 bệnh nhân có nhãn; test = 132 bệnh nhân.'),
      para('Kết luận thực nghiệm: 4 kHz giữ thêm thông tin nhưng tăng thời gian khoảng 43–48% so với 1 kHz; lợi ích Macro-F1 chỉ tăng nhẹ (0,615 → 0,630–0,637) trên một split. Bỏ quantization thêm đạt điểm cao nhất trong benchmark này, nhưng chênh lệch nhỏ và chưa đủ để khẳng định ưu thế thống kê. Vì vậy FE cho phép chọn 1 kHz + 16-bit để demo nhanh hoặc 4 kHz + none để tái hiện tín hiệu gốc.'),

      new Paragraph({ children: [new PageBreak()] }),
      heading('4.2. Full-data matrix: model, filter và feature', HeadingLevel.HEADING_2),
      para('Matrix đầy đủ giữ target fs = 1 kHz, quantization = 16-bit, seed 42 và patient-wise split; chỉ thay 3 loại filter × 4 nhóm feature × 2 classifier. Kết quả dưới đây là các cấu hình đứng đầu theo Macro-F1; toàn bộ 24 dòng được lưu trong artifacts/full_matrix_24/full_matrix.csv.'),
      table(['Model', 'Filter', 'Feature', 'Accuracy', 'Balanced acc.', 'Macro-F1'], [['MLP', 'None', 'MFCC', '0,848', '0,657', '0,693'], ['MLP', 'None', 'Hybrid', '0,841', '0,625', '0,654'], ['SVM', 'None', 'Hybrid', '0,750', '0,664', '0,648'], ['SVM', 'FIR', 'Hybrid', '0,765', '0,646', '0,644'], ['SVM', 'None', 'MFCC', '0,742', '0,659', '0,642'], ['SVM', 'FIR', 'MFCC', '0,758', '0,628', '0,628'], ['SVM', 'Butterworth', 'Hybrid', '0,742', '0,618', '0,615']], [1500, 1700, 1900, 1300, 1700, 1300]),
      caption('Bảng 5. Top full-data configurations theo Macro-F1; test = 132 bệnh nhân.'),
      para('Không filter cho kết quả tốt hơn trong split này; MFCC giúp MLP đạt Macro-F1 cao nhất, trong khi SVM + none + hybrid có balanced accuracy cao nhất. Tuy nhiên confusion matrix của MLP + none + MFCC vẫn cho thấy lớp Present khó nhận diện (9/27 đúng), vì vậy không nên chọn model chỉ theo Accuracy. Đây là kết luận trên một seed, cần lặp nhiều seed trước khi khẳng định ưu thế.'),

      heading('4.3. Pretrained MobileNet baseline', HeadingLevel.HEADING_2),
      para('MobileNetV3-Small dùng ImageNet pretrained weights, backbone frozen, input là log-STFT 128×128 từ tối đa hai cửa sổ 3 giây mỗi recording và mean aggregation về patient-level. Thí nghiệm vẫn là bài toán binary Absent/Present, giữ target fs = 1 kHz, quantization = 16-bit, seed 42 và split 611/131/132; chỉ thay filter để so sánh công bằng với DSP matrix.'),
      table(['Filter', 'Accuracy', 'Balanced acc.', 'Macro-F1'], [['None', '0,841', '0,749', '0,752'], ['Butterworth', '0,826', '0,698', '0,712'], ['FIR', '0,864', '0,777', '0,784']], [3000, 2000, 2200, 1800]),
      caption('Bảng 6. MobileNetV3-Small pretrained frozen, binary test = 132 bệnh nhân.'),
      para('FIR là cấu hình MobileNet tốt nhất trong binary benchmark theo cả Accuracy, Balanced Accuracy và Macro-F1. Đây là kết quả bổ sung để đối chiếu representation spectrogram với feature DSP thủ công; không dùng CNN thay thế cho câu hỏi chính về filter/feature của DSP.'),

      heading('5. Robustness theo sampling, quantization và noise', HeadingLevel.HEADING_1),
      para('Để tách ảnh hưởng của từng yếu tố DSP, hệ thống cố định SVM + Butterworth + hybrid và chạy thêm 10 điều kiện trên pilot subset 100 bệnh nhân. Mỗi điều kiện dùng cùng patient split, seed và metric với matrix chính; các số liệu ở mục này chỉ dùng để minh họa xu hướng robustness, còn benchmark toàn cohort nằm ở mục 4.1.'),
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
      para('FE nằm trong app.py và gọi cùng service inference với CLI. Người dùng upload WAV hoặc chọn bundled demo recording, chọn model preset SVM/MLP hoặc MobileNetV3-Small, chọn sampling rate, quantization, filter và noise preview; sau đó xem audio player, waveform, FFT, PSD, filter frequency response, STFT, class probabilities, confidence và cấu hình DSP được dùng. Với project recordings, Patient-level (benchmark) là mặc định và aggregate toàn bộ AV/MV/PV/TV trước prediction; external upload dùng recording-level. Demo có A/B comparison giữa training baseline và selected DSP, đồng thời cho phép tải analysis JSON. Preset SVM + no filter + hybrid là mặc định DSP; MobileNetV3-Small + FIR là preset benchmark tùy chọn.'),
      para('Lệnh chạy demo:'),
      para('.\\.venv\\Scripts\\python.exe -m streamlit run app.py', { size: 20, color: colors.navy }),
      para('Khi trình bày, demo nên đi theo flow: bấm Use demo recording hoặc upload WAV → chạy training baseline → đổi target fs/bit depth/filter → chạy A/B comparison → nghe các stage → đọc waveform, FFT/PSD và filter response với dải 25–400 Hz → xem STFT và confidence → tải analysis JSON. Đây là minh họa robustness, không phải phép chẩn đoán.'),

      heading('7. Đánh giá và hạn chế', HeadingLevel.HEADING_1),
      bullet('Benchmark full-cohort có test set 132 bệnh nhân nhưng mới dùng một patient-wise split seed 42; khoảng tin cậy và độ ổn định theo nhiều seed chưa được tính.'),
      bullet('Public training release có 942 bệnh nhân; đây chưa phải toàn bộ 1.568 subjects của dataset ban đầu.'),
      bullet('Segmentation annotations được parse và lưu trong manifest nhưng pipeline hiện tại chưa dùng để cắt riêng S1, systole, S2 và diastole.'),
      bullet('Kết quả là proof-of-concept cho môn DSP, không được diễn giải thành hiệu năng chẩn đoán y khoa.'),
      bullet('MobileNet là baseline transfer-learning frozen và đã được nối vào FE như preset spectrogram tùy chọn; chưa fine-tune toàn backbone, calibration xác suất hoặc đánh giá nhiều seed trên CNN.'),

      heading('8. Kết luận và hướng phát triển', HeadingLevel.HEADING_1),
      para('Đồ án đã xây dựng và kiểm chứng một pipeline PCG end-to-end có thể tái lập từ dữ liệu WAV tới prediction và FE visualization. Trên full public training release, full matrix cho thấy MLP + không filter + MFCC đạt Macro-F1 0,693, còn SVM + không filter + hybrid đạt balanced accuracy 0,664 trong split hiện tại. MobileNetV3-Small pretrained frozen đạt Macro-F1 0,784 với FIR trên cùng binary split và đã được tích hợp như preset FE tùy chọn. Preprocess benchmark cho thấy 1 kHz + 16-bit là lựa chọn nhanh nhất; 4 kHz + không quantization thêm đạt Macro-F1 0,637 với chi phí thời gian khoảng 1,48×. Các kết quả cần được kiểm chứng thêm bằng nhiều seed vì lớp Present vẫn khó nhận diện.'),
      para('Các bước phát triển tiếp theo gồm: lặp lại đánh giá với nhiều patient-wise seeds, sử dụng segmentation để tạo cycle-level feature, bổ sung confidence interval/calibration, và nếu cần thì fine-tune hoặc mở rộng MobileNet trên GPU và các subjects ngoài public training release.'),

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
