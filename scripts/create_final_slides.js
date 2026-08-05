const fs = require('fs');
const path = require('path');
const pptxgen = require('pptxgenjs');

const root = path.resolve(__dirname, '..');
const artifacts = path.join(root, 'artifacts');
const output = path.join(artifacts, 'DSP501_final_presentation.pptx');

const pptx = new pptxgen();
pptx.layout = 'LAYOUT_WIDE';
pptx.author = 'DSP501 project team';
pptx.subject = 'Robust PCG murmur detection with DSP';
pptx.title = 'DSP501 Final Project';
pptx.company = 'DSP501';
pptx.lang = 'vi-VN';
pptx.theme = {
  headFontFace: 'Georgia',
  bodyFontFace: 'Aptos',
  lang: 'vi-VN',
};
pptx.defineLayout({ name: 'CUSTOM_WIDE', width: 13.333, height: 7.5 });
pptx.layout = 'CUSTOM_WIDE';

const C = {
  navy: '17313B',
  navy2: '1D3D49',
  teal: '2B7A78',
  mint: '73BFB8',
  orange: 'D96445',
  yellow: 'D9A441',
  ink: '17313B',
  muted: '6D7775',
  light: 'F6F2EA',
  pale: 'E8E3D7',
  line: 'C8C1B5',
  white: 'FFFDF8',
  purple: '6A5C8A',
};

const W = 13.333;
const H = 7.5;

function addText(slide, text, x, y, w, h, options = {}) {
  slide.addText(text, {
    x, y, w, h,
    fontFace: options.fontFace || 'Aptos',
    fontSize: options.fontSize || 18,
    color: options.color || C.ink,
    bold: options.bold || false,
    italic: options.italic || false,
    align: options.align || 'left',
    valign: options.valign || 'top',
    margin: options.margin === undefined ? 0 : options.margin,
    breakLine: options.breakLine,
    fit: 'shrink',
    paraSpaceAfterPt: options.paraSpaceAfterPt || 0,
    bullet: options.bullet,
  });
}

function addRichLines(slide, lines, x, y, w, h, options = {}) {
  const runs = [];
  lines.forEach((line, i) => {
    runs.push({ text: line, options: { bullet: options.bullet ? { indent: options.indent || 18 } : undefined, breakLine: i < lines.length - 1, hanging: options.hanging || 0 } });
  });
  slide.addText(runs, { x, y, w, h, fontFace: 'Aptos', fontSize: options.fontSize || 18, color: options.color || C.ink, margin: 0, breakLine: false, fit: 'shrink', paraSpaceAfterPt: options.paraSpaceAfterPt || 7, valign: 'top' });
}

function addTitle(slide, title, kicker = '') {
  slide.background = { color: C.light };
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: W, h: 0.08, fill: { color: C.orange }, line: { color: C.orange } });
  if (kicker) addText(slide, kicker.toUpperCase(), 0.68, 0.28, 6.0, 0.22, { fontSize: 10, bold: true, color: C.teal, charSpacing: 1.2 });
  addText(slide, 'DSP501 / FIELD REPORT', 10.35, 0.28, 2.35, 0.22, { fontSize: 9, bold: true, color: C.muted, charSpacing: 1.1, align: 'right' });
  addText(slide, title, 0.68, 0.55, 12.0, 0.58, { fontFace: 'Georgia', fontSize: 30, bold: true, color: C.navy });
}

function addFooter(slide, page) {
  slide.addShape(pptx.ShapeType.line, { x: 0.68, y: 7.02, w: 11.98, h: 0, line: { color: C.line, width: 0.7 } });
  addText(slide, `DSP501 · PCG Murmur Detection`, 0.58, 7.18, 5, 0.18, { fontSize: 9, color: C.muted });
  addText(slide, String(page), 12.35, 7.18, 0.35, 0.18, { fontSize: 9, color: C.muted, align: 'right' });
}

function card(slide, x, y, w, h, title, body, color = C.teal, value = '') {
  slide.addShape(pptx.ShapeType.rect, { x, y, w, h, fill: { color: C.white }, line: { color: C.line, width: 0.8 } });
  slide.addShape(pptx.ShapeType.rect, { x, y, w, h: 0.08, fill: { color }, line: { color } });
  if (value) addText(slide, value, x + 0.25, y + 0.18, w - 0.4, 0.46, { fontSize: 27, bold: true, color });
  addText(slide, title, x + 0.25, y + (value ? 0.72 : 0.22), w - 0.4, 0.28, { fontFace: 'Georgia', fontSize: 15, bold: true, color: C.navy });
  addText(slide, body, x + 0.25, y + (value ? 1.04 : 0.58), w - 0.4, h - (value ? 1.15 : 0.7), { fontSize: 13, color: C.muted });
}

function addImage(slide, file, x, y, w, h, transparency = 0) {
  const p = path.join(artifacts, file);
  if (!fs.existsSync(p)) throw new Error(`Missing image: ${p}`);
  slide.addImage({ path: p, x, y, w, h, transparency });
}

function addBullets(slide, lines, x, y, w, h, fontSize = 18, color = C.ink) {
  const runs = [];
  lines.forEach((line, i) => runs.push({ text: line, options: { bullet: { indent: fontSize }, breakLine: i < lines.length - 1 } }));
  slide.addText(runs, { x, y, w, h, fontFace: 'Aptos', fontSize, color, margin: 0, fit: 'shrink', paraSpaceAfterPt: 11, valign: 'top' });
}

function addMetric(slide, x, y, value, label, color = C.teal) {
  addText(slide, value, x, y, 2.2, 0.55, { fontFace: 'Georgia', fontSize: 31, bold: true, color });
  addText(slide, label, x, y + 0.62, 2.2, 0.3, { fontSize: 12, color: C.muted });
}

// 1. Title
{
  const slide = pptx.addSlide();
  slide.background = { color: C.navy2 };
  slide.addShape(pptx.ShapeType.rect, { x: 0, y: 0, w: 0.14, h: H, fill: { color: C.orange }, line: { color: C.orange } });
  slide.addShape(pptx.ShapeType.rect, { x: 0.72, y: 0.72, w: 1.1, h: 0.06, fill: { color: C.orange }, line: { color: C.orange } });
  addText(slide, 'DSP501 · FINAL PROJECT', 0.7, 0.72, 5.4, 0.3, { fontSize: 13, bold: true, color: '8FE3D8', charSpacing: 1.5 });
  addText(slide, 'Robust PCG Murmur\nDetection with DSP', 0.72, 1.5, 5.0, 1.6, { fontFace: 'Georgia', fontSize: 31, bold: true, color: C.white, breakLine: true });
  addText(slide, 'Nghiên cứu ảnh hưởng của sampling, quantization, filtering và time–frequency features trên tín hiệu phonocardiogram.', 0.72, 3.35, 5.2, 1.0, { fontSize: 18, color: 'D7E5EE' });
  addText(slide, 'CirCor DigiScope · SVM/MLP/CNN · Streamlit demo', 0.72, 5.85, 5.6, 0.3, { fontSize: 14, color: '8FE3D8', bold: true });
  slide.addShape(pptx.ShapeType.rect, { x: 5.92, y: 0.88, w: 6.82, h: 5.32, fill: { color: C.white }, line: { color: C.orange, width: 1.2 } });
  addImage(slide, 'circor_100_signal_report/signal_report.png', 6.08, 1.02, 6.5, 5.0, 8);
  slide.addShape(pptx.ShapeType.rect, { x: 5.92, y: 6.35, w: 6.82, h: 0.65, fill: { color: C.orange }, line: { color: C.orange } });
  addText(slide, 'Signal → spectrum → spectrogram → prediction', 6.28, 6.53, 6.05, 0.22, { fontSize: 14, bold: true, color: C.white, align: 'center' });
}

// 2. Problem
{
  const slide = pptx.addSlide(); addTitle(slide, 'Bài toán và động lực', '01 · Motivation');
  addText(slide, 'PCG là tín hiệu âm thanh 1D; murmur có thể làm thay đổi năng lượng, hình dạng phổ và cấu trúc theo thời gian.', 0.7, 1.55, 6.1, 0.8, { fontSize: 22, bold: true, color: C.navy });
  addBullets(slide, ['Input: file WAV phonocardiogram', 'Output: Absent / Present + xác suất lớp', 'Thách thức: nhiễu, sampling, bit depth và recording khác nhau', 'Mục tiêu: đo ảnh hưởng của DSP front-end, không chạy theo model phức tạp'], 0.75, 2.65, 5.5, 2.25, 17);
  card(slide, 7.35, 1.55, 2.35, 1.65, 'Signal', 'Waveform theo thời gian', C.teal, '1D');
  card(slide, 9.95, 1.55, 2.35, 1.65, 'Spectrum', 'FFT và PSD', C.orange, 'f');
  card(slide, 7.35, 3.55, 2.35, 1.65, 'Spectrogram', 'Biểu diễn 2D time–frequency', C.purple, '2D');
  card(slide, 9.95, 3.55, 2.35, 1.65, 'Decision', 'Prediction có xác suất', C.mint, 'ML');
  slide.addShape(pptx.ShapeType.line, { x: 9.7, y: 3.2, w: 0, h: 0.35, line: { color: C.line, width: 1.5, beginArrowType: 'none', endArrowType: 'triangle' } });
  addText(slide, 'Từ tín hiệu 1D đến biểu diễn 2D', 7.35, 5.65, 5, 0.3, { fontSize: 15, color: C.teal, bold: true, align: 'center' });
  addFooter(slide, 2);
}

// 3. Dataset
{
  const slide = pptx.addSlide(); addTitle(slide, 'Dataset và cách xử lý nhãn', '02 · Data');
  addMetric(slide, 0.75, 1.55, '942', 'bệnh nhân metadata', C.teal);
  addMetric(slide, 2.9, 1.55, '3,163', 'WAV hợp lệ', C.orange);
  addMetric(slide, 5.05, 1.55, '4 kHz', 'sample rate nguồn', C.purple);
  addMetric(slide, 7.2, 1.55, '0', 'file lỗi / missing', C.mint);
  slide.addChart(pptx.ChartType.bar, [{ name: 'Patients', labels: ['Absent', 'Present', 'Unknown'], values: [695, 179, 68] }], { x: 0.75, y: 2.65, w: 5.2, h: 3.35, catAxisLabelFontFace: 'Aptos', catAxisLabelFontSize: 13, valAxisLabelFontSize: 11, valAxisMinVal: 0, valAxisMaxVal: 720, valGridLine: { color: 'E2E8F0', width: 1 }, chartColors: [C.teal], showLegend: false, showTitle: false, showValue: true, dataLabelColor: C.ink, dataLabelPosition: 'outEnd', showCatName: false, showValAxisTitle: false, showCatAxisTitle: false, showBorder: false });
  slide.addShape(pptx.ShapeType.rect, { x: 6.45, y: 2.65, w: 5.8, h: 3.35, rectRadius: 0.08, fill: { color: C.white }, line: { color: C.line, width: 1 } });
  addText(slide, 'Patient-wise label handling', 6.8, 2.98, 4.8, 0.35, { fontSize: 20, bold: true, color: C.navy });
  addBullets(slide, ['Absent: 695 patients', 'Present: 179 patients', 'Unknown: 68 patients', '874 labeled patients dùng cho supervised training'], 6.8, 3.55, 4.85, 1.7, 16);
  addText(slide, 'Dataset source: PhysioNet CirCor DigiScope 1.0.3', 6.8, 5.55, 4.9, 0.25, { fontSize: 13, color: C.teal, italic: true });
  addFooter(slide, 3);
}

// 4. Questions/contributions
{
  const slide = pptx.addSlide(); addTitle(slide, 'Câu hỏi nghiên cứu và đóng góp', '03 · Research design');
  const qs = [
    ['01', 'Filter nào tốt nhất?', 'None · Butterworth · FIR', C.teal],
    ['02', 'Feature nào hữu ích?', 'PSD · MFCC · STFT · hybrid', C.orange],
    ['03', 'DSP có bền với thay đổi?', 'Sampling · bit depth · noise', C.purple],
    ['04', 'Model nào đủ tốt?', 'SVM baseline · MLP neural baseline', C.mint],
  ];
  qs.forEach((q, i) => {
    const x = 0.78 + (i % 2) * 6.1;
    const y = 1.55 + Math.floor(i / 2) * 2.25;
    slide.addShape(pptx.ShapeType.rect, { x, y, w: 5.55, h: 1.7, rectRadius: 0.08, fill: { color: C.white }, line: { color: C.line, width: 1 } });
    slide.addShape(pptx.ShapeType.ellipse, { x: x + 0.28, y: y + 0.34, w: 0.72, h: 0.72, fill: { color: q[3] }, line: { color: q[3] } });
    addText(slide, q[0], x + 0.28, y + 0.52, 0.72, 0.2, { fontSize: 17, bold: true, color: C.white, align: 'center' });
    addText(slide, q[1], x + 1.25, y + 0.28, 3.9, 0.35, { fontSize: 21, bold: true, color: C.navy });
    addText(slide, q[2], x + 1.25, y + 0.84, 3.9, 0.32, { fontSize: 15, color: C.muted });
  });
  addText(slide, 'Đóng góp cốt lõi: biến ảnh hưởng của từng khối DSP thành bằng chứng định lượng và trực quan.', 1.0, 6.35, 11.3, 0.4, { fontSize: 19, bold: true, color: C.teal, align: 'center' });
  addFooter(slide, 4);
}

// 5. DSP config
{
  const slide = pptx.addSlide(); addTitle(slide, 'Cấu hình DSP chính', '04 · DSP setup');
  slide.addShape(pptx.ShapeType.rect, { x: 0.75, y: 1.55, w: 5.6, h: 4.7, rectRadius: 0.08, fill: { color: C.white }, line: { color: C.line, width: 1 } });
  addText(slide, 'Baseline configuration', 1.08, 1.9, 4.7, 0.35, { fontSize: 21, bold: true, color: C.navy });
  const params = [['Target fs', '1 kHz'], ['Quantization', '16 bit'], ['Band-pass', '25–400 Hz'], ['Window', '3.0 s'], ['Hop', '1.5 s'], ['Filter', 'Butterworth IIR']];
  params.forEach((p, i) => {
    const y = 2.5 + i * 0.55;
    addText(slide, p[0], 1.08, y, 2.4, 0.25, { fontSize: 16, color: C.muted });
    addText(slide, p[1], 3.75, y, 2.0, 0.25, { fontSize: 16, bold: true, color: C.teal, align: 'right' });
    slide.addShape(pptx.ShapeType.line, { x: 1.08, y: y + 0.34, w: 4.55, h: 0, line: { color: 'E2E8F0', width: 0.7 } });
  });
  slide.addShape(pptx.ShapeType.rect, { x: 6.8, y: 1.55, w: 5.45, h: 2.0, rectRadius: 0.08, fill: { color: 'E6FFFB' }, line: { color: '99F6E4', width: 1 } });
  addText(slide, 'Vì sao 1 kHz đủ?', 7.15, 1.9, 4.6, 0.35, { fontSize: 22, bold: true, color: C.teal });
  addText(slide, 'Dải quan tâm 25–400 Hz nằm dưới Nyquist 500 Hz của tín hiệu 1 kHz.', 7.15, 2.45, 4.55, 0.7, { fontSize: 17, color: C.ink });
  slide.addShape(pptx.ShapeType.rect, { x: 6.8, y: 3.85, w: 5.45, h: 2.4, rectRadius: 0.08, fill: { color: 'FFF7ED' }, line: { color: 'FED7AA', width: 1 } });
  addText(slide, 'Hypothesis về quantization', 7.15, 4.2, 4.6, 0.35, { fontSize: 22, bold: true, color: C.orange });
  addText(slide, 'Bit depth thấp làm mất biến thiên biên độ nhỏ → thay đổi spectral features → giảm macro-F1.', 7.15, 4.75, 4.55, 0.75, { fontSize: 17, color: C.ink });
  addFooter(slide, 5);
}

// 6. Pipeline
{
  const slide = pptx.addSlide(); addTitle(slide, 'Pipeline DSP end-to-end', '05 · Method');
  const nodes = [
    ['WAV', 'read + mono', C.navy2], ['Resample', '1/2/4 kHz', C.teal], ['Quantize', '8/12/16 bit', C.orange], ['Filter', '25–400 Hz', C.purple], ['Segment', '3 s / 1.5 s', C.mint], ['Features', 'FFT · PSD · STFT · MFCC', C.teal], ['Model', 'SVM / MLP / CNN optional', C.orange], ['Output', 'label + probability', C.navy2],
  ];
  const x0 = 0.55; const gap = 0.12; const nw = 1.48; const y = 2.35;
  nodes.forEach((n, i) => {
    const x = x0 + i * (nw + gap);
    slide.addShape(pptx.ShapeType.rect, { x, y, w: nw, h: 1.45, rectRadius: 0.06, fill: { color: n[2] }, line: { color: n[2] } });
    addText(slide, n[0], x + 0.08, y + 0.35, nw - 0.16, 0.28, { fontSize: 18, bold: true, color: C.white, align: 'center' });
    addText(slide, n[1], x + 0.08, y + 0.86, nw - 0.16, 0.3, { fontSize: 11, color: 'E6FFFB', align: 'center' });
    if (i < nodes.length - 1) slide.addShape(pptx.ShapeType.line, { x: x + nw, y: y + 0.72, w: gap, h: 0, line: { color: C.muted, width: 1.5, endArrowType: 'triangle' } });
  });
  addText(slide, 'Một patient vector được tạo bằng cách trung bình các window và tất cả auscultation locations của cùng bệnh nhân.', 1.15, 4.55, 11.0, 0.55, { fontSize: 21, bold: true, color: C.navy, align: 'center' });
  slide.addShape(pptx.ShapeType.rect, { x: 2.45, y: 5.45, w: 8.35, h: 0.75, rectRadius: 0.04, fill: { color: C.pale }, line: { color: C.line, width: 1 } });
  addText(slide, 'No patient leakage: all recordings of one patient stay in one split.', 2.65, 5.69, 7.95, 0.22, { fontSize: 17, color: C.teal, bold: true, align: 'center' });
  addFooter(slide, 6);
}

// 7. Theory/features
{
  const slide = pptx.addSlide(); addTitle(slide, 'Từ signal 1D đến time–frequency image', '06 · DSP theory');
  slide.addShape(pptx.ShapeType.rect, { x: 0.75, y: 1.55, w: 5.4, h: 4.65, rectRadius: 0.08, fill: { color: C.navy2 }, line: { color: C.navy2 } });
  addText(slide, 'Ba phép biến đổi chính', 1.08, 1.9, 4.7, 0.35, { fontSize: 22, bold: true, color: C.white });
  addText(slide, 'FFT', 1.08, 2.55, 0.85, 0.3, { fontSize: 20, bold: true, color: '8FE3D8' });
  addText(slide, 'X[k] = Σ x[n] · e^(−j2πkn/N)', 2.0, 2.55, 3.6, 0.3, { fontSize: 17, color: C.white });
  addText(slide, 'PSD', 1.08, 3.35, 0.85, 0.3, { fontSize: 20, bold: true, color: 'FDBA74' });
  addText(slide, 'Năng lượng theo dải tần', 2.0, 3.35, 3.6, 0.3, { fontSize: 17, color: C.white });
  addText(slide, 'STFT', 1.08, 4.15, 0.85, 0.3, { fontSize: 20, bold: true, color: 'C4B5FD' });
  addText(slide, 'X(m,k) = FFT{x[n]w[n−mH]}', 2.0, 4.15, 3.6, 0.3, { fontSize: 17, color: C.white });
  addText(slide, 'STFT tạo ma trận 2D time–frequency; đây là cầu nối tự nhiên sang Image Processing.', 1.08, 5.05, 4.65, 0.62, { fontSize: 15, color: 'D7E5EE', italic: true });
  const feats = [['Stats', 'RMS · skew · kurtosis'], ['PSD', 'Band powers · entropy'], ['FFT', 'Peak frequencies'], ['MFCC', '13 coefficients'], ['Hybrid', 'Kết hợp toàn bộ'],];
  feats.forEach((f, i) => card(slide, 6.8 + (i % 2) * 2.85, 1.55 + Math.floor(i / 2) * 1.4, 2.55, 1.05, f[0], f[1], [C.teal, C.orange, C.purple, C.mint, C.navy2][i]));
  addText(slide, 'CNN chỉ là baseline bổ sung; mục tiêu chính vẫn là phân tích tác động DSP và tái lập CPU.', 6.85, 5.85, 5.2, 0.45, { fontSize: 16, color: C.orange, bold: true });
  addFooter(slide, 7);
}

// 8. Patient-wise split
{
  const slide = pptx.addSlide(); addTitle(slide, 'Patient-wise split để tránh leakage', '07 · Evaluation');
  addText(slide, 'Một bệnh nhân có nhiều recording ở các vị trí nghe khác nhau.', 0.8, 1.55, 5.2, 0.35, { fontSize: 21, bold: true, color: C.navy });
  const recs = [['AV', C.teal], ['MV', C.teal], ['PV', C.teal], ['TV', C.teal]];
  recs.forEach((r, i) => { const x = 1.05 + i * 1.1; slide.addShape(pptx.ShapeType.rect, { x, y: 2.35, w: 0.85, h: 0.65, rectRadius: 0.04, fill: { color: r[1] }, line: { color: r[1] } }); addText(slide, r[0], x, 2.56, 0.85, 0.18, { fontSize: 16, bold: true, color: C.white, align: 'center' }); });
  addText(slide, 'PATIENT 13918', 1.15, 3.25, 3.95, 0.3, { fontSize: 18, bold: true, color: C.teal, align: 'center' });
  slide.addShape(pptx.ShapeType.line, { x: 3.0, y: 3.0, w: 0, h: 0.2, line: { color: C.teal, width: 2, endArrowType: 'triangle' } });
  const splits = [['Train\n611', C.teal], ['Validation\n131', C.orange], ['Test\n132', C.purple]];
  splits.forEach((s, i) => { const x = 6.2 + i * 2.05; slide.addShape(pptx.ShapeType.rect, { x, y: 2.1, w: 1.65, h: 1.45, rectRadius: 0.06, fill: { color: s[1] }, line: { color: s[1] } }); addText(slide, s[0], x + 0.1, 2.47, 1.45, 0.55, { fontSize: 22, bold: true, color: C.white, align: 'center', breakLine: true }); });
  slide.addShape(pptx.ShapeType.line, { x: 5.25, y: 2.82, w: 0.85, h: 0, line: { color: C.muted, width: 1.5, endArrowType: 'triangle' } });
  addText(slide, 'Tất cả location của cùng patient nằm trong cùng một partition.', 6.25, 4.25, 5.65, 0.55, { fontSize: 19, color: C.navy, bold: true, align: 'center' });
  addText(slide, 'Nếu split theo recording, model có thể học “dấu vân tay” của patient thay vì murmur.', 1.0, 5.45, 11.2, 0.45, { fontSize: 19, color: C.orange, bold: true, align: 'center' });
  addFooter(slide, 8);
}

// 9. Matrix design
{
  const slide = pptx.addSlide(); addTitle(slide, 'Thiết kế matrix thực nghiệm', '08 · Experiment matrix');
  addText(slide, '24 cấu hình = 3 filters × 4 feature modes × 2 classifiers', 0.8, 1.5, 11.2, 0.4, { fontSize: 25, bold: true, color: C.navy, align: 'center' });
  const rows = [
    [{ text: 'Filter \\ Feature', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }, { text: 'PSD', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }, { text: 'MFCC', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }, { text: 'STFT', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }, { text: 'Hybrid', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }],
    ['None', 'SVM/MLP', 'SVM/MLP', 'SVM/MLP', 'SVM/MLP'],
    ['Butterworth', 'SVM/MLP', 'SVM/MLP', 'SVM/MLP', 'SVM/MLP'],
    ['FIR', 'SVM/MLP', 'SVM/MLP', 'SVM/MLP', 'SVM/MLP'],
  ];
  slide.addTable(rows, { x: 1.05, y: 2.25, w: 8.1, h: 2.65, colW: [2.0, 1.5, 1.5, 1.5, 1.6], rowH: [0.55, 0.7, 0.7, 0.7], border: { pt: 1, color: C.line }, fill: C.white, color: C.ink, fontFace: 'Aptos', fontSize: 17, align: 'center', valign: 'mid', margin: 0.08, bold: false });
  card(slide, 9.75, 2.15, 2.2, 1.4, 'Primary metric', 'Macro-F1', C.teal, 'F1');
  card(slide, 9.75, 3.85, 2.2, 1.4, 'Split', 'Patient-wise', C.orange, 'P');
  card(slide, 9.75, 5.55, 2.2, 1.0, 'Seeds', '42 · fixed', C.purple);
  addText(slide, 'Mỗi run lưu model bundle + config + metrics.json để tái lập inference.', 1.05, 5.55, 8.1, 0.45, { fontSize: 17, color: C.teal, bold: true, align: 'center' });
  addFooter(slide, 9);
}

// 10. Main results
{
  const slide = pptx.addSlide(); addTitle(slide, 'PILOT ONLY: hybrid + SVM dẫn đầu', '09 · Pilot subset results');
  addImage(slide, 'circor_100_f1.png', 0.75, 1.55, 7.15, 4.1);
  slide.addShape(pptx.ShapeType.rect, { x: 8.35, y: 1.55, w: 4.3, h: 1.45, rectRadius: 0.08, fill: { color: 'E6FFFB' }, line: { color: '99F6E4', width: 1 } });
  addText(slide, 'BEST', 8.7, 1.83, 0.8, 0.24, { fontSize: 13, bold: true, color: C.teal });
  addText(slide, 'Macro-F1 0.928', 9.65, 1.76, 2.55, 0.38, { fontSize: 27, bold: true, color: C.teal });
  addText(slide, 'SVM + Butterworth/FIR + hybrid', 8.7, 2.36, 3.35, 0.24, { fontSize: 14, color: C.ink });
  addText(slide, 'Confusion matrix — best model', 8.45, 3.35, 3.9, 0.3, { fontSize: 17, bold: true, color: C.navy });
  slide.addTable([[{ text: '', options: { fill: { color: C.navy2 }, color: C.white } }, { text: 'Pred A', options: { fill: { color: C.navy2 }, color: C.white, bold: true } }, { text: 'Pred P', options: { fill: { color: C.navy2 }, color: C.white, bold: true } }], [{ text: 'Actual A', options: { bold: true } }, '9', '1'], [{ text: 'Actual P', options: { bold: true } }, '0', '5']], { x: 8.45, y: 3.78, w: 3.65, h: 1.55, colW: [1.25, 1.2, 1.2], rowH: [0.4, 0.55, 0.55], border: { pt: 1, color: C.line }, fontFace: 'Aptos', fontSize: 17, align: 'center', valign: 'mid', margin: 0.06 });
  addText(slide, 'PILOT ONLY · 100 patients · not the full-cohort result', 8.45, 5.7, 3.8, 0.42, { fontSize: 14, color: C.orange, bold: true });
  addFooter(slide, 10);
}

// 11. Full-cohort benchmark
{
  const slide = pptx.addSlide(); addTitle(slide, 'Full-cohort benchmark: sampling & quantization', '10 · Native benchmark');
  addText(slide, 'Cùng SVM + Butterworth + hybrid · seed 42 · patient-wise split', 0.75, 1.35, 8.2, 0.3, { fontSize: 18, bold: true, color: C.navy });
  const rows = [
    [{ text: 'Configuration', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }, { text: 'Time (s)', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }, { text: '× vs 1 kHz', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }, { text: 'Accuracy', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }, { text: 'Macro-F1', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }],
    ['1 kHz + 16-bit', '85.8', '1.00×', '0.742', '0.615'],
    ['2 kHz + 16-bit', '98.4', '1.15×', '0.758', '0.628'],
    ['4 kHz + 16-bit', '122.8', '1.43×', '0.773', '0.630'],
    ['4 kHz + none', '126.9', '1.48×', '0.780', '0.637'],
    ['4 kHz + 8-bit', '128.7', '1.50×', '0.765', '0.634'],
    ['4 kHz + 12-bit', '126.3', '1.47×', '0.773', '0.630'],
  ];
  slide.addTable(rows, { x: 0.75, y: 1.85, w: 8.25, h: 4.55, colW: [2.55, 1.25, 1.45, 1.35, 1.45], rowH: [0.55, 0.66, 0.66, 0.66, 0.66, 0.66, 0.66], border: { pt: 1, color: C.line }, fill: C.white, color: C.ink, fontFace: 'Aptos', fontSize: 15, align: 'center', valign: 'mid', margin: 0.07 });
  card(slide, 9.35, 1.85, 3.05, 1.45, 'Fastest baseline', '1 kHz + 16-bit', C.teal, '85.8 s');
  card(slide, 9.35, 3.55, 3.05, 1.45, 'Best Macro-F1', '4 kHz + no extra quantization', C.orange, '0.637');
  card(slide, 9.35, 5.25, 3.05, 1.15, 'Trade-off', '4 kHz costs ≈43–48% more time', C.purple);
  addText(slide, '942 metadata patients · 3,163 WAV · 874 labeled · split 611 / 131 / 132', 0.75, 6.62, 8.7, 0.28, { fontSize: 15, color: C.teal, bold: true });
  addFooter(slide, 11);
}

// 12. Full-data matrix
{
  const slide = pptx.addSlide(); addTitle(slide, 'Full-data matrix: model × filter × feature', '11 · Full matrix');
  addText(slide, 'Target fs = 1 kHz · quantization = 16-bit · seed 42 · test = 132 patients', 0.75, 1.35, 9.2, 0.3, { fontSize: 18, bold: true, color: C.navy });
  const rows = [
    [{ text: 'Model', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }, { text: 'Filter', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }, { text: 'Feature', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }, { text: 'Accuracy', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }, { text: 'Balanced acc.', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }, { text: 'Macro-F1', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }],
    ['MLP', 'None', 'MFCC', '0.848', '0.657', '0.693'],
    ['MLP', 'None', 'Hybrid', '0.841', '0.625', '0.654'],
    ['SVM', 'None', 'Hybrid', '0.750', '0.664', '0.648'],
    ['SVM', 'FIR', 'Hybrid', '0.765', '0.646', '0.644'],
    ['SVM', 'None', 'MFCC', '0.742', '0.659', '0.642'],
    ['SVM', 'FIR', 'MFCC', '0.758', '0.628', '0.628'],
    ['SVM', 'Butterworth', 'Hybrid', '0.742', '0.618', '0.615'],
  ];
  slide.addTable(rows, { x: 0.7, y: 1.85, w: 8.85, h: 4.75, colW: [1.15, 1.65, 1.7, 1.35, 1.65, 1.35], rowH: [0.5, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58, 0.58], border: { pt: 1, color: C.line }, fill: C.white, color: C.ink, fontFace: 'Aptos', fontSize: 14, align: 'center', valign: 'mid', margin: 0.06 });
  card(slide, 9.85, 1.85, 2.45, 1.45, 'Best Macro-F1', 'MLP + none + MFCC', C.orange, '0.693');
  card(slide, 9.85, 3.55, 2.45, 1.45, 'Best balanced acc.', 'SVM + none + hybrid', C.teal, '0.664');
  card(slide, 9.85, 5.25, 2.45, 1.15, 'Cảnh báo', 'Present: 9/27 đúng ở best F1', C.purple);
  addText(slide, 'Không filter thắng trong split này; cần nhiều seed trước khi kết luận ưu thế tổng quát.', 0.75, 6.7, 8.7, 0.25, { fontSize: 15, color: C.teal, bold: true });
  addFooter(slide, 12);
}

// 13. CNN baseline and ablation
{
  const slide = pptx.addSlide(); addTitle(slide, 'CNN nhẹ: MobileNetV3-Small + log-STFT', '12 · CNN ablation');
  addText(slide, '874 labeled patients · same patient-wise split · 2 windows/recording · frozen pretrained backbone', 0.75, 1.35, 11.5, 0.3, { fontSize: 17, bold: true, color: C.navy });
  const rows = [
    [{ text: 'Aggregation / training', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }, { text: 'Accuracy', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }, { text: 'Balanced acc.', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }, { text: 'Macro-F1', options: { bold: true, color: C.white, fill: { color: C.navy2 } } }],
    ['Mean + tuned threshold', '0.841', '0.666', '0.697'],
    ['Median + tuned threshold', '0.833', '0.689', '0.710'],
    ['Top-25% windows', '0.765', '0.660', '0.653'],
    ['Max window', '0.667', '0.543', '0.536'],
    ['Median + last block fine-tuned', '0.841', '0.680', '0.708'],
  ];
  slide.addTable(rows, { x: 0.75, y: 1.9, w: 8.2, h: 3.9, colW: [3.55, 1.35, 1.55, 1.35], rowH: [0.55, 0.66, 0.66, 0.66, 0.66, 0.66], border: { pt: 1, color: C.line }, fill: C.white, color: C.ink, fontFace: 'Aptos', fontSize: 15, align: 'center', valign: 'mid', margin: 0.06 });
  card(slide, 9.35, 1.9, 3.05, 1.35, 'Best CNN Macro-F1', 'Median aggregation', C.teal, '0.710');
  card(slide, 9.35, 3.5, 3.05, 1.35, 'Threshold', 'Tuned on validation only', C.orange, '0.62');
  card(slide, 9.35, 5.1, 3.05, 1.15, 'Seed stability', 'Macro-F1 0.688 ± 0.025', C.purple);
  addText(slide, 'Kết luận: median ổn định hơn; max/top-25% overreact với window bất thường; unfreeze block cuối không đem lại lợi ích rõ.', 0.75, 6.35, 8.25, 0.45, { fontSize: 15, color: C.teal, bold: true });
  addFooter(slide, 13);
}

// 14. Robustness
{
  const slide = pptx.addSlide(); addTitle(slide, 'Pilot robustness: quantization và noise', '13 · Pilot robustness');
  addImage(slide, 'circor_100_robustness.png', 0.7, 1.45, 8.0, 4.25);
  card(slide, 9.05, 1.55, 3.15, 1.15, 'Sampling 1/2/4 kHz', 'Metric gần như không đổi', C.teal, '0.928');
  card(slide, 9.05, 2.95, 3.15, 1.15, 'Quantization 8-bit', 'Mất độ phân giải biên độ', C.orange, '0.732');
  card(slide, 9.05, 4.35, 3.15, 1.15, 'Pink / impulse noise', 'Suy giảm mạnh hơn white noise', C.purple, '0.796');
  addText(slide, 'Interpretation: giữ đúng dải tần là cần thiết, nhưng bit depth và loại nhiễu quyết định độ bền của feature.', 0.85, 6.1, 11.6, 0.45, { fontSize: 18, color: C.navy, bold: true, align: 'center' });
  addFooter(slide, 14);
}

// 15. Signal-level
{
  const slide = pptx.addSlide(); addTitle(slide, 'Signal-level analysis: từ waveform đến spectrogram', '14 · Signal + image');
  addImage(slide, 'circor_100_signal_report/signal_report.png', 0.65, 1.45, 7.3, 4.55);
  slide.addShape(pptx.ShapeType.rect, { x: 8.35, y: 1.55, w: 4.0, h: 4.35, rectRadius: 0.08, fill: { color: C.white }, line: { color: C.line, width: 1 } });
  addText(slide, 'Cách đọc hình', 8.7, 1.9, 3.2, 0.35, { fontSize: 22, bold: true, color: C.navy });
  addBullets(slide, ['Waveform: transient và nhịp tim', 'FFT: thành phần tần số', 'PSD: năng lượng theo band', 'STFT: năng lượng thay đổi theo thời gian', 'Spectrogram là biểu diễn 2D cho phần Image Processing'], 8.7, 2.55, 3.1, 2.35, 15);
  addText(slide, 'Điểm nhấn: model không chỉ “nhìn” raw audio; nó sử dụng các đặc trưng được giải thích bằng DSP.', 8.7, 5.18, 3.1, 0.55, { fontSize: 15, color: C.teal, bold: true });
  addFooter(slide, 15);
}

// 16. FE demo
{
  const slide = pptx.addSlide(); addTitle(slide, 'FE demo và kịch bản trình diễn', '15 · Demo');
  const steps = [['1', 'Upload WAV', 'Chọn một recording PCG', C.teal], ['2', 'Analyze', 'Chạy đúng DSP pipeline', C.orange], ['3', 'Inspect', 'Waveform · FFT · PSD · STFT', C.purple], ['4', 'Decide', 'Label + probabilities', C.mint]];
  steps.forEach((s, i) => {
    const x = 0.75 + i * 3.02;
    slide.addShape(pptx.ShapeType.rect, { x, y: 1.65, w: 2.55, h: 1.55, rectRadius: 0.06, fill: { color: C.white }, line: { color: C.line, width: 1 } });
    slide.addShape(pptx.ShapeType.ellipse, { x: x + 0.22, y: 2.02, w: 0.58, h: 0.58, fill: { color: s[3] }, line: { color: s[3] } });
    addText(slide, s[0], x + 0.22, 2.19, 0.58, 0.17, { fontSize: 16, bold: true, color: C.white, align: 'center' });
    addText(slide, s[1], x + 0.95, 1.98, 1.35, 0.25, { fontSize: 18, bold: true, color: C.navy });
    addText(slide, s[2], x + 0.95, 2.42, 1.35, 0.35, { fontSize: 12, color: C.muted });
    if (i < steps.length - 1) slide.addShape(pptx.ShapeType.line, { x: x + 2.55, y: 2.43, w: 0.47, h: 0, line: { color: C.line, width: 1.5, endArrowType: 'triangle' } });
  });
  addImage(slide, 'circor_100_signal_report/signal_report.png', 0.75, 3.75, 5.95, 2.55);
  slide.addShape(pptx.ShapeType.rect, { x: 7.05, y: 3.75, w: 5.15, h: 2.55, rectRadius: 0.08, fill: { color: C.navy2 }, line: { color: C.navy2 } });
  addText(slide, 'Kịch bản nói', 7.4, 4.05, 4.3, 0.3, { fontSize: 21, bold: true, color: C.white });
  addBullets(slide, ['“Đây là raw PCG và các biểu diễn sau DSP.”', '“Tôi đổi 16-bit sang 8-bit để kiểm tra robustness.”', '“FE mặc định dùng SVM; CNN chỉ là benchmark, chưa tích hợp vào app.”', '“Đây là research prototype, không phải diagnosis.”'], 7.4, 4.62, 4.3, 1.25, 15, 'E6FFFB');
  addFooter(slide, 16);
}

// 17. Limitations/conclusion
{
  const slide = pptx.addSlide();
  slide.background = { color: C.navy };
  addText(slide, 'KẾT LUẬN', 0.75, 0.65, 4.5, 0.3, { fontSize: 13, bold: true, color: '8FE3D8', charSpacing: 1.5 });
  addText(slide, 'DSP front-end quyết định\nđộ bền của hệ thống.', 0.75, 1.25, 6.0, 1.3, { fontSize: 36, bold: true, color: C.white, breakLine: true });
  addBullets(slide, ['Full benchmark: 1 kHz nhanh nhất; 4 kHz + none đạt Macro-F1 0.637 trong split hiện tại.', 'CNN frozen + median aggregation đạt Macro-F1 0.710; seed stability khoảng 0.688 ± 0.025.', 'Pipeline, model bundle và Streamlit FE đã chạy end-to-end.', 'Bước tiếp theo: segmentation-aware features và calibration.'], 0.82, 3.15, 6.2, 2.25, 18, 'D7E5EE');
  slide.addShape(pptx.ShapeType.rect, { x: 7.65, y: 1.25, w: 4.6, h: 3.95, rectRadius: 0.08, fill: { color: '17324D' }, line: { color: '2C526F', width: 1 } });
  addText(slide, 'Hạn chế cần nói rõ', 8.05, 1.65, 3.8, 0.3, { fontSize: 21, bold: true, color: 'FDBA74' });
  addBullets(slide, ['Public training release 942 bệnh nhân', 'Test set 132 bệnh nhân · seed 42', 'Chưa clinical validation', 'Chưa cycle-level segmentation'], 8.05, 2.25, 3.5, 1.75, 16, 'D7E5EE');
  addText(slide, 'Cảm ơn · Q&A', 7.65, 5.85, 4.6, 0.55, { fontSize: 27, bold: true, color: '8FE3D8', align: 'center' });
  addFooter(slide, 17);
}

pptx.writeFile({ fileName: output });
console.log(`Wrote ${output}`);
