# Project deliverables

Final presentation and report files for the DSP501 PCG murmur detection project.

- `DSP501_final_presentation.pptx` — editable presentation
- `DSP501_final_presentation.pdf` — presentation for quick reading
- `final_report_dsp501.docx` — editable final report
- `final_report_dsp501.pdf` — final report for quick reading
- `final_report_dsp501.txt` — plain-text report export
- `cnn_ablation_summary.md` — CNN aggregation, threshold, fine-tuning and seed-stability results
- `cnn_mobilenet_full_metrics.json` — original frozen MobileNetV3-Small run
- `cnn_ablation_median_seed42_metrics.json` — tuned median-aggregation CNN run

The report and presentation now include the full-cohort DSP matrix, native
sampling/quantization benchmark, lightweight CNN baseline, aggregation and
threshold ablations, partial fine-tuning comparison, and fixed-split seed
stability. CNN results use all 874 labeled patients but at most two windows per
recording; the MobileNet backbone is frozen in the recommended configuration.
