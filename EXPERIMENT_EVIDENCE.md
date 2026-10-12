# Evidence map and evaluation qualifications

All figures here were extracted from the submitted `SOAL_2.ipynb` and retained as converted WebP files. The source PDF is included in `reports/`.

| Saved notebook (zero-based) cell | Output |
|---:|---|
| 14 | Five class sample images |
| 60 | Manual AlexNet accuracy/loss curves |
| 115 | EfficientNetB0 accuracy/loss curves |
| 127 | Weighted classification metrics |
| 132 | Confusion matrix |
| 136 | Model comparison results table |
| 148 | Multiclass ROC curves |

Training generated 2,550 samples; both validation and so-called test reused the remaining 1,092 images. The final multi-model evaluation passed `test_data` from the original rescaled manual-CNN pipeline, not the EfficientNet-specific pipeline. This leads to a cross-model input preprocessing mismatch. The reported EfficientNet scores remain archived academic outputs and should not be marketed as independent test performance or a rigorously controlled architecture comparison.
