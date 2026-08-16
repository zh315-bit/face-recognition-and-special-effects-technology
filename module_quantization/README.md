# Face Recognition Quantization Module

This module packages the ResNet50-IR ArcFace epoch-2 encoder artifacts and the scripts used to quantize, compare, and export it.

## Contents

- `models/epoch_2.pt`: FP32 training checkpoint.
- `models/epoch_2_dynamic_int8.pt`: CPU dynamic-INT8 encoder artifact.
- `models/epoch_2.onnx`: ONNX encoder with a dynamic batch axis.
- `scripts/quantize_model.py`: dynamic INT8 quantization exporter.
- `scripts/compare_quantization.py`: LFW accuracy, CPU speed, and artifact-size comparison.
- `scripts/export_onnx.py`: ONNX export, ONNX checker, and ONNX Runtime validation.
- `reports/`: generated quantization, ONNX, and comparison reports.

## Environment

Use Python 3.11 and install the dependencies:

```powershell
py -m pip install -r requirements.txt
```

`numpy<2` is required for the installed PyTorch build to interoperate with NumPy during ONNX validation.

## Reproduce

From this directory, run:

```powershell
py scripts/quantize_model.py --checkpoint models/epoch_2.pt --output models/epoch_2_dynamic_int8.pt --report reports/dynamic_quantization_report.json
py scripts/export_onnx.py --checkpoint models/epoch_2.pt --output models/epoch_2.onnx --report reports/onnx_export_report.json
py scripts/compare_quantization.py --checkpoint models/epoch_2.pt --quantized models/epoch_2_dynamic_int8.pt --lfw-root ../lfw --max-pairs 100
```

The comparison command writes `quantization_comparison.json` and `quantization_comparison.png` beside the FP32 checkpoint. Move those two outputs to `reports/` if regenerating the module artifacts.

## Existing Results

See `reports/quantization_comparison.json` and `reports/onnx_export_report.json` for the measured LFW, CPU latency, artifact size, ONNX output-error, and ONNX Runtime results.
