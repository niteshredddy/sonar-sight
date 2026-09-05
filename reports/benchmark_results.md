# SonarSight — Benchmark Results

> Generated on: 2026-09-05 06:48:25


## Inference Latency (640×640 input)

| Stage | Mean (ms) | Std (ms) | Min (ms) | Max (ms) |
|-------|-----------|----------|----------|----------|
| CFAR Preprocessing | 35.2 | 2.6 | 31.0 | 46.0 |
| Full Pipeline (mock) | 52.1 | 3.1 | 46.4 | 57.1 |

## Model File Sizes

> No trained model weights found. Run `python model/train.py` first.

> Using mock detection mode for benchmarking.


### Expected sizes (YOLOv8n-seg):

| Format | Estimated Size |
|--------|---------------|
| PyTorch (.pt) | ~6.7 MB |
| ONNX FP32 | ~13 MB |
| ONNX INT8 | ~3.5 MB |

**Expected compression**: ~1.9x (FP32 → INT8)

## Edge Deployability Assessment

| Metric | Value | Target |
|--------|-------|--------|
| Preprocessing latency | 35 ms | < 50 ms ✅ |
| Pipeline latency (mock) | 52 ms | < 500 ms ✅ |
| Model format | ONNX + INT8 | ✅ Cross-platform |
| Runtime dependency | ONNX Runtime | ✅ No GPU required |

## Notes

- Latencies measured on CPU (no GPU acceleration)
- Mock detection mode used when no trained model is available
- INT8 quantization reduces model size ~2x with <1% mAP drop (typical)
- ONNX Runtime supports ARM/x86/GPU deployment without framework dependency