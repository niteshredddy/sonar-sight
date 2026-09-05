"""
YOLOv8-seg training script for sonar debris detection.

Trains a YOLOv8n-seg (nano segmentation) model on the tiled, augmented
synthetic sonar dataset. Exports best weights to ONNX and INT8 ONNX.

Usage:
    python model/train.py [--data data/synthetic/sonar_data.yaml] \\
                          [--epochs 100] [--imgsz 640] [--batch 16]
"""

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def train_model(
    data_yaml: str,
    model_name: str = "yolov8n-seg.pt",
    epochs: int = 100,
    imgsz: int = 640,
    batch: int = 16,
    device: str = "",
    project: str = "runs/sonar",
    name: str = "train",
):
    """
    Train YOLOv8-seg on the sonar dataset.

    Args:
        data_yaml: Path to dataset YAML config
        model_name: Pretrained model checkpoint
        epochs: Number of training epochs
        imgsz: Input image size
        batch: Batch size
        device: Device string ('' for auto, 'cpu', '0', etc.)
        project: Output project directory
        name: Run name
    """
    try:
        from ultralytics import YOLO
    except ImportError:
        print("❌ ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    print(f"🚀 Training {model_name} on {data_yaml}")
    print(f"   Epochs: {epochs} | Image size: {imgsz} | Batch: {batch}")

    # Load pretrained model
    model = YOLO(model_name)

    # Train
    start_time = time.time()
    results = model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device if device else None,
        project=project,
        name=name,
        patience=20,         # Early stopping
        save=True,
        save_period=10,       # Save checkpoint every 10 epochs
        plots=True,
        verbose=True,
        # Sonar-friendly hyperparams
        hsv_h=0.0,           # No hue augmentation (grayscale sonar)
        hsv_s=0.0,           # No saturation augmentation
        hsv_v=0.2,           # Slight value/brightness augmentation
        degrees=10.0,        # Small rotation
        flipud=0.3,          # Vertical flip (sonar is symmetric)
        fliplr=0.5,          # Horizontal flip
        mosaic=0.8,          # Mosaic augmentation
        mixup=0.1,           # Light mixup
    )

    train_time = time.time() - start_time
    print(f"✅ Training complete in {train_time:.1f}s")

    return model, results, train_time


def export_model(model, data_yaml: str, output_dir: Path):
    """Export trained model to ONNX and INT8 ONNX formats."""
    print("\n📦 Exporting model weights...")

    output_dir.mkdir(parents=True, exist_ok=True)

    # Full precision ONNX
    print("   → ONNX (full precision)")
    try:
        onnx_path = model.export(format="onnx", imgsz=640, simplify=True)
        print(f"   ✅ ONNX: {onnx_path}")
    except Exception as e:
        print(f"   ⚠️ ONNX export failed: {e}")
        onnx_path = None

    # INT8 quantized ONNX
    print("   → ONNX (INT8 quantized)")
    try:
        int8_path = model.export(format="onnx", imgsz=640, int8=True, data=data_yaml)
        print(f"   ✅ INT8 ONNX: {int8_path}")
    except Exception as e:
        print(f"   ⚠️ INT8 export failed: {e}")
        int8_path = None

    return onnx_path, int8_path


def benchmark_inference(model, data_yaml: str, imgsz: int = 640):
    """Measure inference latency on CPU and GPU."""
    import numpy as np

    print("\n⏱️  Benchmarking inference latency...")

    # Create dummy input
    dummy = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    results = {}

    # CPU benchmark
    print("   → CPU inference...")
    try:
        times = []
        # Warmup
        for _ in range(3):
            model.predict(dummy, device="cpu", verbose=False)
        # Measure
        for _ in range(20):
            t0 = time.time()
            model.predict(dummy, device="cpu", verbose=False)
            times.append((time.time() - t0) * 1000)

        results["cpu_ms"] = {
            "mean": float(np.mean(times)),
            "std": float(np.std(times)),
            "min": float(np.min(times)),
            "max": float(np.max(times)),
        }
        print(f"   ✅ CPU: {results['cpu_ms']['mean']:.1f} ± {results['cpu_ms']['std']:.1f} ms")
    except Exception as e:
        print(f"   ⚠️ CPU benchmark failed: {e}")

    # GPU benchmark (if available)
    try:
        import torch
        if torch.cuda.is_available():
            print("   → GPU inference...")
            times = []
            for _ in range(3):
                model.predict(dummy, device="0", verbose=False)
            for _ in range(50):
                t0 = time.time()
                model.predict(dummy, device="0", verbose=False)
                times.append((time.time() - t0) * 1000)

            results["gpu_ms"] = {
                "mean": float(np.mean(times)),
                "std": float(np.std(times)),
                "min": float(np.min(times)),
                "max": float(np.max(times)),
            }
            print(f"   ✅ GPU: {results['gpu_ms']['mean']:.1f} ± {results['gpu_ms']['std']:.1f} ms")
        else:
            print("   ℹ️  No CUDA GPU available, skipping GPU benchmark")
    except ImportError:
        print("   ℹ️  PyTorch not available for GPU check")

    return results


def log_results(results, train_time: float, benchmark: dict, output_dir: Path):
    """Log training results and benchmarks to JSON."""
    output_dir.mkdir(parents=True, exist_ok=True)

    log = {
        "training_time_seconds": train_time,
        "benchmark": benchmark,
    }

    # Extract metrics if available
    try:
        if hasattr(results, 'results_dict'):
            log["metrics"] = {
                k: float(v) if hasattr(v, '__float__') else str(v)
                for k, v in results.results_dict.items()
            }
    except Exception:
        pass

    log_path = output_dir / "training_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f, indent=2)

    print(f"\n📊 Results logged → {log_path}")


def main():
    parser = argparse.ArgumentParser(description="Train YOLOv8-seg for sonar debris detection")
    parser.add_argument("--data", type=str,
                        default=str(PROJECT_ROOT / "data" / "synthetic" / "sonar_data.yaml"),
                        help="Path to dataset YAML")
    parser.add_argument("--model", type=str, default="yolov8n-seg.pt",
                        help="Pretrained model name/path")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", type=str, default="", help="Device ('' for auto)")
    parser.add_argument("--no-export", action="store_true", help="Skip ONNX export")
    parser.add_argument("--no-benchmark", action="store_true", help="Skip benchmarking")
    args = parser.parse_args()

    # Train
    model, results, train_time = train_model(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
    )

    # Export
    if not args.no_export:
        export_model(model, args.data, PROJECT_ROOT / "runs" / "sonar" / "weights")

    # Benchmark
    benchmark = {}
    if not args.no_benchmark:
        benchmark = benchmark_inference(model, args.data, args.imgsz)

    # Log
    log_results(results, train_time, benchmark, PROJECT_ROOT / "runs" / "sonar")


if __name__ == "__main__":
    main()
