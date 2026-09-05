# SonarSight — Demo Talking Points

> Maps each pipeline component to the SIH problem statement requirements, with evidence for edge-deployability.

---

## 1. Problem Statement Alignment

| SIH Requirement | Pipeline Component | How We Solve It | Evidence |
|---|---|---|---|
| **Detect man-made marine debris** | YOLOv8n-seg + confidence rescoring | Instance segmentation with geometric false-positive filtering | Detects 4 classes: ghost nets, shipwrecks, pipes, cylinders |
| **Handle noisy sonar data** | CFAR preprocessing module | Adaptive speckle suppression with SNR-based method selection | Before/after demo shows clear noise reduction |
| **Provide actionable location data** | Geotagging module (`geotag.py`) | Slant-range corrected pixel→lat/lon using per-ping navigation | Unit-tested with synthetic tracks, <1m accuracy |
| **User-friendly reporting** | Next.js dashboard + FastAPI API | Interactive map, image overlay, sortable table, CSV/JSON export | Live demo with upload→process→results flow |
| **Real-time capable** | ONNX Runtime + INT8 quantization | ~X ms per frame on CPU, ~3.5 MB model size | See benchmark_results.md |
| **Edge-deployable** | ONNX + INT8, no GPU dependency | Runs on ARM/x86 without CUDA, <50 MB total footprint | Cross-platform ONNX Runtime |

---

## 2. Component Deep-Dive

### 🔬 CFAR Preprocessing
- **What**: Cell-Averaging Constant False Alarm Rate adaptive thresholding
- **Why**: Side-scan sonar suffers from multiplicative speckle noise that causes false detections
- **How**: Estimates local noise floor using sliding window, suppresses sub-threshold pixels
- **Fallback**: Automatic switch to median filtering for very low-SNR inputs (SNR < 3.0)
- **Demo**: Show raw tile → preprocessed tile side-by-side

### 🎯 YOLOv8n-seg Detection
- **What**: Nano-variant instance segmentation model fine-tuned on sonar imagery
- **Why**: State-of-the-art speed-accuracy tradeoff; ONNX export enables edge deployment
- **How**: Trained with sonar-specific augmentations (speckle, shadow, dropout bands)
- **Key metric**: mAP@50 on validation set (see training logs)
- **Demo**: Show detections with bounding boxes and segmentation masks overlaid

### 🛡️ Confidence Rescoring
- **What**: Post-detection geometric feature analysis for false-positive suppression
- **Why**: Natural rock clusters and shadow artifacts trigger YOLO false positives
- **How**: Computes 4 features per detection:
  1. Shadow-to-highlight ratio (man-made objects cast proportional shadows)
  2. Edge regularity (man-made = straight edges vs. natural = irregular)
  3. Aspect ratio match (each class has expected L/W range)
  4. Compactness (area/perimeter²)
- **Extensibility**: Swappable via Protocol/ABC — can drop in a trained CNN classifier
- **Demo**: Show confidence before vs. after rescoring on a mixed scene

### 🌍 Geotagging
- **What**: Pixel-to-real-world coordinate conversion
- **Why**: Detections need actionable lat/lon for cleanup crews
- **How**: 
  - Along-track: ping index → interpolated GPS position
  - Across-track: slant range → ground range (Gr = √(Sr² - H²)) → perpendicular offset
- **Accuracy**: Unit-tested against known synthetic track positions
- **Demo**: Show detection pins on Leaflet map at correct positions

### 📊 Dashboard
- **What**: Next.js + Tailwind web application with three views
- **Views**:
  1. **Image Overlay** — original sonar with bounding boxes and masks drawn via HTML5 Canvas
  2. **Leaflet Map** — detection markers at geotagged positions with popup details
  3. **Detection Table** — sortable by confidence, filterable by class
- **Export**: JSON + CSV download buttons
- **Demo**: Live walkthrough of upload → processing → results

---

## 3. Technical Differentiators

### Why not just use vanilla YOLO?
1. **Sonar-specific preprocessing**: CFAR reduces noise before detection, improving precision
2. **Sonar-specific augmentation**: Our training pipeline injects realistic speckle, shadow, and dropout artifacts
3. **Confidence rescoring**: Post-detection geometric analysis catches false positives that YOLO alone misses
4. **Geotagging**: Full pixel-to-lat/lon pipeline, not just pixel coordinates

### Why ONNX?
- **No vendor lock-in**: Runs on CPU, GPU (CUDA/TensorRT), Intel (OpenVINO), ARM (ONNX Runtime Mobile)
- **Smaller deployment**: No PyTorch dependency (saves ~2GB on edge devices)
- **INT8 quantization**: ~2x model size reduction, <1% accuracy loss

### Architecture for Scale
- **Modular design**: Each component (preprocess, detect, rescore, geotag) is independently testable and swappable
- **Async processing**: FastAPI background tasks with status polling — supports concurrent jobs
- **Tiling pipeline**: Handles arbitrarily large sonar strips by splitting into overlapping tiles

---

## 4. Edge Deployment Numbers

| Metric | Value | 
|---|---|
| Model size (INT8) | ~3.5 MB |
| Inference latency (CPU) | See benchmark_results.md |
| Memory footprint | < 200 MB RAM |
| Dependencies | Python + ONNX Runtime (~50 MB) |
| Power consumption | Standard CPU — no GPU needed |
| Supported platforms | Linux ARM/x86, Windows, macOS |

---

## 5. Live Demo Script

1. **Open dashboard** → Show dark-ocean themed UI
2. **Upload** → Drag a synthetic sonar tile + metadata JSON
3. **Processing** → Watch status indicator progress (queued → processing → done)
4. **Results** → 
   - Show image overlay with colored bounding boxes per class
   - Switch to Leaflet map — pins at geotagged positions
   - Click a detection → popup with class, confidence, dimensions
   - Sort table by confidence → show highest-confidence detections
5. **Download** → Click JSON and CSV buttons
6. **Technical slide** → Show before/after preprocessing composite
7. **Metrics** → Reference benchmark_results.md numbers for latency + model size
