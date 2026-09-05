# 🌊 SonarSight — Marine Debris Detection System

> **Smart India Hackathon (SIH)** — Automated detection of man-made marine debris in side-scan sonar imagery.

![Python](https://img.shields.io/badge/Python-3.12-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-0.111-green)
![Next.js](https://img.shields.io/badge/Next.js-15-black)
![YOLOv8](https://img.shields.io/badge/YOLOv8--seg-Ultralytics-purple)

## 🎯 Problem Statement

Detect and geolocate man-made marine debris — **ghost nets**, **shipwrecks**, **pipes**, and **cylinders** — in side-scan sonar (SSS) imagery, providing actionable reports for maritime cleanup and safety operations.

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Next.js Dashboard                      │
│  ┌──────────┐  ┌──────────────┐  ┌───────────────────┐  │
│  │  Upload   │  │ Image+BBoxes │  │  Leaflet Map      │  │
│  │  Page     │  │ Overlay      │  │  + Detection Pins │  │
│  └────┬─────┘  └──────────────┘  └───────────────────┘  │
│       │              ▲                    ▲               │
└───────┼──────────────┼────────────────────┼──────────────┘
        │              │                    │
   REST API        JSON Report         GeoJSON
        │              │                    │
┌───────▼──────────────┴────────────────────┴──────────────┐
│                   FastAPI Backend                         │
│  /upload → /process/{id} → /status/{id} → /report/{id}  │
│              │                                            │
│         geotag.py (pixel → lat/lon)                      │
└──────────────┼───────────────────────────────────────────┘
               │
┌──────────────▼───────────────────────────────────────────┐
│              Model Inference Pipeline                     │
│  CFAR Preprocess → YOLOv8-seg → Confidence Rescorer     │
│  (speckle suppression)  (ONNX)   (geometric features)   │
└─────────────────────────────────────────────────────────┘
```

## 📂 Repository Structure

| Directory | Purpose |
|-----------|---------|
| `/schemas` | Shared Pydantic + TypeScript types |
| `/model` | Preprocessing, training, inference |
| `/backend` | FastAPI REST service |
| `/frontend` | Next.js + Tailwind dashboard |
| `/data` | Datasets, synthetic data, augmentation previews |
| `/reports` | Benchmarks, demo assets, talking points |
| `/tests` | Unit + integration tests |

## 🚀 Quick Start

### Prerequisites
- Python 3.12+
- Node.js 20+
- (Optional) CUDA GPU for training

### Backend
```bash
pip install -r requirements.txt
cd backend
uvicorn main:app --reload --port 8000
```

### Frontend
```bash
cd frontend
npm install
npm run dev  # → http://localhost:3000
```

### Generate Synthetic Data
```bash
python data/synthetic/generate_synthetic.py
python model/augmentation.py --preview
python model/tiling.py --input data/synthetic --output data/processed
```

### Run Tests
```bash
pytest tests/ -v
```

## 🔬 Pipeline Components

1. **CFAR Preprocessing** — Adaptive speckle suppression using Cell-Averaging CFAR with median filter fallback
2. **YOLOv8-seg Detection** — Instance segmentation fine-tuned on sonar imagery, exported to ONNX + INT8
3. **Confidence Rescoring** — Geometric feature analysis (shadow ratio, edge regularity, aspect ratio) to suppress natural formation false positives
4. **Geotagging** — Slant-range corrected pixel-to-lat/lon conversion using per-ping navigation data
5. **Dashboard** — Real-time results with image overlay, Leaflet map, and exportable reports

## 📊 Target Classes

| Class | Description | Sonar Signature |
|-------|-------------|-----------------|
| `ghost_net` | Abandoned fishing nets | Diffuse bright returns, irregular shadow |
| `shipwreck` | Sunken vessel remains | Large structured highlight + deep shadow |
| `pipe` | Underwater pipelines | Linear bright return, parallel shadow |
| `cylinder` | Cylindrical debris/UXO | Compact bright spot, sharp shadow |

## 📄 License

MIT — Built for Smart India Hackathon 2026
