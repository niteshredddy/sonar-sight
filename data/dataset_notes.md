# Public Side-Scan Sonar Datasets for Marine Debris Detection

> Curated list of publicly available datasets suitable for training man-made debris detectors on SSS imagery.

---

## 1. AI4Shipwrecks — Shipwreck Segmentation Benchmark

- **Source**: [OpenReview / Zenodo](https://zenodo.org)
- **Paper**: NeurIPS 2023 workshop
- **Size**: ~500 annotated side-scan sonar images
- **Classes**: Shipwreck (segmentation masks)
- **Format**: COCO-style JSON annotations
- **Resolution**: Varies (waterfall strips)
- **Relevance**: ★★★★★ — Direct match for `shipwreck` class
- **Notes**: High-quality pixel-level annotations. Includes diverse wreck types and seabed textures.

## 2. SSS-Mine Dataset (MILCO/NOMBO)

- **Source**: [Figshare](https://figshare.com) / MDPI Remote Sensing paper
- **Paper**: "Deep learning for mine-like object detection in SSS" (2023)
- **Size**: 1,170 real sonar images (2010–2021)
- **Classes**: Mine-Like Contacts (MILCO), Non-Mine-like Bottom Objects (NOMBO)
- **Format**: PNG images with classification labels
- **Resolution**: Cropped targets, ~100×100 to 300×300 px
- **Relevance**: ★★★★☆ — MILCO analogous to `cylinder` class; NOMBO provides negative examples
- **Notes**: Real operational data from naval surveys. Good for transfer learning to cylindrical object detection.

## 3. Roboflow Universe — Side-Scan Sonar Collections

- **Source**: [Roboflow Universe](https://universe.roboflow.com/) (search "side scan sonar")
- **Size**: Multiple projects, 200–800 images each
- **Classes**: Varies (ships, mines, debris, seabed features)
- **Format**: YOLO, COCO, Pascal VOC (export any format)
- **Relevance**: ★★★☆☆ — Mixed quality; useful for augmenting training data
- **Notable projects**:
  - "Side Scan Sonar Object Detection" — annotated targets with bounding boxes
  - "Underwater Sonar Image" — general sonar object detection
- **Notes**: Check license per project. Some are CC-BY, others research-only.

## 4. OpenSonarDatasets (REMARO Network)

- **Source**: [github.com/remaro-network/OpenSonarDatasets](https://github.com/remaro-network/OpenSonarDatasets)
- **Size**: Aggregated index of 15+ sonar datasets
- **Classes**: Various (objects, terrain, structures)
- **Format**: Mixed (links to original repositories)
- **Relevance**: ★★★☆☆ — Comprehensive starting point for discovering additional data
- **Notes**: Maintained by the EU REMARO network. Not all datasets have permissive licenses.

## 5. SWDD — Sonar Wall Detection Dataset

- **Source**: [Zenodo](https://zenodo.org)
- **Size**: ~300 annotated sonar images
- **Classes**: Wall / man-made structures
- **Format**: COCO-style annotations
- **Relevance**: ★★★☆☆ — Man-made structures applicable to `pipe` detection
- **Notes**: Useful for transfer learning on linear structural features.

## 6. NOAA Sonar Archives (Unannotated)

- **Source**: [NOAA National Centers for Environmental Information](https://www.ncei.noaa.gov/)
- **Size**: Petabytes of raw multibeam/sidescan data
- **Classes**: Unannotated
- **Format**: Raw XTF, JSF, or GeoTIFF
- **Relevance**: ★★☆☆☆ — Requires manual annotation; useful for background/negative samples
- **Notes**: Real-world operational data. Can be used for domain adaptation (unsupervised).

## 7. Marine Debris Datasets (Optical — Transfer Learning)

- **Source**: [Kaggle](https://kaggle.com), Roboflow Universe
- **Size**: 1,000–10,000+ images
- **Classes**: Plastic, nets, ropes, metal debris
- **Format**: YOLO, COCO
- **Relevance**: ★★☆☆☆ — Optical (not sonar), but useful for understanding debris morphology
- **Notes**: Can inform class definitions and augmentation strategies even though modality differs.

---

## Recommended Strategy for This Project

1. **Primary training data**: AI4Shipwrecks + SSS-Mine → covers `shipwreck` and `cylinder`
2. **Supplement with**: Roboflow SSS collections → additional object diversity
3. **Gap classes** (`ghost_net`, `pipe`): Use synthetic data generation (`data/synthetic/generate_synthetic.py`) to create procedural sonar tiles with known object placements
4. **Background/negative data**: NOAA raw sonar strips for realistic seabed textures
5. **Label format**: Convert everything to YOLOv8-seg format (txt files with polygon masks)

### Conversion Commands
```bash
# Roboflow: export directly in YOLOv8 format from the web UI
# COCO → YOLO: use ultralytics built-in converter
from ultralytics.data.converter import convert_coco

convert_coco(labels_dir="path/to/coco/annotations", use_segments=True)
```
