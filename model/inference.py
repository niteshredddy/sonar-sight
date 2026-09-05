"""
Full inference pipeline for sonar debris detection.

Chains: preprocess → tile → YOLO-seg → NMS across tiles → confidence rescore
        → Detection objects in original image coordinates.

Usage:
    from model.inference import SonarInferencePipeline

    pipeline = SonarInferencePipeline(model_path="runs/sonar/train/weights/best.pt")
    detections = pipeline.run("path/to/sonar_image.png")
"""

from __future__ import annotations

import json
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schemas.detection import BBox, Detection
from model.preprocessing import preprocess_sonar, detect_and_mask_nadir
from model.tiling import compute_tile_positions, extract_tile
from model.confidence_rescorer import GeometricRescorer, BaseRescorer


class SonarInferencePipeline:
    """
    End-to-end inference pipeline for sonar debris detection.
    """

    CLASS_MAP = {0: "ghost_net", 1: "shipwreck", 2: "pipe", 3: "cylinder"}

    def __init__(
        self,
        model_path: Optional[str] = None,
        tile_size: int = 640,
        tile_overlap: float = 0.5,
        conf_threshold: float = 0.45,
        iou_threshold: float = 0.45,
        min_area_pixels: int = 100,
        enable_nadir_mask: bool = True,
        preprocess_method: str = "auto",
        rescorer: Optional[BaseRescorer] = None,
        device: str = "cpu",
    ):
        self.tile_size = tile_size
        self.tile_overlap = tile_overlap
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.min_area_pixels = min_area_pixels
        self.enable_nadir_mask = enable_nadir_mask
        self.preprocess_method = preprocess_method
        self.rescorer = rescorer or GeometricRescorer()
        self.device = device
        self.model = None
        self.model_path = model_path

        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str):
        """Load YOLO model (supports .pt, .onnx formats)."""
        try:
            from ultralytics import YOLO
            self.model = YOLO(path)
            print(f"✅ Model loaded: {path}")
        except ImportError:
            print("⚠️  ultralytics not installed. Running in mock mode.")
            self.model = None
        except Exception as e:
            print(f"⚠️  Could not load model {path}: {e}. Running in mock mode.")
            self.model = None

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        """Apply CFAR/median preprocessing."""
        return preprocess_sonar(image, method=self.preprocess_method)

    def _tile_image(self, image: np.ndarray) -> list[tuple[np.ndarray, int, int]]:
        """Split image into overlapping tiles. Returns (tile, x_offset, y_offset) tuples."""
        h, w = image.shape[:2]

        # If image fits in a single tile, just return it
        if h <= self.tile_size and w <= self.tile_size:
            tile = np.zeros((self.tile_size, self.tile_size) + image.shape[2:], dtype=image.dtype)
            tile[:h, :w] = image
            return [(tile, 0, 0)]

        positions = compute_tile_positions(h, w, self.tile_size, self.tile_overlap)
        tiles = []
        for x, y in positions:
            tile = extract_tile(image, x, y, self.tile_size)
            tiles.append((tile, x, y))

        return tiles

    def _run_yolo_on_tile(
        self, tile: np.ndarray
    ) -> list[dict]:
        """
        Run YOLO-seg inference on a single tile.

        Returns list of raw detections with keys:
            class_id, confidence, bbox (x_min, y_min, x_max, y_max in tile coords),
            mask (polygon in tile coords)
        """
        if self.model is None:
            return self._mock_detections(tile)

        results = self.model.predict(
            tile,
            conf=self.conf_threshold,
            iou=self.iou_threshold,
            device=self.device,
            verbose=False,
        )

        detections = []
        for result in results:
            if result.boxes is None:
                continue

            for i, box in enumerate(result.boxes):
                det = {
                    "class_id": int(box.cls[0]),
                    "confidence": float(box.conf[0]) * 100,
                    "bbox": box.xyxy[0].cpu().numpy().tolist(),  # [x_min, y_min, x_max, y_max]
                }

                # Extract mask polygon if available
                if result.masks is not None and i < len(result.masks):
                    mask_xy = result.masks[i].xy
                    if len(mask_xy) > 0:
                        det["mask"] = [(float(p[0]), float(p[1])) for p in mask_xy[0]]

                detections.append(det)

        return detections

    def _mock_detections(self, tile: np.ndarray) -> list[dict]:
        """
        Generate mock detections for demo/testing when no model is loaded.

        Uses simple brightness-based heuristics to find potential objects
        in the tile, simulating what YOLO would detect.
        """
        gray = cv2.cvtColor(tile, cv2.COLOR_BGR2GRAY) if len(tile.shape) == 3 else tile

        # Find bright regions (potential objects)
        _, binary = cv2.threshold(gray, 160, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        detections = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < 200:  # Skip tiny blobs
                continue

            x, y, w, h = cv2.boundingRect(contour)
            aspect = w / max(h, 1)

            # Classify based on shape heuristics
            if aspect > 5:
                class_id = 2  # pipe
            elif aspect > 1.5 and area > 2000:
                class_id = 1  # shipwreck
            elif area < 1000:
                class_id = 3  # cylinder
            else:
                class_id = 0  # ghost_net

            # Confidence based on brightness contrast
            roi = gray[y:y + h, x:x + w]
            contrast = float(np.mean(roi)) / max(float(np.mean(gray)), 1)
            confidence = min(95, max(30, contrast * 50))

            # Polygon mask from contour
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            mask = [(float(p[0][0]), float(p[0][1])) for p in approx]

            detections.append({
                "class_id": class_id,
                "confidence": confidence,
                "bbox": [float(x), float(y), float(x + w), float(y + h)],
                "mask": mask,
            })

        return detections

    def _remap_to_original(
        self,
        tile_detections: list[dict],
        x_offset: int,
        y_offset: int,
    ) -> list[dict]:
        """Remap tile-space detections to original image coordinates."""
        remapped = []
        for det in tile_detections:
            bbox = det["bbox"]
            remapped_det = {
                **det,
                "bbox": [
                    bbox[0] + x_offset,
                    bbox[1] + y_offset,
                    bbox[2] + x_offset,
                    bbox[3] + y_offset,
                ],
            }

            # Remap mask polygon if present
            if "mask" in det and det["mask"]:
                remapped_det["mask"] = [
                    (p[0] + x_offset, p[1] + y_offset) for p in det["mask"]
                ]

            remapped.append(remapped_det)

        return remapped

    def _nms_across_tiles(self, all_detections: list[dict]) -> list[dict]:
        """
        Apply Non-Maximum Suppression across tile boundaries.

        Removes duplicate detections from overlapping tile regions.
        """
        if not all_detections:
            return []

        # Group by class
        by_class: dict[int, list[dict]] = {}
        for det in all_detections:
            cls_id = det["class_id"]
            by_class.setdefault(cls_id, []).append(det)

        kept = []
        for cls_id, dets in by_class.items():
            if not dets:
                continue

            # Sort by confidence (descending)
            dets.sort(key=lambda d: d["confidence"], reverse=True)

            # Greedy NMS
            suppressed = set()
            for i, det_i in enumerate(dets):
                if i in suppressed:
                    continue
                kept.append(det_i)

                for j in range(i + 1, len(dets)):
                    if j in suppressed:
                        continue
                    iou = self._compute_iou(det_i["bbox"], dets[j]["bbox"])
                    if iou > self.iou_threshold:
                        suppressed.add(j)

        return kept

    @staticmethod
    def _compute_iou(bbox1: list[float], bbox2: list[float]) -> float:
        """Compute Intersection over Union between two bboxes."""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])

        if x2 <= x1 or y2 <= y1:
            return 0.0

        intersection = (x2 - x1) * (y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection

        return intersection / max(union, 1e-6)

    def _rescore_detections(
        self, detections: list[dict], image: np.ndarray
    ) -> list[dict]:
        """Apply confidence rescoring to all detections."""
        rescored = []
        for det in detections:
            bbox = det["bbox"]
            x_min, y_min, x_max, y_max = [int(v) for v in bbox]

            # Clamp to image bounds
            h, w = image.shape[:2]
            x_min = max(0, x_min)
            y_min = max(0, y_min)
            x_max = min(w, x_max)
            y_max = min(h, y_max)

            if x_max <= x_min or y_max <= y_min:
                continue

            crop = image[y_min:y_max, x_min:x_max]

            # Create temporary Detection for rescorer
            class_label = self.CLASS_MAP.get(det["class_id"], "unknown")
            temp_detection = Detection(
                class_label=class_label,
                confidence=det["confidence"],
                bbox=BBox(x_min=bbox[0], y_min=bbox[1], x_max=bbox[2], y_max=bbox[3]),
                lat=0.0, lon=0.0,  # Will be filled by geotagging
                source_image="",
            )

            adjusted_conf = self.rescorer.rescore(crop, temp_detection)
            det["confidence"] = adjusted_conf
            rescored.append(det)

        return rescored

    def _to_detection_objects(
        self, detections: list[dict], source_image: str
    ) -> list[Detection]:
        """Convert raw detection dicts to Detection schema objects."""
        results = []
        for det in detections:
            class_label = self.CLASS_MAP.get(det["class_id"], "unknown")
            bbox = det["bbox"]

            mask = None
            if "mask" in det and det["mask"]:
                mask = [(float(p[0]), float(p[1])) for p in det["mask"]]

            results.append(Detection(
                id=str(uuid.uuid4()),
                class_label=class_label,
                confidence=round(det["confidence"], 1),
                bbox=BBox(
                    x_min=bbox[0], y_min=bbox[1],
                    x_max=bbox[2], y_max=bbox[3],
                ),
                mask=mask,
                lat=0.0,   # Placeholder — filled by backend geotagging
                lon=0.0,
                source_image=source_image,
            ))

        return results

    def _filter_detections(
        self,
        detections: list[dict],
        image_shape: tuple[int, ...],
        nadir_bounds: tuple[int, int],
        min_area: int,
        conf_thresh: float,
        enable_nadir: bool,
    ) -> list[dict]:
        """
        Filter out low-confidence, micro-area, and nadir-stripe false positive detections.
        """
        filtered = []
        col_start, col_end = nadir_bounds

        for det in detections:
            bbox = det["bbox"]
            w = max(0, bbox[2] - bbox[0])
            h = max(0, bbox[3] - bbox[1])
            area = w * h

            # 1. Confidence thresholding
            if det["confidence"] < conf_thresh:
                continue

            # 2. Area thresholding
            if area < min_area:
                continue

            # 3. Nadir water-column stripe filtering
            if enable_nadir and col_end > col_start:
                cx = (bbox[0] + bbox[2]) / 2.0
                if col_start <= cx <= col_end:
                    continue

            filtered.append(det)

        return filtered

    def run(
        self,
        image_path: str | Path,
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        min_area_pixels: Optional[int] = None,
        enable_nadir_mask: Optional[bool] = None,
    ) -> tuple[list[Detection], dict]:
        image_path = Path(image_path)
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Could not read image: {image_path}")

        return self.run_on_array(
            image=image,
            source_name=image_path.name,
            conf_threshold=conf_threshold,
            iou_threshold=iou_threshold,
            min_area_pixels=min_area_pixels,
            enable_nadir_mask=enable_nadir_mask,
        )

    def run_on_array(
        self,
        image: np.ndarray,
        source_name: str = "uploaded_image.png",
        conf_threshold: Optional[float] = None,
        iou_threshold: Optional[float] = None,
        min_area_pixels: Optional[int] = None,
        enable_nadir_mask: Optional[bool] = None,
    ) -> tuple[list[Detection], dict]:
        """Run pipeline on a numpy array with dynamic parameter overrides."""
        conf_thresh = conf_threshold if conf_threshold is not None else self.conf_threshold
        iou_thresh = iou_threshold if iou_threshold is not None else self.iou_threshold
        min_area = min_area_pixels if min_area_pixels is not None else self.min_area_pixels
        enable_nadir = enable_nadir_mask if enable_nadir_mask is not None else self.enable_nadir_mask

        timings = {}

        # 1. Preprocess & Nadir detection
        t0 = time.time()
        preprocessed = self._preprocess(image)
        nadir_mask, nadir_bounds = detect_and_mask_nadir(image)
        timings["preprocess_ms"] = (time.time() - t0) * 1000

        # 2. Tile image
        t0 = time.time()
        tiles = self._tile_image(preprocessed)
        timings["tiling_ms"] = (time.time() - t0) * 1000
        timings["num_tiles"] = len(tiles)

        # 3. Inference
        t0 = time.time()
        all_detections = []
        for tile, x_off, y_off in tiles:
            tile_dets = self._run_yolo_on_tile(tile)
            remapped = self._remap_to_original(tile_dets, x_off, y_off)
            all_detections.extend(remapped)
        timings["inference_ms"] = (time.time() - t0) * 1000

        # 4. NMS across tiles
        t0 = time.time()
        # Temporarily override iou threshold for NMS
        old_iou = self.iou_threshold
        self.iou_threshold = iou_thresh
        nms_detections = self._nms_across_tiles(all_detections)
        self.iou_threshold = old_iou
        timings["nms_ms"] = (time.time() - t0) * 1000

        # 5. Rescore & Filter
        t0 = time.time()
        rescored = self._rescore_detections(nms_detections, image)
        filtered = self._filter_detections(
            rescored, image.shape, nadir_bounds, min_area, conf_thresh, enable_nadir
        )
        timings["rescore_ms"] = (time.time() - t0) * 1000

        # 6. Convert to Detection objects
        detections = self._to_detection_objects(filtered, source_name)
        timings["total_ms"] = sum(v for k, v in timings.items() if k.endswith("_ms"))
        timings["num_detections"] = len(detections)

        return detections, timings
