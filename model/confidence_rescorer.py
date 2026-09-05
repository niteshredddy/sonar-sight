"""
Confidence rescoring module for false-positive suppression.

Computes geometric features per detection crop to distinguish man-made
debris from natural rock clusters, sand ripples, and acoustic shadow
artifacts. Produces an adjusted confidence score (0-100).

The rescorer is exposed via a Protocol/ABC so it can be swapped for a
small CNN classifier in the future without changing the pipeline.

Features computed:
    1. Shadow-to-highlight ratio — length of trailing dark region vs bright object
    2. Edge regularity — Canny edge analysis, man-made objects have straighter edges
    3. Aspect ratio — debris classes have characteristic L/W distributions
    4. Compactness — area/(perimeter²), distinguishes circles from irregular blobs
"""

from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schemas.detection import BBox, Detection


class BaseRescorer(ABC):
    """Abstract base class for confidence rescorers."""

    @abstractmethod
    def rescore(self, crop: np.ndarray, detection: Detection) -> float:
        """
        Compute adjusted confidence for a single detection.

        Args:
            crop: Image crop of the detection region (BGR or grayscale)
            detection: Original detection with raw YOLO confidence

        Returns:
            Adjusted confidence score (0-100)
        """
        ...

    def rescore_batch(self, crops: list[np.ndarray], detections: list[Detection]) -> list[float]:
        """Rescore a batch of detections."""
        return [self.rescore(c, d) for c, d in zip(crops, detections)]


class GeometricRescorer(BaseRescorer):
    """
    Heuristic-based rescorer using geometric features.

    Analyzes each detection crop for properties characteristic of
    man-made objects vs. natural seabed formations.
    """

    # Expected aspect ratios per class (width/height)
    CLASS_ASPECT_RATIOS = {
        "ghost_net": (1.0, 4.0),    # Wider than tall, irregular
        "shipwreck": (1.5, 6.0),    # Elongated
        "pipe": (5.0, 50.0),        # Very elongated (linear)
        "cylinder": (0.5, 2.5),     # Roughly compact
    }

    # Weights for combining feature scores
    FEATURE_WEIGHTS = {
        "shadow_ratio": 0.25,
        "edge_regularity": 0.30,
        "aspect_ratio": 0.20,
        "compactness": 0.25,
    }

    def __init__(
        self,
        shadow_threshold: int = 60,
        highlight_threshold: int = 150,
        min_confidence: float = 5.0,
        max_boost: float = 15.0,
        max_penalty: float = 30.0,
    ):
        self.shadow_threshold = shadow_threshold
        self.highlight_threshold = highlight_threshold
        self.min_confidence = min_confidence
        self.max_boost = max_boost
        self.max_penalty = max_penalty

    def _compute_shadow_ratio(self, gray: np.ndarray) -> float:
        """
        Compute shadow-to-highlight area ratio.

        Man-made objects produce consistent, proportional shadows.
        Natural formations have irregular shadow patterns.

        Returns: Score 0-1 (1 = consistent with man-made object)
        """
        h, w = gray.shape[:2]
        if h < 4 or w < 4:
            return 0.5

        # Split into top half (expected highlight) and bottom half (expected shadow)
        mid = h // 2
        top_half = gray[:mid, :]
        bottom_half = gray[mid:, :]

        highlight_area = np.sum(top_half > self.highlight_threshold)
        shadow_area = np.sum(bottom_half < self.shadow_threshold)

        total_pixels = h * w / 2
        if total_pixels == 0:
            return 0.5

        highlight_ratio = highlight_area / total_pixels
        shadow_ratio = shadow_area / total_pixels

        # Man-made objects: highlight and shadow both present, proportional
        if highlight_ratio > 0.05 and shadow_ratio > 0.05:
            # Ratio of shadow to highlight area — typical range 0.3-3.0 for debris
            ratio = shadow_ratio / max(highlight_ratio, 1e-6)
            if 0.3 <= ratio <= 3.0:
                return min(1.0, 0.7 + 0.3 * (1 - abs(ratio - 1.0)))
            else:
                return max(0.1, 0.5 - abs(ratio - 1.5) * 0.1)

        return 0.3

    def _compute_edge_regularity(self, gray: np.ndarray) -> float:
        """
        Measure edge regularity using Canny edge detection + line fitting.

        Man-made objects tend to have straight edges; natural formations
        have more irregular, fractal-like edges.

        Returns: Score 0-1 (1 = very regular/straight edges)
        """
        if gray.shape[0] < 8 or gray.shape[1] < 8:
            return 0.5

        # Canny edge detection
        edges = cv2.Canny(gray, 50, 150)
        edge_pixels = np.sum(edges > 0)
        total_pixels = edges.shape[0] * edges.shape[1]

        if edge_pixels < 10:
            return 0.3

        # Use Hough lines to detect straight edges
        lines = cv2.HoughLinesP(
            edges, rho=1, theta=np.pi / 180, threshold=15,
            minLineLength=max(10, min(gray.shape) // 4),
            maxLineGap=5,
        )

        if lines is None or len(lines) == 0:
            return 0.3

        # Compute total straight-line pixel coverage
        line_pixel_count = 0
        for line in lines:
            pts = line.ravel()
            if len(pts) >= 4:
                x1, y1, x2, y2 = pts[:4]
                line_pixel_count += float(np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2))

        # Ratio of straight-line edges to total edges
        regularity = min(1.0, line_pixel_count / max(edge_pixels, 1))

        return float(regularity)

    def _compute_aspect_ratio_score(self, bbox: BBox, class_label: str) -> float:
        """
        Score how well the detection's aspect ratio matches expected
        range for its class.

        Returns: Score 0-1 (1 = perfect match)
        """
        w = bbox.width
        h = bbox.height

        if h <= 0 or w <= 0:
            return 0.3

        aspect = w / h
        expected_range = self.CLASS_ASPECT_RATIOS.get(class_label, (0.5, 5.0))
        low, high = expected_range

        if low <= aspect <= high:
            # Within expected range — score based on how centered
            center = (low + high) / 2
            deviation = abs(aspect - center) / (high - low)
            return max(0.5, 1.0 - deviation)
        else:
            # Outside expected range — penalty
            if aspect < low:
                return max(0.1, 0.5 - (low - aspect) * 0.1)
            else:
                return max(0.1, 0.5 - (aspect - high) * 0.05)

    def _compute_compactness(self, gray: np.ndarray) -> float:
        """
        Compute compactness: 4π·area/(perimeter²).

        Circles score 1.0. Irregular blobs score lower.
        Man-made objects tend to have higher compactness than
        natural rock clusters.

        Returns: Score 0-1
        """
        # Threshold to get object mask
        _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return 0.5

        # Use the largest contour
        largest = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest)
        perimeter = cv2.arcLength(largest, True)

        if perimeter <= 0:
            return 0.5

        compactness = (4 * np.pi * area) / (perimeter ** 2)
        return min(1.0, compactness)

    def rescore(self, crop: np.ndarray, detection: Detection) -> float:
        """
        Compute adjusted confidence using geometric feature analysis.

        Combines shadow ratio, edge regularity, aspect ratio match,
        and compactness into a weighted score that adjusts the raw
        YOLO confidence.
        """
        if crop is None or crop.size == 0:
            return max(self.min_confidence, detection.confidence * 0.5)

        # Convert to grayscale
        if len(crop.shape) == 3:
            gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            gray = crop.copy()

        # Compute feature scores
        shadow_score = self._compute_shadow_ratio(gray)
        edge_score = self._compute_edge_regularity(gray)
        aspect_score = self._compute_aspect_ratio_score(detection.bbox, detection.class_label)
        compact_score = self._compute_compactness(gray)

        # Weighted combination
        feature_score = (
            self.FEATURE_WEIGHTS["shadow_ratio"] * shadow_score +
            self.FEATURE_WEIGHTS["edge_regularity"] * edge_score +
            self.FEATURE_WEIGHTS["aspect_ratio"] * aspect_score +
            self.FEATURE_WEIGHTS["compactness"] * compact_score
        )

        # Adjust raw confidence
        # feature_score ranges ~0-1; center at 0.5 = neutral
        adjustment = (feature_score - 0.5) * 2  # range: -1 to +1

        if adjustment > 0:
            delta = adjustment * self.max_boost
        else:
            delta = adjustment * self.max_penalty

        adjusted = detection.confidence + delta
        adjusted = max(self.min_confidence, min(100.0, adjusted))

        return round(adjusted, 1)


class CNNRescorer(BaseRescorer):
    """
    Placeholder for a CNN-based rescorer.

    Drop-in replacement for GeometricRescorer. Train a small
    classification CNN (e.g., MobileNetV3-Small) on crops labeled
    as true-positive vs false-positive to learn the decision boundary.
    """

    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        if model_path:
            self._load_model(model_path)

    def _load_model(self, path: str):
        """Load a trained CNN model for rescoring."""
        # TODO: Implement CNN loading (ONNX or PyTorch)
        raise NotImplementedError(
            "CNN rescorer not yet implemented. Use GeometricRescorer for now."
        )

    def rescore(self, crop: np.ndarray, detection: Detection) -> float:
        if self.model is None:
            return detection.confidence
        # TODO: Run CNN inference on crop, return probability * 100
        raise NotImplementedError


def get_rescorer(method: str = "geometric", **kwargs) -> BaseRescorer:
    """Factory function for creating rescorer instances."""
    if method == "geometric":
        return GeometricRescorer(**kwargs)
    elif method == "cnn":
        return CNNRescorer(**kwargs)
    else:
        raise ValueError(f"Unknown rescorer method: {method}")
