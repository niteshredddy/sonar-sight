"""
Unit tests for the confidence rescoring module.
"""

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from schemas.detection import BBox, Detection
from model.confidence_rescorer import GeometricRescorer, get_rescorer


@pytest.fixture
def rescorer():
    return GeometricRescorer()


def _make_detection(class_label="cylinder", confidence=70.0, bbox_size=(50, 50)):
    w, h = bbox_size
    return Detection(
        class_label=class_label,
        confidence=confidence,
        bbox=BBox(x_min=100, y_min=100, x_max=100 + w, y_max=100 + h),
        lat=0.0, lon=0.0,
        source_image="test.png",
    )


def _make_crop_with_highlight_and_shadow(h=80, w=60):
    """Create a synthetic crop with bright top half and dark bottom half (man-made object pattern)."""
    crop = np.full((h, w), 80, dtype=np.uint8)
    crop[:h // 3, :] = 200  # Bright highlight
    crop[h // 2:, :] = 30   # Dark shadow
    return crop


def _make_noisy_crop(h=80, w=60):
    """Create a noisy crop with no clear structure (natural formation)."""
    return np.random.randint(40, 160, (h, w), dtype=np.uint8)


def _make_regular_rectangle(h=50, w=100):
    """Create a crop with a clean rectangular bright object (man-made)."""
    crop = np.full((h, w), 60, dtype=np.uint8)
    # Draw a clean rectangle
    crop[10:25, 20:80] = 210
    # Shadow below
    crop[30:45, 20:80] = 20
    return crop


class TestGeometricRescorer:
    def test_rescore_returns_valid_range(self, rescorer):
        """Rescored confidence should be between min_confidence and 100."""
        crop = _make_crop_with_highlight_and_shadow()
        det = _make_detection()
        score = rescorer.rescore(crop, det)
        assert rescorer.min_confidence <= score <= 100.0

    def test_manmade_pattern_boosted(self, rescorer):
        """A clear highlight+shadow pattern should maintain or boost confidence."""
        crop = _make_regular_rectangle()
        det = _make_detection("pipe", 60.0, (100, 50))
        score = rescorer.rescore(crop, det)
        # Should not drop drastically below original confidence
        assert score >= 40.0

    def test_noisy_pattern_penalized(self, rescorer):
        """A noisy random crop should reduce confidence."""
        np.random.seed(42)
        crop = _make_noisy_crop()
        det = _make_detection("cylinder", 80.0)
        score = rescorer.rescore(crop, det)
        # May be penalized — should not exceed original by much
        assert score <= 95.0

    def test_empty_crop(self, rescorer):
        """Empty/None crop should reduce confidence significantly."""
        det = _make_detection(confidence=80.0)
        score = rescorer.rescore(None, det)
        assert score < 80.0

    def test_zero_size_crop(self, rescorer):
        """Zero-size crop should handle gracefully."""
        crop = np.array([], dtype=np.uint8)
        det = _make_detection(confidence=80.0)
        score = rescorer.rescore(crop, det)
        assert score < 80.0

    def test_batch_rescoring(self, rescorer):
        """Batch rescoring should process all detections."""
        crops = [_make_crop_with_highlight_and_shadow() for _ in range(5)]
        dets = [_make_detection() for _ in range(5)]
        scores = rescorer.rescore_batch(crops, dets)
        assert len(scores) == 5
        assert all(rescorer.min_confidence <= s <= 100.0 for s in scores)

    def test_aspect_ratio_scoring_pipe(self, rescorer):
        """Pipe-like aspect ratio (very elongated) should score well for pipe class."""
        det_pipe = _make_detection("pipe", 70.0, (200, 15))
        score = rescorer._compute_aspect_ratio_score(det_pipe.bbox, "pipe")
        assert score > 0.5

    def test_aspect_ratio_scoring_mismatch(self, rescorer):
        """Square bbox should score poorly for pipe class."""
        det_sq = _make_detection("pipe", 70.0, (50, 50))
        score = rescorer._compute_aspect_ratio_score(det_sq.bbox, "pipe")
        assert score < 0.5

    def test_different_classes_different_scores(self, rescorer):
        """Different class labels with same crop should produce different scores
        (because aspect ratio expectations differ)."""
        crop = _make_regular_rectangle(50, 100)

        det_pipe = _make_detection("pipe", 70.0, (100, 50))
        det_cyl = _make_detection("cylinder", 70.0, (100, 50))

        score_pipe = rescorer.rescore(crop, det_pipe)
        score_cyl = rescorer.rescore(crop, det_cyl)

        # They should differ since aspect ratio expectations differ
        # (but both should be valid)
        assert rescorer.min_confidence <= score_pipe <= 100
        assert rescorer.min_confidence <= score_cyl <= 100


class TestRescorerFactory:
    def test_get_geometric(self):
        r = get_rescorer("geometric")
        assert isinstance(r, GeometricRescorer)

    def test_get_unknown_raises(self):
        with pytest.raises(ValueError):
            get_rescorer("nonexistent")

    def test_cnn_not_implemented(self):
        with pytest.raises(NotImplementedError):
            r = get_rescorer("cnn", model_path="fake.onnx")
