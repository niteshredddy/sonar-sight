"""
Sonar image preprocessing module.

Implements CFAR-style adaptive thresholding and speckle suppression
for side-scan sonar imagery. Callable as both a standalone function
and an Albumentations-compatible training transform.

Methods:
    1. CA-CFAR (Cell-Averaging Constant False Alarm Rate): Adaptive
       local thresholding using sliding-window noise estimation
    2. Median filter fallback: Used when CFAR produces artifacts on
       very low-SNR inputs

Usage:
    from model.preprocessing import preprocess_sonar, CFARTransform

    # Standalone
    denoised = preprocess_sonar(raw_image)

    # As Albumentations transform
    pipeline = A.Compose([CFARTransform(p=1.0), ...])
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

import albumentations as A
import cv2
import numpy as np
from scipy.ndimage import median_filter, uniform_filter

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def ca_cfar_2d(
    image: np.ndarray,
    num_train: int = 20,
    num_guard: int = 4,
    pfa: float = 1e-3,
    clip_output: bool = True,
) -> np.ndarray:
    """
    2D Cell-Averaging CFAR (Constant False Alarm Rate) detector.

    Estimates local noise floor using a sliding window of training cells,
    excluding guard cells around the cell under test. Pixels exceeding
    the adaptive threshold are preserved; others are suppressed.

    Args:
        image: Grayscale input image (uint8 or float32)
        num_train: Number of training cells in each direction
        num_guard: Number of guard cells in each direction
        pfa: Target probability of false alarm (lower = stricter)
        clip_output: Whether to clip output to [0, 255]

    Returns:
        Filtered image with background suppressed (same shape as input)
    """
    img = image.astype(np.float64)

    # Total window sizes
    outer_size = 2 * (num_train + num_guard) + 1
    inner_size = 2 * num_guard + 1

    # Compute sum over outer and inner windows using uniform_filter
    outer_sum = uniform_filter(img, size=outer_size, mode='reflect') * (outer_size ** 2)
    inner_sum = uniform_filter(img, size=inner_size, mode='reflect') * (inner_size ** 2)

    # Training cell sum = outer - inner
    num_training_cells = outer_size ** 2 - inner_size ** 2
    training_sum = outer_sum - inner_sum

    # Noise floor estimate (mean of training cells)
    noise_floor = training_sum / max(num_training_cells, 1)

    # Threshold scaling factor from Pfa
    # For CA-CFAR: alpha = N * (Pfa^(-1/N) - 1), where N = num training cells
    alpha = num_training_cells * (pfa ** (-1.0 / num_training_cells) - 1)

    # Adaptive threshold
    threshold = alpha * noise_floor

    # Suppress sub-threshold pixels, preserve super-threshold
    result = np.where(img > threshold, img, img * 0.3)

    if clip_output:
        result = np.clip(result, 0, 255)

    return result.astype(np.uint8) if image.dtype == np.uint8 else result


def adaptive_median_denoise(
    image: np.ndarray,
    kernel_size: int = 5,
    strength: float = 0.7,
) -> np.ndarray:
    """
    Median filter-based denoising fallback for low-SNR sonar images.

    Blends the median-filtered result with the original to preserve
    some texture while reducing speckle.

    Args:
        image: Input image
        kernel_size: Median filter kernel size (must be odd)
        strength: Blending strength (0=original, 1=fully filtered)

    Returns:
        Denoised image
    """
    filtered = median_filter(image, size=kernel_size).astype(np.float32)
    original = image.astype(np.float32)
    blended = original * (1 - strength) + filtered * strength
    return np.clip(blended, 0, 255).astype(np.uint8)


def estimate_snr(image: np.ndarray) -> float:
    """
    Estimate Signal-to-Noise Ratio of a sonar image.

    Uses the ratio of mean to standard deviation as a simple SNR proxy.
    Low SNR (<3) indicates the image is very noisy.
    """
    img = image.astype(np.float64)
    if len(img.shape) == 3:
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float64)
    std = np.std(img)
    if std < 1e-10:
        return float('inf')
    return float(np.mean(img) / std)


def preprocess_sonar(
    image: np.ndarray,
    method: str = "auto",
    cfar_num_train: int = 20,
    cfar_num_guard: int = 4,
    cfar_pfa: float = 1e-3,
    median_kernel: int = 5,
    median_strength: float = 0.7,
    snr_threshold: float = 3.0,
) -> np.ndarray:
    """
    Main preprocessing function for sonar imagery.

    Applies CFAR-style filtering to suppress speckle noise and enhance
    target returns. Falls back to median filtering for very low-SNR inputs
    where CFAR may produce artifacts.

    Args:
        image: Input sonar image (grayscale or BGR)
        method: "cfar", "median", or "auto" (chooses based on SNR)
        cfar_*: CFAR parameters
        median_*: Median filter parameters
        snr_threshold: SNR below which to use median fallback

    Returns:
        Preprocessed image (same dtype and shape as input)
    """
    is_color = len(image.shape) == 3
    if is_color:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    # Choose method
    if method == "auto":
        snr = estimate_snr(gray)
        method = "cfar" if snr >= snr_threshold else "median"

    if method == "cfar":
        processed = ca_cfar_2d(gray, cfar_num_train, cfar_num_guard, cfar_pfa)
    elif method == "median":
        processed = adaptive_median_denoise(gray, median_kernel, median_strength)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'cfar', 'median', or 'auto'.")

    # Convert back to color if input was color
    if is_color:
        # Apply the processing as a luminance adjustment
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
        ratio = np.where(gray > 0, processed.astype(np.float32) / gray.astype(np.float32), 1.0)
        hsv[:, :, 2] = np.clip(hsv[:, :, 2] * ratio, 0, 255)
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    return processed


def detect_and_mask_nadir(
    image: np.ndarray,
    nadir_width_ratio: float = 0.15,
) -> tuple[np.ndarray, tuple[int, int]]:
    """
    Detect the central vertical water-column return (nadir zone) in side-scan sonar.

    Side-scan sonar waterfall strips contain a dark vertical band down the center
    representing the water column before acoustic signals hit the seabed. Filtering
    this band prevents false positives along the nadir boundaries.

    Args:
        image: Grayscale or BGR image
        nadir_width_ratio: Max expected fraction of width for nadir zone (0.05 to 0.3)

    Returns:
        mask: Binary uint8 mask (0 in nadir zone, 255 elsewhere)
        (col_start, col_end): Column index bounds of nadir zone
    """
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image.copy()

    h, w = gray.shape
    col_means = np.mean(gray, axis=0)

    # Search in central 40% of image width
    search_margin = int(w * 0.3)
    center_region = col_means[search_margin : w - search_margin]

    if len(center_region) == 0:
        return np.ones((h, w), dtype=np.uint8) * 255, (0, 0)

    # Min intensity column near center
    min_col_idx = search_margin + int(np.argmin(center_region))
    min_val = col_means[min_col_idx]
    max_val = np.max(col_means)

    # Threshold for nadir column expansion (values close to minimum)
    threshold = min_val + 0.25 * (max_val - min_val)

    # Expand left and right from min column while intensity stays below threshold
    col_start = min_col_idx
    while col_start > 0 and col_means[col_start] <= threshold and (min_col_idx - col_start) < (w * nadir_width_ratio):
        col_start -= 1

    col_end = min_col_idx
    while col_end < w - 1 and col_means[col_end] <= threshold and (col_end - min_col_idx) < (w * nadir_width_ratio):
        col_end += 1

    mask = np.ones((h, w), dtype=np.uint8) * 255
    if col_end > col_start + 2:
        mask[:, col_start:col_end] = 0

    return mask, (col_start, col_end)


class CFARTransform(A.ImageOnlyTransform):
    """
    Albumentations-compatible CFAR preprocessing transform.

    Can be used in training augmentation pipelines to apply CFAR-style
    filtering as a pre-training step (normalizes input distribution).
    """

    def __init__(
        self,
        num_train: int = 20,
        num_guard: int = 4,
        pfa: float = 1e-3,
        method: str = "auto",
        always_apply=False,
        p=1.0,
    ):
        super().__init__(p=p)
        self.num_train = num_train
        self.num_guard = num_guard
        self.pfa = pfa
        self.method = method
        self.always_apply = always_apply

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        return preprocess_sonar(
            img,
            method=self.method,
            cfar_num_train=self.num_train,
            cfar_num_guard=self.num_guard,
            cfar_pfa=self.pfa,
        )

    def get_transform_init_args_names(self):
        return ("num_train", "num_guard", "pfa", "method")


def main():
    """CLI for testing preprocessing on a single image."""
    parser = argparse.ArgumentParser(description="Sonar preprocessing")
    parser.add_argument("input", help="Input image path")
    parser.add_argument("--output", help="Output image path")
    parser.add_argument("--method", default="auto", choices=["cfar", "median", "auto"])
    args = parser.parse_args()

    img = cv2.imread(args.input)
    if img is None:
        print(f"Error: Could not read {args.input}")
        sys.exit(1)

    result = preprocess_sonar(img, method=args.method)
    snr = estimate_snr(img)
    print(f"Input SNR: {snr:.2f} | Method: {args.method}")

    output = args.output or args.input.replace(".", f"_preprocessed.")
    cv2.imwrite(output, result)
    print(f"✅ Saved → {output}")


if __name__ == "__main__":
    main()
