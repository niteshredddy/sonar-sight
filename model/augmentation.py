"""
Sonar-specific data augmentation module.

Implements four sonar-domain augmentations that simulate real acquisition
artifacts in side-scan sonar imagery:

1. Speckle Noise — multiplicative Gaussian noise (I_out = I * (1 + N(0,σ)))
2. Acoustic Shadow — directional darkening trailing bright objects
3. Data Dropout Bands — horizontal/vertical stripe blanking (AUV heave/pitch/roll)
4. Resolution Variation — random downscale + upscale with different interpolations

All transforms are Albumentations-compatible (inherit from ImageOnlyTransform)
and can be composed into training pipelines.

Usage:
    python model/augmentation.py --preview   # Saves before/after to data/augment_preview/
"""

import argparse
import sys
from pathlib import Path

import albumentations as A
import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


class SpeckleNoise(A.ImageOnlyTransform):
    """
    Multiplicative Gaussian speckle noise typical of coherent imaging systems.
    I_out = I * (1 + N(0, sigma))
    """

    def __init__(self, sigma_range: tuple[float, float] = (0.05, 0.25), always_apply=False, p=0.5):
        super().__init__(p=p)
        self.sigma_range = sigma_range
        self.always_apply = always_apply

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        sigma = np.random.uniform(*self.sigma_range)
        noise = np.random.normal(1.0, sigma, img.shape).astype(np.float32)
        noisy = img.astype(np.float32) * noise
        return np.clip(noisy, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("sigma_range",)


class AcousticShadow(A.ImageOnlyTransform):
    """
    Simulates acoustic shadows by applying directional darkening below
    bright regions. In SSS, objects cast shadows away from the sonar,
    appearing as dark bands trailing highlight returns.
    """

    def __init__(
        self,
        shadow_intensity: tuple[float, float] = (0.3, 0.7),
        shadow_length: tuple[int, int] = (20, 80),
        brightness_threshold: int = 150,
        always_apply=False,
        p=0.4,
    ):
        super().__init__(p=p)
        self.shadow_intensity = shadow_intensity
        self.shadow_length = shadow_length
        self.brightness_threshold = brightness_threshold
        self.always_apply = always_apply

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img
        bright_mask = gray > self.brightness_threshold

        if not bright_mask.any():
            return img

        result = img.copy().astype(np.float32)
        intensity = np.random.uniform(*self.shadow_intensity)
        length = np.random.randint(*self.shadow_length)

        # For each bright row, darken pixels below
        rows, cols = np.where(bright_mask)
        for r, c in zip(rows, cols):
            end_r = min(r + length, img.shape[0])
            if len(img.shape) == 3:
                result[r + 1:end_r, c, :] *= (1 - intensity)
            else:
                result[r + 1:end_r, c] *= (1 - intensity)

        return np.clip(result, 0, 255).astype(np.uint8)

    def get_transform_init_args_names(self):
        return ("shadow_intensity", "shadow_length", "brightness_threshold")


class DataDropoutBands(A.ImageOnlyTransform):
    """
    Simulates data dropout bands caused by AUV heave, pitch, or roll
    during acquisition. Appears as horizontal or vertical stripes of
    missing (zero-valued) data in the waterfall image.
    """

    def __init__(
        self,
        num_bands: tuple[int, int] = (1, 5),
        band_width: tuple[int, int] = (2, 10),
        direction: str = "horizontal",  # "horizontal", "vertical", or "both"
        fill_value: int = 0,
        always_apply=False,
        p=0.4,
    ):
        super().__init__(p=p)
        self.num_bands = num_bands
        self.band_width = band_width
        self.direction = direction
        self.fill_value = fill_value
        self.always_apply = always_apply

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        result = img.copy()
        h, w = img.shape[:2]
        n_bands = np.random.randint(*self.num_bands)

        for _ in range(n_bands):
            bw = np.random.randint(*self.band_width)
            direction = self.direction
            if direction == "both":
                direction = np.random.choice(["horizontal", "vertical"])

            if direction == "horizontal":
                start = np.random.randint(0, max(1, h - bw))
                result[start:start + bw, :] = self.fill_value
            else:
                start = np.random.randint(0, max(1, w - bw))
                result[:, start:start + bw] = self.fill_value

        return result

    def get_transform_init_args_names(self):
        return ("num_bands", "band_width", "direction", "fill_value")


class ResolutionVariation(A.ImageOnlyTransform):
    """
    Simulates resolution variation by downscaling then upscaling the image
    with different interpolation methods. Mimics varying sonar frequency,
    range settings, or different sonar hardware.
    """

    def __init__(
        self,
        scale_range: tuple[float, float] = (0.25, 0.75),
        always_apply=False,
        p=0.3,
    ):
        super().__init__(p=p)
        self.scale_range = scale_range
        self.always_apply = always_apply

    def apply(self, img: np.ndarray, **params) -> np.ndarray:
        h, w = img.shape[:2]
        scale = np.random.uniform(*self.scale_range)
        new_h, new_w = max(1, int(h * scale)), max(1, int(w * scale))

        # Downscale with area interpolation (averaging), upscale with random method
        down = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
        up_interp = np.random.choice([cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_CUBIC])
        up = cv2.resize(down, (w, h), interpolation=up_interp)

        return up

    def get_transform_init_args_names(self):
        return ("scale_range",)


def get_sonar_augmentation_pipeline(p: float = 0.5) -> A.Compose:
    """
    Returns a composed augmentation pipeline with all sonar-specific transforms
    plus standard geometric augmentations suitable for sonar data.
    """
    return A.Compose([
        # Geometric (standard)
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.3),
        A.ShiftScaleRotate(
            shift_limit=0.05, scale_limit=0.1, rotate_limit=10,
            border_mode=cv2.BORDER_REFLECT_101, p=0.3
        ),

        # Sonar-specific
        SpeckleNoise(sigma_range=(0.05, 0.2), p=p),
        AcousticShadow(shadow_intensity=(0.3, 0.6), shadow_length=(15, 60), p=p * 0.8),
        DataDropoutBands(num_bands=(1, 4), band_width=(2, 8), direction="both", p=p * 0.6),
        ResolutionVariation(scale_range=(0.35, 0.75), p=p * 0.5),

        # Intensity (standard)
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.3),
        A.GaussianBlur(blur_limit=(3, 5), p=0.2),
    ])


def generate_preview(output_dir: Path, num_samples: int = 6):
    """Generate before/after augmentation preview images."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Try to load a real synthetic tile, or generate a test pattern
    synthetic_dir = PROJECT_ROOT / "data" / "synthetic" / "train" / "images"
    if synthetic_dir.exists():
        image_files = list(synthetic_dir.glob("*.png"))
    else:
        image_files = []

    pipeline = get_sonar_augmentation_pipeline(p=1.0)  # Force all transforms for preview

    for i in range(num_samples):
        if image_files:
            img = cv2.imread(str(image_files[i % len(image_files)]))
        else:
            # Generate a simple test pattern
            img = np.random.randint(40, 120, (640, 640, 3), dtype=np.uint8)
            # Add a bright rectangle as a synthetic object
            cv2.rectangle(img, (200, 200), (350, 280), (200, 200, 200), -1)

        # Apply augmentation
        augmented = pipeline(image=img)["image"]

        # Create side-by-side comparison
        divider = np.ones((img.shape[0], 4, 3), dtype=np.uint8) * 255
        comparison = np.hstack([img, divider, augmented])

        # Add labels
        cv2.putText(comparison, "Original", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(comparison, "Augmented", (img.shape[1] + 14, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        cv2.imwrite(str(output_dir / f"augment_preview_{i:02d}.png"), comparison)

    print(f"✅ Saved {num_samples} augmentation previews → {output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Sonar augmentation module")
    parser.add_argument("--preview", action="store_true", help="Generate preview images")
    parser.add_argument("--output", type=str,
                        default=str(PROJECT_ROOT / "data" / "augment_preview"),
                        help="Preview output directory")
    parser.add_argument("--count", type=int, default=6, help="Number of preview samples")
    args = parser.parse_args()

    if args.preview:
        generate_preview(Path(args.output), args.count)
    else:
        print("Use --preview to generate augmentation sample images.")
        print("Import get_sonar_augmentation_pipeline() for training pipelines.")


if __name__ == "__main__":
    main()
