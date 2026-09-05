"""
Synthetic side-scan sonar image generator.

Generates procedural sonar tiles with man-made debris patterns for training
when real annotated data is unavailable. Produces YOLO-seg format labels.

Classes:
    0 = ghost_net   — diffuse bright returns, irregular trailing shadow
    1 = shipwreck   — large structured highlight + deep shadow
    2 = pipe        — linear bright return, parallel shadow
    3 = cylinder    — compact bright spot, sharp conical shadow

Usage:
    python data/synthetic/generate_synthetic.py [--count 200] [--size 640] [--output data/synthetic]
"""

import argparse
import json
import os
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

CLASS_NAMES = ["ghost_net", "shipwreck", "pipe", "cylinder"]
CLASS_COLORS = {
    "ghost_net": (180, 220, 200),
    "shipwreck": (240, 240, 220),
    "pipe": (200, 210, 230),
    "cylinder": (230, 230, 240),
}


def generate_seabed_texture(size: int, rng: np.random.Generator) -> np.ndarray:
    """Generate realistic seabed background texture using multi-scale noise."""
    img = np.zeros((size, size), dtype=np.float32)

    # Multi-scale Perlin-like noise via summed octaves of Gaussian-blurred noise
    for octave, amplitude in [(64, 0.4), (32, 0.25), (16, 0.15), (8, 0.1), (4, 0.1)]:
        noise = rng.normal(0, 1, (size // octave + 1, size // octave + 1)).astype(np.float32)
        noise_upscaled = cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)
        img += amplitude * noise_upscaled

    # Normalize to 30-120 range (dark seabed with variation)
    img = cv2.normalize(img, None, 30, 120, cv2.NORM_MINMAX)

    # Along-track gradient (brighter near nadir, darker at range)
    gradient = np.linspace(1.0, 0.6, size).reshape(1, -1).astype(np.float32)
    img *= gradient

    return np.clip(img, 0, 255).astype(np.uint8)


def add_speckle_noise(img: np.ndarray, rng: np.random.Generator, sigma: float = 0.15) -> np.ndarray:
    """Add multiplicative speckle noise typical of sonar imagery."""
    noise = rng.normal(1.0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) * noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def draw_ghost_net(img: np.ndarray, rng: np.random.Generator, size: int):
    """Draw a ghost net: diffuse bright returns with irregular shadow."""
    cx = rng.integers(size // 4, 3 * size // 4)
    cy = rng.integers(size // 4, 3 * size // 4)
    w = rng.integers(40, 120)
    h = rng.integers(30, 80)

    # Bright fibrous highlight — irregular ellipse with noise
    mask = np.zeros((size, size), dtype=np.uint8)
    pts = []
    for angle in np.linspace(0, 2 * np.pi, 20, endpoint=False):
        r_w = w // 2 + rng.integers(-10, 10)
        r_h = h // 2 + rng.integers(-8, 8)
        px = int(cx + r_w * np.cos(angle))
        py = int(cy + r_h * np.sin(angle))
        pts.append([px, py])
    pts = np.array(pts, dtype=np.int32)
    cv2.fillPoly(mask, [pts], 255)

    # Bright highlight
    brightness = rng.integers(160, 220)
    img[mask > 0] = np.clip(img[mask > 0].astype(np.int32) + brightness - 80, 0, 255).astype(np.uint8)

    # Trailing acoustic shadow (below the object)
    shadow_h = rng.integers(20, 50)
    shadow_mask = np.zeros_like(mask)
    shadow_pts = pts.copy()
    shadow_pts[:, 1] += h // 2 + 5
    shadow_pts_bottom = shadow_pts.copy()
    shadow_pts_bottom[:, 1] += shadow_h
    all_shadow = np.vstack([shadow_pts, shadow_pts_bottom[::-1]])
    cv2.fillPoly(shadow_mask, [all_shadow], 255)
    img[shadow_mask > 0] = np.clip(img[shadow_mask > 0].astype(np.int32) - 40, 0, 255).astype(np.uint8)

    # Return bbox and polygon for label
    x_min = max(0, int(pts[:, 0].min()))
    y_min = max(0, int(pts[:, 1].min()))
    x_max = min(size - 1, int(pts[:, 0].max()))
    y_max = min(size - 1, int(pts[:, 1].max()))
    polygon = [(float(p[0]) / size, float(p[1]) / size) for p in pts]

    return 0, (x_min, y_min, x_max, y_max), polygon


def draw_shipwreck(img: np.ndarray, rng: np.random.Generator, size: int):
    """Draw a shipwreck: large rectangular highlight with deep shadow."""
    cx = rng.integers(size // 4, 3 * size // 4)
    cy = rng.integers(size // 4, 3 * size // 4)
    w = rng.integers(80, 180)
    h = rng.integers(30, 70)
    angle = rng.integers(-30, 30)

    # Rotated rectangle for hull
    rect = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(np.int32)
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [rect], 255)

    brightness = rng.integers(180, 240)
    img[mask > 0] = np.clip(img[mask > 0].astype(np.int32) + brightness - 60, 0, 255).astype(np.uint8)

    # Deep shadow
    shadow_offset = rng.integers(10, 30)
    shadow_rect = rect.copy()
    shadow_rect[:, 1] += h // 2 + shadow_offset
    shadow_mask = np.zeros_like(mask)
    cv2.fillPoly(shadow_mask, [shadow_rect], 255)
    img[shadow_mask > 0] = np.clip(img[shadow_mask > 0].astype(np.int32) - 60, 0, 255).astype(np.uint8)

    x_min = max(0, int(rect[:, 0].min()))
    y_min = max(0, int(rect[:, 1].min()))
    x_max = min(size - 1, int(rect[:, 0].max()))
    y_max = min(size - 1, int(rect[:, 1].max()))
    polygon = [(float(p[0]) / size, float(p[1]) / size) for p in rect]

    return 1, (x_min, y_min, x_max, y_max), polygon


def draw_pipe(img: np.ndarray, rng: np.random.Generator, size: int):
    """Draw a pipe: long linear bright return with parallel shadow."""
    y_start = rng.integers(size // 4, 3 * size // 4)
    x_start = rng.integers(0, size // 4)
    length = rng.integers(200, min(500, size - x_start))
    thickness = rng.integers(6, 16)
    angle = rng.uniform(-0.2, 0.2)  # slight angle from horizontal

    pts = []
    for i in range(4):
        if i == 0:
            pts.append([x_start, int(y_start - thickness // 2)])
        elif i == 1:
            pts.append([x_start + length, int(y_start - thickness // 2 + length * np.sin(angle))])
        elif i == 2:
            pts.append([x_start + length, int(y_start + thickness // 2 + length * np.sin(angle))])
        else:
            pts.append([x_start, int(y_start + thickness // 2)])
    pts = np.array(pts, dtype=np.int32)

    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.fillPoly(mask, [pts], 255)
    brightness = rng.integers(170, 230)
    img[mask > 0] = np.clip(brightness, 0, 255).astype(np.uint8)

    # Parallel shadow
    shadow_pts = pts.copy()
    shadow_pts[:, 1] += thickness + rng.integers(5, 15)
    shadow_mask = np.zeros_like(mask)
    cv2.fillPoly(shadow_mask, [shadow_pts], 255)
    img[shadow_mask > 0] = np.clip(img[shadow_mask > 0].astype(np.int32) - 50, 0, 255).astype(np.uint8)

    pts_clipped = np.clip(pts, 0, size - 1)
    x_min, y_min = pts_clipped[:, 0].min(), pts_clipped[:, 1].min()
    x_max, y_max = pts_clipped[:, 0].max(), pts_clipped[:, 1].max()
    polygon = [(float(p[0]) / size, float(p[1]) / size) for p in pts]

    return 2, (int(x_min), int(y_min), int(x_max), int(y_max)), polygon


def draw_cylinder(img: np.ndarray, rng: np.random.Generator, size: int):
    """Draw a cylinder: compact bright ellipse with sharp conical shadow."""
    cx = rng.integers(size // 4, 3 * size // 4)
    cy = rng.integers(size // 4, 3 * size // 4)
    w = rng.integers(15, 40)
    h = rng.integers(10, 25)

    # Bright elliptical highlight
    mask = np.zeros((size, size), dtype=np.uint8)
    cv2.ellipse(mask, (cx, cy), (w, h), 0, 0, 360, 255, -1)
    brightness = rng.integers(190, 250)
    img[mask > 0] = np.clip(brightness, 0, 255).astype(np.uint8)

    # Sharp conical shadow
    shadow_len = rng.integers(25, 60)
    shadow_pts = np.array([
        [cx - w // 2, cy + h + 3],
        [cx + w // 2, cy + h + 3],
        [cx, cy + h + shadow_len],
    ], dtype=np.int32)
    shadow_mask = np.zeros_like(mask)
    cv2.fillPoly(shadow_mask, [shadow_pts], 255)
    img[shadow_mask > 0] = np.clip(img[shadow_mask > 0].astype(np.int32) - 55, 0, 255).astype(np.uint8)

    x_min = max(0, cx - w)
    y_min = max(0, cy - h)
    x_max = min(size - 1, cx + w)
    y_max = min(size - 1, cy + h)

    # Polygon approximation of ellipse
    angles = np.linspace(0, 2 * np.pi, 16, endpoint=False)
    polygon = [(float(cx + w * np.cos(a)) / size, float(cy + h * np.sin(a)) / size) for a in angles]

    return 3, (x_min, y_min, x_max, y_max), polygon


DRAW_FUNCTIONS = [draw_ghost_net, draw_shipwreck, draw_pipe, draw_cylinder]


def generate_single_tile(
    tile_id: int,
    size: int,
    rng: np.random.Generator,
    max_objects: int = 3,
) -> tuple[np.ndarray, list[str]]:
    """Generate one synthetic sonar tile with random debris objects."""
    img = generate_seabed_texture(size, rng)
    labels = []

    num_objects = rng.integers(1, max_objects + 1)
    for _ in range(num_objects):
        draw_fn = rng.choice(DRAW_FUNCTIONS)
        try:
            class_id, (x_min, y_min, x_max, y_max), polygon = draw_fn(img, rng, size)

            # YOLO-seg format: class_id x1 y1 x2 y2 ... (normalized polygon)
            poly_str = " ".join(f"{x:.6f} {y:.6f}" for x, y in polygon)
            labels.append(f"{class_id} {poly_str}")
        except Exception:
            continue  # Skip malformed objects

    # Add speckle noise
    img = add_speckle_noise(img, rng, sigma=0.12)

    return img, labels


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic sonar training tiles")
    parser.add_argument("--count", type=int, default=200, help="Number of tiles to generate")
    parser.add_argument("--size", type=int, default=640, help="Tile size in pixels")
    parser.add_argument("--output", type=str, default=str(PROJECT_ROOT / "data" / "synthetic"),
                        help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--val-split", type=float, default=0.15, help="Validation split ratio")
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    output = Path(args.output)

    # Create YOLO dataset structure
    for split in ["train", "val"]:
        (output / split / "images").mkdir(parents=True, exist_ok=True)
        (output / split / "labels").mkdir(parents=True, exist_ok=True)

    val_count = int(args.count * args.val_split)
    train_count = args.count - val_count

    stats = {name: 0 for name in CLASS_NAMES}

    for i in range(args.count):
        split = "val" if i >= train_count else "train"
        img, labels = generate_single_tile(i, args.size, rng)

        fname = f"sonar_{i:04d}"
        cv2.imwrite(str(output / split / "images" / f"{fname}.png"), img)

        with open(output / split / "labels" / f"{fname}.txt", "w") as f:
            f.write("\n".join(labels))

        # Count class distribution
        for label in labels:
            cls_id = int(label.split()[0])
            stats[CLASS_NAMES[cls_id]] += 1

    # Write dataset YAML for YOLO training
    yaml_content = f"""# Auto-generated sonar dataset config
path: {output.resolve()}
train: train/images
val: val/images

names:
  0: ghost_net
  1: shipwreck
  2: pipe
  3: cylinder

nc: 4
"""
    with open(output / "sonar_data.yaml", "w") as f:
        f.write(yaml_content)

    # Write stats
    print(f"✅ Generated {args.count} synthetic sonar tiles → {output}")
    print(f"   Train: {train_count}, Val: {val_count}")
    print(f"   Class distribution: {json.dumps(stats, indent=2)}")


if __name__ == "__main__":
    main()
