"""
Sonar image tiling module.

Splits large waterfall-style sonar strips into overlapping fixed-size tiles,
recording each tile's pixel offset in the original image as a JSON mapping
file. Remaps any YOLO-format labels into tile-local coordinates.

Usage:
    python model/tiling.py --input data/synthetic/train --output data/processed/train \\
                           --tile-size 640 --overlap 0.5
"""

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def compute_tile_positions(
    img_h: int, img_w: int, tile_size: int, overlap: float
) -> list[tuple[int, int]]:
    """
    Compute top-left (x, y) positions for overlapping tiles.

    Returns a list of (x_offset, y_offset) tuples covering the entire image.
    If the image is smaller than tile_size in either dimension, a single
    tile at (0, 0) is returned (padded during extraction).
    """
    stride = max(1, int(tile_size * (1 - overlap)))
    positions = []

    y_positions = list(range(0, max(1, img_h - tile_size + 1), stride))
    if not y_positions or y_positions[-1] + tile_size < img_h:
        y_positions.append(max(0, img_h - tile_size))

    x_positions = list(range(0, max(1, img_w - tile_size + 1), stride))
    if not x_positions or x_positions[-1] + tile_size < img_w:
        x_positions.append(max(0, img_w - tile_size))

    # Deduplicate while preserving order
    y_positions = list(dict.fromkeys(y_positions))
    x_positions = list(dict.fromkeys(x_positions))

    for y in y_positions:
        for x in x_positions:
            positions.append((x, y))

    return positions


def extract_tile(img: np.ndarray, x: int, y: int, tile_size: int) -> np.ndarray:
    """Extract a tile from the image, padding with zeros if needed."""
    h, w = img.shape[:2]
    tile = np.zeros((tile_size, tile_size) + img.shape[2:], dtype=img.dtype)

    src_x_end = min(x + tile_size, w)
    src_y_end = min(y + tile_size, h)
    tile_w = src_x_end - x
    tile_h = src_y_end - y

    tile[:tile_h, :tile_w] = img[y:src_y_end, x:src_x_end]
    return tile


def remap_yolo_labels(
    labels: list[str],
    x_offset: int,
    y_offset: int,
    tile_size: int,
    img_w: int,
    img_h: int,
    min_visibility: float = 0.15,
) -> list[str]:
    """
    Remap YOLO-seg labels from original image coords to tile-local coords.

    Labels in YOLO-seg format: class_id x1 y1 x2 y2 ... (normalized polygon)
    Filters out labels with < min_visibility fraction visible in the tile.
    """
    remapped = []
    for label_line in labels:
        parts = label_line.strip().split()
        if len(parts) < 5:
            continue

        class_id = parts[0]
        # Parse normalized polygon points
        coords = [float(v) for v in parts[1:]]
        if len(coords) % 2 != 0:
            continue

        # Convert to absolute pixel coords
        points = []
        for i in range(0, len(coords), 2):
            px = coords[i] * img_w
            py = coords[i + 1] * img_h
            points.append((px, py))

        # Compute bounding box of the polygon in original coords
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        obj_x_min, obj_x_max = min(xs), max(xs)
        obj_y_min, obj_y_max = min(ys), max(ys)
        obj_area = (obj_x_max - obj_x_min) * (obj_y_max - obj_y_min)

        if obj_area <= 0:
            continue

        # Compute intersection with tile
        tile_x_min = x_offset
        tile_y_min = y_offset
        tile_x_max = x_offset + tile_size
        tile_y_max = y_offset + tile_size

        inter_x_min = max(obj_x_min, tile_x_min)
        inter_y_min = max(obj_y_min, tile_y_min)
        inter_x_max = min(obj_x_max, tile_x_max)
        inter_y_max = min(obj_y_max, tile_y_max)

        if inter_x_min >= inter_x_max or inter_y_min >= inter_y_max:
            continue

        inter_area = (inter_x_max - inter_x_min) * (inter_y_max - inter_y_min)
        visibility = inter_area / obj_area

        if visibility < min_visibility:
            continue

        # Remap polygon points to tile-local normalized coords
        tile_points = []
        for px, py in points:
            local_x = (px - x_offset) / tile_size
            local_y = (py - y_offset) / tile_size
            # Clamp to [0, 1]
            local_x = max(0.0, min(1.0, local_x))
            local_y = max(0.0, min(1.0, local_y))
            tile_points.append(f"{local_x:.6f} {local_y:.6f}")

        remapped.append(f"{class_id} " + " ".join(tile_points))

    return remapped


def tile_image_with_labels(
    image_path: Path,
    label_path: Path | None,
    output_dir: Path,
    tile_size: int = 640,
    overlap: float = 0.5,
) -> dict:
    """
    Tile a single image and its labels.

    Returns a dict mapping tile filenames to their metadata.
    """
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"Could not read image: {image_path}")

    h, w = img.shape[:2]

    # Load labels if available
    labels = []
    if label_path and label_path.exists():
        with open(label_path) as f:
            labels = [line.strip() for line in f if line.strip()]

    positions = compute_tile_positions(h, w, tile_size, overlap)

    img_dir = output_dir / "images"
    lbl_dir = output_dir / "labels"
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    tile_mapping = {}
    stem = image_path.stem

    for idx, (x, y) in enumerate(positions):
        tile = extract_tile(img, x, y, tile_size)
        tile_name = f"{stem}_tile_{idx:04d}"

        cv2.imwrite(str(img_dir / f"{tile_name}.png"), tile)

        # Remap labels
        tile_labels = remap_yolo_labels(labels, x, y, tile_size, w, h)
        with open(lbl_dir / f"{tile_name}.txt", "w") as f:
            f.write("\n".join(tile_labels))

        tile_mapping[tile_name] = {
            "source_image": image_path.name,
            "x_offset": x,
            "y_offset": y,
            "tile_width": tile_size,
            "tile_height": tile_size,
            "source_width": w,
            "source_height": h,
        }

    return tile_mapping


def tile_dataset(
    input_dir: Path,
    output_dir: Path,
    tile_size: int = 640,
    overlap: float = 0.5,
):
    """Tile an entire dataset directory (images/ + labels/ subdirectories)."""
    images_dir = input_dir / "images"
    labels_dir = input_dir / "labels"

    if not images_dir.exists():
        # Try flat directory
        images_dir = input_dir
        labels_dir = input_dir

    image_files = sorted(
        list(images_dir.glob("*.png")) +
        list(images_dir.glob("*.jpg")) +
        list(images_dir.glob("*.jpeg"))
    )

    if not image_files:
        print(f"⚠️  No images found in {images_dir}")
        return

    all_mappings = {}

    for img_path in image_files:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.exists():
            label_path = None

        try:
            mapping = tile_image_with_labels(
                img_path, label_path, output_dir, tile_size, overlap
            )
            all_mappings.update(mapping)
        except Exception as e:
            print(f"⚠️  Error tiling {img_path.name}: {e}")

    # Save tile mapping
    mapping_path = output_dir / "tile_mapping.json"
    with open(mapping_path, "w") as f:
        json.dump(all_mappings, f, indent=2)

    print(f"✅ Tiled {len(image_files)} images → {len(all_mappings)} tiles")
    print(f"   Mapping saved → {mapping_path}")


def main():
    parser = argparse.ArgumentParser(description="Tile sonar images into overlapping patches")
    parser.add_argument("--input", type=str, required=True, help="Input directory with images/ and labels/")
    parser.add_argument("--output", type=str, required=True, help="Output directory")
    parser.add_argument("--tile-size", type=int, default=640, help="Tile size in pixels")
    parser.add_argument("--overlap", type=float, default=0.5, help="Overlap fraction (0-1)")
    args = parser.parse_args()

    tile_dataset(Path(args.input), Path(args.output), args.tile_size, args.overlap)


if __name__ == "__main__":
    main()
