"""
Generate the demo before/after composite image.

Creates a 3-panel horizontal composite:
    1. Raw noisy sonar tile
    2. After CFAR preprocessing (denoised)
    3. With final detections, masks, and confidence scores overlaid

Output: /reports/demo_before_after.png

Usage:
    python reports/generate_demo_image.py
"""

import sys
from pathlib import Path

import cv2
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from model.preprocessing import preprocess_sonar
from model.inference import SonarInferencePipeline


CLASS_COLORS = {
    "ghost_net": (16, 185, 129),   # Green
    "shipwreck": (245, 158, 11),   # Amber
    "pipe": (59, 130, 246),        # Blue
    "cylinder": (239, 68, 68),     # Red
}

CLASS_ICONS = {
    "ghost_net": "Net",
    "shipwreck": "Wreck",
    "pipe": "Pipe",
    "cylinder": "Cyl",
}


def generate_demo_image(output_path: Path):
    """Generate the 3-panel before/after demo composite."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    size = 640

    # Generate a realistic synthetic sonar tile
    rng = np.random.default_rng(42)

    # Base seabed texture
    img = np.zeros((size, size), dtype=np.float32)
    for octave, amp in [(64, 0.4), (32, 0.25), (16, 0.15), (8, 0.1)]:
        noise = rng.normal(0, 1, (size // octave + 1, size // octave + 1)).astype(np.float32)
        noise_up = cv2.resize(noise, (size, size), interpolation=cv2.INTER_CUBIC)
        img += amp * noise_up
    img = cv2.normalize(img, None, 30, 110, cv2.NORM_MINMAX)

    # Range gradient
    gradient = np.linspace(1.0, 0.6, size).reshape(1, -1).astype(np.float32)
    img *= gradient

    # Add synthetic objects
    # Shipwreck (large rectangular highlight + shadow)
    cv2.rectangle(img, (180, 250), (340, 300), 210, -1)
    cv2.rectangle(img, (180, 310), (340, 360), 25, -1)

    # Pipe (linear feature)
    cv2.line(img, (400, 100), (580, 130), 200, 4)
    cv2.line(img, (400, 115), (580, 145), 20, 3)

    # Cylinder (small bright ellipse + shadow)
    cv2.ellipse(img, (120, 450), (20, 12), 0, 0, 360, 220, -1)
    pts = np.array([[100, 465], [140, 465], [120, 510]], np.int32)
    cv2.fillPoly(img, [pts], 20)

    img = np.clip(img, 0, 255).astype(np.uint8)

    # Add heavy speckle noise
    speckle = rng.normal(1.0, 0.18, img.shape).astype(np.float32)
    raw = np.clip(img.astype(np.float32) * speckle, 0, 255).astype(np.uint8)

    # Panel 1: Raw noisy tile
    panel1 = cv2.cvtColor(raw, cv2.COLOR_GRAY2BGR)

    # Panel 2: After CFAR preprocessing
    denoised = preprocess_sonar(raw, method="cfar")
    panel2 = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)

    # Panel 3: With detections overlaid
    pipeline = SonarInferencePipeline(model_path=None)
    panel3_input = cv2.cvtColor(denoised, cv2.COLOR_GRAY2BGR)
    detections, _ = pipeline.run_on_array(panel3_input)
    panel3 = panel3_input.copy()

    for det in detections:
        color = CLASS_COLORS.get(det.class_label, (255, 255, 255))
        bbox = det.bbox
        x1, y1, x2, y2 = int(bbox.x_min), int(bbox.y_min), int(bbox.x_max), int(bbox.y_max)

        # Draw mask polygon
        if det.mask and len(det.mask) > 2:
            mask_pts = np.array([(int(p[0]), int(p[1])) for p in det.mask], np.int32)
            overlay = panel3.copy()
            cv2.fillPoly(overlay, [mask_pts], color)
            panel3 = cv2.addWeighted(panel3, 0.8, overlay, 0.2, 0)

        # Draw bbox
        cv2.rectangle(panel3, (x1, y1), (x2, y2), color, 2)

        # Label
        label = f"{CLASS_ICONS.get(det.class_label, '?')} {det.confidence:.0f}%"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(panel3, (x1, y1 - th - 8), (x1 + tw + 8, y1), color, -1)
        cv2.putText(panel3, label, (x1 + 4, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    # Assemble composite
    bar_width = 3
    bar = np.ones((size, bar_width, 3), dtype=np.uint8) * 40

    composite = np.hstack([panel1, bar, panel2, bar, panel3])

    # Add panel labels at bottom
    label_h = 40
    label_bar = np.zeros((label_h, composite.shape[1], 3), dtype=np.uint8)
    label_bar[:, :, :] = 15  # Dark background

    texts = [
        ("Raw Sonar (with speckle noise)", size // 2),
        ("After CFAR Preprocessing", size + bar_width + size // 2),
        ("Detections + Confidence Scores", 2 * (size + bar_width) + size // 2),
    ]
    for text, cx in texts:
        (tw, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
        cv2.putText(label_bar, text, (cx - tw // 2, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

    final = np.vstack([composite, label_bar])

    cv2.imwrite(str(output_path), final)
    print(f"[OK] Demo composite saved -> {output_path}")
    print(f"   Size: {final.shape[1]}x{final.shape[0]} px")
    print(f"   Detections shown: {len(detections)}")


if __name__ == "__main__":
    output = PROJECT_ROOT / "reports" / "demo_before_after.png"
    generate_demo_image(output)
