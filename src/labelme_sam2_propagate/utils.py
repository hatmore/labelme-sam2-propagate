"""
Utility functions for mask/polygon conversion and visualization.
"""

import base64
import hashlib
import json
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image, ImageDraw


def polygon_to_mask(points: List[List[float]], height: int, width: int) -> np.ndarray:
    """Convert polygon points to binary mask.

    Args:
        points: List of [x, y] coordinates
        height: Image height
        width: Image width

    Returns:
        Binary mask as boolean numpy array
    """
    im = Image.new("L", (width, height), 0)
    ImageDraw.Draw(im).polygon(
        [(float(x), float(y)) for x, y in points],
        outline=1,
        fill=1
    )
    return np.array(im, dtype=bool)


def mask_to_polygon(
    mask: np.ndarray,
    min_area: int = 80,
    max_points: int = 200,
    min_points: int = 8
) -> Optional[List[List[float]]]:
    """Convert binary mask to polygon using contour approximation.

    Uses binary search on epsilon to find the coarsest approximation
    that stays within max_points limit while preserving detail.

    Args:
        mask: Binary mask (H, W)
        min_area: Minimum contour area to keep
        max_points: Maximum polygon vertices
        min_points: Minimum polygon vertices

    Returns:
        List of [x, y] coordinates, or None if no valid contour
    """
    m = mask.astype(np.uint8)
    if m.sum() < min_area:
        return None

    # Morphological operations to clean up mask
    kernel = np.ones((3, 3), np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, kernel)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, kernel)

    # Find largest contour
    contours, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None

    contour = max(contours, key=cv2.contourArea)
    if cv2.contourArea(contour) < min_area:
        return None

    perimeter = cv2.arcLength(contour, True)

    # Binary search for optimal epsilon: smallest value that gives <= max_points
    lo, hi = 0.0, 0.05
    best = cv2.approxPolyDP(contour, hi * perimeter, True)

    for _ in range(30):
        mid = (lo + hi) / 2
        approx = cv2.approxPolyDP(contour, mid * perimeter, True)
        if len(approx) > max_points:
            lo = mid
        else:
            best = approx
            hi = mid

    # Handle extremely simple shapes
    if len(best) < min_points:
        fine = cv2.approxPolyDP(contour, 0.001 * perimeter, True)
        if min_points <= len(fine) <= max_points:
            best = fine

    points = best.reshape(-1, 2).astype(float)
    if len(points) < 3:
        return None

    return [[round(float(x), 1), round(float(y), 1)] for x, y in points]


def visualize_annotation(
    image_path: str,
    shapes: List[dict],
    output_path: str,
    alpha: float = 0.35
) -> None:
    """Create overlay visualization of annotations.

    Args:
        image_path: Path to source image
        shapes: List of LabelMe shape dicts
        output_path: Path to save visualization
        alpha: Overlay transparency (0-1)
    """
    img = cv2.imread(image_path)
    if img is None:
        return

    overlay = img.copy()

    # Color palette for different labels
    palette = [
        (0, 255, 0),      # green
        (255, 128, 0),    # orange
        (0, 128, 255),    # light blue
        (255, 0, 255),    # magenta
        (0, 255, 255),    # cyan
        (255, 255, 0),    # yellow
        (128, 0, 255),    # purple
        (0, 0, 255),      # blue
    ]

    labels = sorted({s["label"] for s in shapes})

    # Draw filled polygons on overlay
    for shape in shapes:
        color = palette[labels.index(shape["label"]) % len(palette)]
        pts = np.array(shape["points"], dtype=np.int32).reshape(-1, 1, 2)
        cv2.fillPoly(overlay, [pts], color)
        cv2.polylines(img, [pts], True, color, 2)

    # Blend overlay with original
    img = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    # Draw legend
    for i, label in enumerate(labels):
        color = palette[i % len(palette)]
        cv2.putText(
            img,
            label,
            (10, 30 + 26 * i),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            color,
            2
        )

    # Save output
    output_dir = os.path.dirname(output_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 85])


def shapes_hash(shapes: List[dict]) -> str:
    """Compute SHA1 hash of shapes for change detection.

    Args:
        shapes: List of LabelMe shape dicts

    Returns:
        40-character hex digest
    """
    return hashlib.sha1(
        json.dumps(shapes, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


def make_labelme_shape(
    label: str,
    points: List[List[float]],
    group_id: Optional[int] = None,
    description: str = ""
) -> dict:
    """Create a LabelMe shape dict.

    Args:
        label: Object class name
        points: List of [x, y] coordinates
        group_id: Optional group ID for instance grouping
        description: Optional text description

    Returns:
        LabelMe shape dictionary
    """
    return {
        "label": label,
        "points": points,
        "group_id": group_id,
        "description": description,
        "shape_type": "polygon",
        "flags": {},
        "mask": None,
    }


def load_labelme_json(json_path: str) -> dict:
    """Load LabelMe JSON file.

    Args:
        json_path: Path to .json file

    Returns:
        Parsed JSON dict
    """
    with open(json_path, encoding="utf-8") as f:
        return json.load(f)


def save_labelme_json(
    json_path: str,
    image_path: str,
    shapes: List[dict],
    height: int,
    width: int,
    embed_image: bool = True,
    version: str = "5.10.1"
) -> None:
    """Save LabelMe JSON file.

    Args:
        json_path: Path to save .json file
        image_path: Path to source image
        shapes: List of shape dicts
        height: Image height
        width: Image width
        embed_image: Whether to embed base64 image data
        version: LabelMe version string
    """
    image_data = None
    if embed_image:
        with open(image_path, "rb") as f:
            image_data = base64.b64encode(f.read()).decode("utf-8")

    doc = {
        "version": version,
        "flags": {},
        "shapes": shapes,
        "imagePath": os.path.basename(image_path),
        "imageData": image_data,
        "imageHeight": height,
        "imageWidth": width,
    }

    # Atomic write
    tmp = json_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(doc, f, ensure_ascii=False, indent=2)
    os.replace(tmp, json_path)
