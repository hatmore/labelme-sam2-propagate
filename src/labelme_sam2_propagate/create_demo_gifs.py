"""
Demo GIF generator - creates visualizations of propagation results.
"""

import os
import sys
from pathlib import Path
from typing import List

import cv2
import numpy as np
from PIL import Image


def load_images_from_preview(preview_dir: str, max_frames: int = None) -> List[np.ndarray]:
    """Load preview images in sorted order.

    Args:
        preview_dir: Path to _preview directory
        max_frames: Maximum number of frames to load (None = all)

    Returns:
        List of images as numpy arrays (BGR format)
    """
    preview_path = Path(preview_dir)
    if not preview_path.exists():
        raise FileNotFoundError(f"Preview directory not found: {preview_dir}")

    # Get all jpg files sorted by name
    image_files = sorted(preview_path.glob("*.jpg"))

    if max_frames:
        image_files = image_files[:max_frames]

    images = []
    for img_file in image_files:
        img = cv2.imread(str(img_file))
        if img is not None:
            images.append(img)

    return images


def create_gif(
    images: List[np.ndarray],
    output_path: str,
    fps: int = 5,
    loop: int = 0
) -> None:
    """Create animated GIF from images.

    Args:
        images: List of images (BGR numpy arrays)
        output_path: Path to save GIF
        fps: Frames per second
        loop: Number of loops (0 = infinite)
    """
    if not images:
        print("No images to create GIF")
        return

    # Convert BGR to RGB and create PIL images
    pil_images = []
    for img in images:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil_images.append(Image.fromarray(rgb))

    # Save as GIF
    duration = int(1000 / fps)  # milliseconds per frame
    pil_images[0].save(
        output_path,
        save_all=True,
        append_images=pil_images[1:],
        duration=duration,
        loop=loop,
        optimize=True
    )

    file_size = os.path.getsize(output_path) / 1e6
    print(f"Created {output_path} ({len(images)} frames, {file_size:.1f} MB)")


def create_demo_gifs(preview_dir: str, output_dir: str = ".") -> None:
    """Generate multiple demo GIFs from preview directory.

    Creates:
    - demo_full.gif: All frames at 5 fps
    - demo_fast.gif: Every 3rd frame at 8 fps
    - demo_key.gif: 5 evenly spaced key frames at 2 fps

    Args:
        preview_dir: Path to _preview directory
        output_dir: Directory to save GIFs
    """
    print(f"Loading preview images from {preview_dir}...")
    images = load_images_from_preview(preview_dir)

    if not images:
        print("No images found in preview directory")
        return

    print(f"Loaded {len(images)} images")

    # 1. Full sequence (all frames)
    create_gif(
        images,
        os.path.join(output_dir, "demo_full.gif"),
        fps=5,
        loop=0
    )

    # 2. Fast preview (every 3rd frame)
    fast_images = images[::3]
    create_gif(
        fast_images,
        os.path.join(output_dir, "demo_fast.gif"),
        fps=8,
        loop=0
    )

    # 3. Key frames (5 evenly spaced)
    n_keyframes = min(5, len(images))
    indices = np.linspace(0, len(images) - 1, n_keyframes, dtype=int)
    key_images = [images[i] for i in indices]
    create_gif(
        key_images,
        os.path.join(output_dir, "demo_key.gif"),
        fps=2,
        loop=0
    )

    print("\nDemo GIFs created successfully!")


def main():
    """CLI entry point."""
    if len(sys.argv) < 2:
        print("Usage: python create_demo_gifs.py <preview_dir> [output_dir]")
        print("\nExample:")
        print("  python create_demo_gifs.py 20260904_131/_preview")
        sys.exit(1)

    preview_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    create_demo_gifs(preview_dir, output_dir)


if __name__ == "__main__":
    main()
