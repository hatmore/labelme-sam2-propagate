#!/usr/bin/env python3
"""
Generate demo GIFs from preview images for README

This script creates three different GIF animations:
1. Full sequence: All frames at 400ms per frame
2. Fast view: Every 3rd frame at 300ms per frame
3. Key frames: 5 representative frames at 800ms per frame

Usage:
    python create_demo_gifs.py ../20260904_131/_preview
"""

import sys
from pathlib import Path
import cv2
import numpy as np
from PIL import Image


def create_gifs(preview_dir: str, output_dir: str = "docs"):
    """Generate three demo GIFs from preview images"""

    preview_path = Path(preview_dir)
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)

    # Get all preview images
    preview_files = sorted(list(preview_path.glob('P_camera_0_link_*.jpg')))

    if not preview_files:
        print(f"Error: No preview images found in {preview_dir}")
        print("Please run: labelme-sam2-propagate --dir <data_dir> --preview")
        return

    print(f"Found {len(preview_files)} preview images")

    # Read first frame to get dimensions
    first = cv2.imread(str(preview_files[0]))
    if first is None:
        print(f"Error: Could not read {preview_files[0]}")
        return

    h, w = first.shape[:2]
    print(f"Original size: {w}x{h}")

    # Resize for GitHub display (640px wide)
    target_w = 640
    target_h = int(h * target_w / w)
    print(f"Target size: {target_w}x{target_h}")

    # Option 1: Full sequence
    print("\nGenerating Option 1: Full sequence...")
    frames1 = []
    for idx, f in enumerate(preview_files):
        img = cv2.imread(str(f))
        img = cv2.resize(img, (target_w, target_h))
        # Add frame counter
        cv2.putText(img, f'Frame {idx+1}/{len(preview_files)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        frames1.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    pil_frames1 = [Image.fromarray(f) for f in frames1]
    output_file1 = output_path / "demo_full.gif"
    pil_frames1[0].save(
        output_file1,
        save_all=True,
        append_images=pil_frames1[1:],
        duration=400,  # 400ms per frame
        loop=0,
        optimize=False
    )
    print(f"[OK] Saved: {output_file1} ({len(frames1)} frames, ~{len(frames1)*0.4:.1f}s)")

    # Option 2: Fast view (every 3rd frame)
    print("\nGenerating Option 2: Fast view...")
    frames2 = []
    for i in range(0, len(preview_files), 3):
        img = cv2.imread(str(preview_files[i]))
        img = cv2.resize(img, (target_w, target_h))
        cv2.putText(img, f'Frame {i+1}/{len(preview_files)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        frames2.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    pil_frames2 = [Image.fromarray(f) for f in frames2]
    output_file2 = output_path / "demo_fast.gif"
    pil_frames2[0].save(
        output_file2,
        save_all=True,
        append_images=pil_frames2[1:],
        duration=300,  # 300ms per frame
        loop=0,
        optimize=False
    )
    print(f"[OK] Saved: {output_file2} ({len(frames2)} frames, ~{len(frames2)*0.3:.1f}s)")

    # Option 3: Key frames
    print("\nGenerating Option 3: Key frames...")
    key_indices = [
        0,
        len(preview_files) // 4,
        len(preview_files) // 2,
        3 * len(preview_files) // 4,
        len(preview_files) - 1
    ]

    frames3 = []
    for i in key_indices:
        img = cv2.imread(str(preview_files[i]))
        img = cv2.resize(img, (target_w, target_h))
        cv2.putText(img, f'Frame {i+1}/{len(preview_files)}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        frames3.append(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    pil_frames3 = [Image.fromarray(f) for f in frames3]
    output_file3 = output_path / "demo_key.gif"
    pil_frames3[0].save(
        output_file3,
        save_all=True,
        append_images=pil_frames3[1:],
        duration=800,  # 800ms per frame
        loop=0,
        optimize=False
    )
    print(f"[OK] Saved: {output_file3} ({len(frames3)} frames, ~{len(frames3)*0.8:.1f}s)")

    print("\n" + "="*50)
    print("All demo GIFs generated successfully!")
    print("="*50)
    print("\nRecommendation for README:")
    print(f"  - Use demo_fast.gif for quick overview (smaller file)")
    print(f"  - Use demo_full.gif for detailed documentation")
    print(f"  - Use demo_key.gif for minimal examples")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        preview_dir = sys.argv[1]
    else:
        # Default: look for preview directory in parent
        preview_dir = "../20260904_131/_preview"

    create_gifs(preview_dir)
