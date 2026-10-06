"""
Core SAM2 video propagation logic.
"""

import os
import shutil
import tempfile
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple

import numpy as np
import torch
from PIL import Image

from .ledger import is_auto_generated, is_manual_seed, load_ledger, mark_as_auto, save_ledger
from .models import get_model_config
from .sequence import collect_sequences
from .utils import (
    load_labelme_json,
    make_labelme_shape,
    mask_to_polygon,
    polygon_to_mask,
    save_labelme_json,
    visualize_annotation,
)


def build_video_frames_dir(
    frames: List[Tuple[str, str]],
    indices: List[int],
    output_dir: str
) -> Tuple[int, int]:
    """Prepare sequential JPEG directory for SAM2 video predictor.

    SAM2 video predictor expects frames named 00000.jpg, 00001.jpg, etc.

    Args:
        frames: List of (stem, image_path) tuples
        indices: Frame indices to include
        output_dir: Directory to write numbered JPEGs

    Returns:
        Tuple of (width, height) from first frame
    """
    os.makedirs(output_dir, exist_ok=True)

    first_image = None
    for i, frame_idx in enumerate(indices):
        _, img_path = frames[frame_idx]
        img = Image.open(img_path).convert("RGB")

        if first_image is None:
            first_image = img

        output_path = os.path.join(output_dir, f"{i:05d}.jpg")
        img.save(output_path, quality=95)

    return first_image.size if first_image else (0, 0)


def propagate_from_seed(
    predictor,
    frames: List[Tuple[str, str]],
    seed_index: int,
    target_indices: List[int],
    seed_shapes: List[dict],
    height: int,
    width: int,
    min_area: int = 80,
    max_points: int = 200,
) -> Dict[int, List[dict]]:
    """Propagate annotations from seed frame to target frames.

    Args:
        predictor: SAM2 video predictor instance
        frames: List of (stem, path) tuples for the sequence
        seed_index: Index of seed frame
        target_indices: List of target frame indices (in propagation order)
        seed_shapes: List of shape dicts from seed frame
        height: Image height
        width: Image width
        min_area: Minimum mask area to keep
        max_points: Maximum polygon vertices

    Returns:
        Dict mapping frame_index -> list of propagated shapes
    """
    if not target_indices:
        return {}

    # Build frame order: seed first, then targets
    frame_order = [seed_index] + list(target_indices)

    # Create temporary directory with numbered frames
    temp_dir = tempfile.mkdtemp(prefix="sam2_frames_")

    try:
        build_video_frames_dir(frames, frame_order, temp_dir)

        # Initialize SAM2 video state
        state = predictor.init_state(
            video_path=temp_dir,
            offload_video_to_cpu=True,
            offload_state_to_cpu=True,
        )
        predictor.reset_state(state)

        # Add seed frame masks as conditioning
        object_metadata = {}
        for obj_id, shape in enumerate(seed_shapes, start=1):
            mask = polygon_to_mask(shape["points"], height, width)
            if mask.sum() < min_area:
                continue

            object_metadata[obj_id] = shape
            predictor.add_new_mask(state, frame_idx=0, obj_id=obj_id, mask=mask)

        if not object_metadata:
            return {}

        # Propagate through video
        results = {}
        for local_idx, obj_ids, logits in predictor.propagate_in_video(state):
            if local_idx == 0:  # Skip seed frame
                continue

            # Map back to original frame index
            frame_idx = frame_order[local_idx]
            shapes = []

            for i, obj_id in enumerate(obj_ids):
                if obj_id not in object_metadata:
                    continue

                # Convert logits to binary mask
                mask = (logits[i] > 0.0).cpu().numpy().squeeze()

                # Convert mask back to polygon
                polygon = mask_to_polygon(mask, min_area=min_area, max_points=max_points)
                if polygon is None:
                    continue

                # Create shape with same metadata as seed
                source_shape = object_metadata[obj_id]
                shapes.append(
                    make_labelme_shape(
                        label=source_shape["label"],
                        points=polygon,
                        group_id=source_shape.get("group_id"),
                        description=source_shape.get("description", ""),
                    )
                )

            results[frame_idx] = shapes

        return results

    finally:
        shutil.rmtree(temp_dir, ignore_errors=True)


def process_sequence(
    predictor,
    sequence_name: str,
    frames: List[Tuple[str, str]],
    data_dir: str,
    static_labels: Set[str],
    overwrite: bool,
    trust_auto: bool,
    forward_only: bool,
    preview: bool,
    embed_image: bool,
    min_area: int,
    max_points: int,
    ledger: Dict[str, str],
) -> int:
    """Process one camera sequence.

    Args:
        predictor: SAM2 video predictor
        sequence_name: Camera prefix (e.g., "N_camera_0_link")
        frames: List of (stem, path) tuples
        data_dir: Base directory containing annotations
        static_labels: Set of labels to copy instead of track
        overwrite: Whether to overwrite manual annotations
        trust_auto: Treat auto results as seeds
        forward_only: Only forward propagation
        preview: Generate preview images
        embed_image: Embed image data in JSON
        min_area: Minimum mask area
        max_points: Maximum polygon points
        ledger: Ledger dict (modified in-place)

    Returns:
        Number of files written
    """
    n_frames = len(frames)

    # Classify frames
    seed_indices = []
    empty_indices = []
    auto_count = 0
    locked_count = 0

    for i, (stem, _) in enumerate(frames):
        json_path = os.path.join(data_dir, stem + ".json")

        if not os.path.exists(json_path):
            empty_indices.append(i)
        elif is_auto_generated(json_path, ledger) and not trust_auto:
            auto_count += 1
            empty_indices.append(i)  # Regenerate
        elif is_manual_seed(json_path, ledger):
            seed_indices.append(i)
        elif overwrite:
            empty_indices.append(i)
        else:
            locked_count += 1  # Empty but don't overwrite

    print(
        f"\n[{sequence_name}] {n_frames} frames: "
        f"{len(seed_indices)} seeds, {len(empty_indices)} to generate"
    )
    if auto_count > 0:
        print(f"  ({auto_count} auto results will be regenerated)")
    if locked_count > 0:
        print(f"  (skipping {locked_count} empty JSONs; use --overwrite to regenerate)")

    if not seed_indices:
        print(f"[{sequence_name}] No seed frames found. Annotate at least one frame manually.")
        return 0

    if not empty_indices:
        print(f"[{sequence_name}] All frames already annotated.")
        return 0

    # Get image dimensions from first frame
    probe = Image.open(frames[0][1])
    width, height = probe.size

    # Assign empty frames to nearest seed with direction
    forward_tasks = defaultdict(list)  # seed_idx -> [target_idx] (ascending)
    backward_tasks = defaultdict(list)  # seed_idx -> [target_idx] (descending)

    for empty_idx in empty_indices:
        prev_seed = max([s for s in seed_indices if s < empty_idx], default=None)
        next_seed = min([s for s in seed_indices if s > empty_idx], default=None)

        if prev_seed is None:
            # Before first seed: backward from next seed
            backward_tasks[next_seed].append(empty_idx)
        elif next_seed is None or forward_only:
            # After last seed or forward-only mode
            forward_tasks[prev_seed].append(empty_idx)
        elif (empty_idx - prev_seed) <= (next_seed - empty_idx):
            # Closer to previous seed
            forward_tasks[prev_seed].append(empty_idx)
        else:
            # Closer to next seed
            backward_tasks[next_seed].append(empty_idx)

    # Sort targets
    for seed_idx in forward_tasks:
        forward_tasks[seed_idx].sort()
    for seed_idx in backward_tasks:
        backward_tasks[seed_idx].sort(reverse=True)

    # Process each seed
    written_count = 0
    all_seeds = sorted(set(list(forward_tasks.keys()) + list(backward_tasks.keys())))

    for seed_idx in all_seeds:
        stem, _ = frames[seed_idx]
        seed_json_path = os.path.join(data_dir, stem + ".json")
        seed_doc = load_labelme_json(seed_json_path)
        seed_shapes = seed_doc.get("shapes", [])

        # Split into tracked and static shapes
        tracked_shapes = [s for s in seed_shapes if s["label"] not in static_labels]
        static_shapes = [s for s in seed_shapes if s["label"] in static_labels]

        # Process forward and backward tasks
        for targets in [forward_tasks.get(seed_idx, []), backward_tasks.get(seed_idx, [])]:
            if not targets:
                continue

            # Propagate tracked shapes
            propagation_results = propagate_from_seed(
                predictor,
                frames,
                seed_idx,
                targets,
                tracked_shapes,
                height,
                width,
                min_area,
                max_points,
            )

            # Write results
            for target_idx in targets:
                shapes = list(propagation_results.get(target_idx, []))

                # Add static shapes (copied verbatim)
                for static_shape in static_shapes:
                    shapes.append(
                        make_labelme_shape(
                            label=static_shape["label"],
                            points=static_shape["points"],
                            group_id=static_shape.get("group_id"),
                            description=static_shape.get("description", ""),
                        )
                    )

                if not shapes:
                    continue

                target_stem, target_img_path = frames[target_idx]
                target_json_path = os.path.join(data_dir, target_stem + ".json")

                # Backup if overwriting manual annotation
                if os.path.exists(target_json_path) and not is_auto_generated(target_json_path, ledger):
                    if overwrite:
                        shutil.copy2(target_json_path, target_json_path + ".bak")
                    else:
                        continue

                # Write JSON
                save_labelme_json(
                    target_json_path,
                    target_img_path,
                    shapes,
                    height,
                    width,
                    embed_image=embed_image,
                )

                # Mark as auto-generated
                mark_as_auto(target_json_path, shapes, ledger)
                written_count += 1

                # Generate preview if requested
                if preview:
                    preview_path = os.path.join(data_dir, "_preview", target_stem + ".jpg")
                    visualize_annotation(target_img_path, shapes, preview_path)

            direction = "forward" if targets == forward_tasks.get(seed_idx, []) else "backward"
            print(f"  Seed {stem[-20:]} -> {len(targets)} frames ({direction})")

    return written_count


def run_propagation(
    data_dir: str,
    preview: bool = False,
    overwrite: bool = False,
    static_labels: Optional[List[str]] = None,
    model_size: str = "small",
    forward_only: bool = False,
    max_points: int = 200,
    checkpoint: Optional[str] = None,
    model_cfg: Optional[str] = None,
    embed_image: bool = True,
    min_area: int = 80,
    trust_auto: bool = False,
    sequence_filter: Optional[str] = None,
) -> int:
    """Main entry point for propagation.

    Args:
        data_dir: Directory containing images and annotations
        preview: Generate preview overlays
        overwrite: Overwrite existing annotations
        static_labels: List of labels to copy instead of track
        model_size: SAM2 model size
        forward_only: Only forward propagation
        max_points: Maximum polygon vertices
        checkpoint: Custom checkpoint path
        model_cfg: Custom config path
        embed_image: Embed image data in JSON
        min_area: Minimum mask area
        trust_auto: Treat auto results as seeds
        sequence_filter: Only process sequences matching this prefix

    Returns:
        Total number of annotations written
    """
    if not os.path.isdir(data_dir):
        raise ValueError(f"Directory not found: {data_dir}")

    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[env] torch {torch.__version__}  device={device}  model={model_size}")

    if device == "cuda":
        torch.autocast("cuda", dtype=torch.bfloat16).__enter__()
        if torch.cuda.get_device_properties(0).major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True

    # Load model
    from sam2.build_sam import build_sam2_video_predictor

    if checkpoint and model_cfg:
        config_path = model_cfg
        checkpoint_path = checkpoint
    else:
        config_path, checkpoint_path = get_model_config(model_size)

    predictor = build_sam2_video_predictor(config_path, checkpoint_path, device=device)

    # Collect sequences
    sequences = collect_sequences(data_dir)
    print(f"[scan] Found {len(sequences)} sequences: {', '.join(sequences.keys())}")

    if sequence_filter:
        sequences = {k: v for k, v in sequences.items() if sequence_filter in k}
        if not sequences:
            raise ValueError(f"No sequences match filter: {sequence_filter}")
        print(f"[scan] Filtered to: {', '.join(sequences.keys())}")

    # Load ledger
    ledger = load_ledger(data_dir)
    static_set = set(static_labels or [])

    # Process each sequence
    total_written = 0
    try:
        for seq_name, frames in sorted(sequences.items()):
            written = process_sequence(
                predictor,
                seq_name,
                frames,
                data_dir,
                static_set,
                overwrite,
                trust_auto,
                forward_only,
                preview,
                embed_image,
                min_area,
                max_points,
                ledger,
            )
            total_written += written
    finally:
        save_ledger(data_dir, ledger)

    print(f"\n[complete] Wrote {total_written} annotations to {data_dir}")
    return total_written
