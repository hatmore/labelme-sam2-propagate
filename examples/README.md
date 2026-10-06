# Examples

This directory contains example workflows and use cases for labelme-sam2-propagate.

## Example 1: Basic Video Annotation

**Scenario:** 30-frame sequence of a person walking.

```bash
# 1. Annotate frame 0 and frame 15 in LabelMe
# 2. Run propagation
labelme-sam2-propagate --dir person_walking --preview

# 3. Review _preview/ directory
# 4. Fix any bad frames
# 5. Re-run (your fixes become seeds automatically)
```

## Example 2: Warehouse Logistics

**Scenario:** Forklift loading containers with static background.

```bash
# Optimize for static objects
labelme-sam2-propagate --dir warehouse_loading \
    --preview \
    --static truck_wall,truck_roof,truck_floor,dock_board,warehouse_wall
```

**Labels:**
- `forklift` - moving object (tracked)
- `cargo` - moving object (tracked)
- `truck_wall`, `truck_floor`, `truck_roof` - static (copied)
- `warehouse_wall`, `dock_board` - static (copied)

## Example 3: Multi-Camera Setup

**Scenario:** Two cameras in one directory.

```
data/
  N_camera_0_link_1788425812_781000000.jpg
  N_camera_0_link_1788425813_281000000.jpg
  P_camera_0_link_1788425812_781000000.jpg
  P_camera_0_link_1788425813_281000000.jpg
```

```bash
# Process both cameras
labelme-sam2-propagate --dir data --preview

# Or process one at a time
labelme-sam2-propagate --dir data --only N_camera_0_link
labelme-sam2-propagate --dir data --only P_camera_0_link
```

## Example 4: Low VRAM Setup

**Scenario:** GPU with only 2GB VRAM.

```bash
# Use tiny model
labelme-sam2-propagate --dir video_sequence --model tiny --preview
```

## Example 5: High-Quality Mode

**Scenario:** Research project requiring maximum accuracy.

```bash
labelme-sam2-propagate --dir research_data \
    --model base_plus \
    --max-points 300 \
    --preview
```

## Example 6: Batch Processing

**Scenario:** Process 20 video sequences.

```bash
#!/bin/bash
for dir in sequences/*/; do
    echo "Processing $(basename $dir)..."
    labelme-sam2-propagate --dir "$dir" --preview --static background
done
```

## Example 7: Python API Integration

```python
from pathlib import Path
from labelme_sam2_propagate import run_propagation

# Process all subdirectories
data_root = Path("videos")
for sequence_dir in data_root.iterdir():
    if not sequence_dir.is_dir():
        continue
    
    print(f"Processing {sequence_dir.name}...")
    
    written = run_propagation(
        data_dir=str(sequence_dir),
        preview=True,
        static_labels=["background", "floor"],
        model_size="small",
    )
    
    print(f"  → Generated {written} annotations")
```

## Example 8: Scene Change Handling

**Scenario:** Video with camera cut or scene change at frame 50.

```bash
# Strategy: Add seed frames at boundaries
# - Annotate frames 0, 25, 50, 75, 100
# - Frames 50 becomes boundary between two segments
labelme-sam2-propagate --dir scene_changes --preview
```

## Example 9: New Object Appearance

**Scenario:** Person enters frame at frame 30.

```bash
# Use forward-only to prevent backward leakage
labelme-sam2-propagate --dir new_object --preview --forward-only
```

**Seed placement:**
- Frame 0: annotate background only
- Frame 30: annotate background + person
- Frame 60: annotate both (if needed)

## Example 10: Occlusion Recovery

**Scenario:** Object goes behind another object briefly.

```bash
# Strategy: Seed before and after occlusion
# - Frame 10: object visible (seed)
# - Frames 20-30: object occluded (propagated)
# - Frame 40: object visible again (seed)
labelme-sam2-propagate --dir occlusion_test --preview
```

SAM2's memory handles short occlusions automatically, but add seeds for long ones.

## Common Patterns

### Pattern 1: Seed Every N Frames
```bash
# For 100-frame video, annotate frames: 0, 20, 40, 60, 80
# Run once, get 95 frames automatically
```

### Pattern 2: Iterative Refinement
```bash
# First pass with minimal seeds
labelme-sam2-propagate --dir data --preview

# Review _preview/
# Fix frames: 15, 47, 82 in LabelMe

# Second pass (automatic, no flags needed)
labelme-sam2-propagate --dir data --preview

# Repeat until satisfied
```

### Pattern 3: Static + Dynamic Mix
```bash
# Identify static vs moving objects first
# Then optimize
labelme-sam2-propagate --dir mixed_scene \
    --static wall,floor,ceiling,table,chair \
    --preview
```

## Quality Metrics

After propagation, check:
1. **_preview/** for visual QA
2. **Polygon count** - should be similar to seed frames
3. **Frame gaps** - any missing outputs indicate issues
4. **Edge alignment** - zoom in on object boundaries

## Next Steps

- See [configuration.md](../docs/configuration.md) for advanced options
- Check [README.md](../README.md) for troubleshooting
- Join discussions at GitHub Issues
