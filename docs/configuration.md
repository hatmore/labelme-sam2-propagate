# Configuration Guide

## Model Selection

Choose model size based on your hardware:

| Model | VRAM | Speed | Accuracy | Use Case |
|-------|------|-------|----------|----------|
| `tiny` | ~2GB | ~0.8s/frame | Good | Low-end GPUs, quick tests |
| `small` | ~3.5GB | ~1.3s/frame | Better | **Default, balanced** |
| `base_plus` | ~5GB | ~2.0s/frame | Best | High accuracy needed |
| `large` | ~8GB | ~3.5s/frame | Excellent | Research, maximum quality |

```bash
# Use tiny model for 2GB GPUs
labelme-sam2-propagate --dir images --model tiny
```

## Static Labels Optimization

For scenes with non-moving objects (warehouse, indoor surveillance):

```bash
# Container truck scenario
labelme-sam2-propagate --dir truck_loading \
    --static truck_wall,truck_roof,truck_floor,dock_board

# Indoor warehouse
labelme-sam2-propagate --dir warehouse \
    --static wall,ceiling,floor,rack,column
```

**Benefits:**
- 2-3× faster processing
- No drift for static objects
- Lower GPU memory usage

## Propagation Direction

### Default: Bidirectional
```bash
labelme-sam2-propagate --dir images
```
Splits gaps between seeds: forward from left, backward from right.

### Forward-Only Mode
```bash
labelme-sam2-propagate --dir images --forward-only
```

**Use when:**
- New objects appear mid-sequence
- Seeds have different label sets
- One-way temporal dependency

## Preview Generation

```bash
# Generate overlay visualizations
labelme-sam2-propagate --dir images --preview
```

Creates `images/_preview/` with color-coded overlays.

**Workflow:**
1. Run with `--preview`
2. Open `_preview/` in image viewer
3. Scan through quickly (arrow keys)
4. Note frame numbers with issues
5. Fix those frames in LabelMe
6. Re-run (auto-detects your fixes)

## Polygon Simplification

Control output polygon complexity:

```bash
# Simpler polygons (faster, less detail)
labelme-sam2-propagate --dir images --max-points 50

# Complex shapes (slower, more detail)
labelme-sam2-propagate --dir images --max-points 300
```

**Default: 200 points** - good balance for most cases.

## Environment Variables

```bash
# Force CPU mode
CUDA_VISIBLE_DEVICES="" labelme-sam2-propagate --dir images

# Limit to specific GPU
CUDA_VISIBLE_DEVICES=0 labelme-sam2-propagate --dir images

# Set custom cache directory
SAM2_CACHE_DIR=/path/to/cache labelme-sam2-propagate --dir images
```

## Batch Processing

Process multiple directories:

```bash
# Bash
for dir in data/*/; do
    labelme-sam2-propagate --dir "$dir" --preview --static wall,floor
done

# PowerShell
Get-ChildItem -Directory data | ForEach-Object {
    labelme-sam2-propagate --dir $_.FullName --preview --static wall,floor
}
```

## Performance Tuning

### Low VRAM (<4GB)
```bash
labelme-sam2-propagate --dir images --model tiny
```

### CPU-Only
```bash
# Install CPU-only PyTorch first
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu

# Then run
labelme-sam2-propagate --dir images --model tiny
```

### High-Quality Mode
```bash
labelme-sam2-propagate --dir images \
    --model base_plus \
    --max-points 300
```

## Troubleshooting

### Out of Memory
```bash
# Reduce model size
labelme-sam2-propagate --dir images --model tiny

# OR process fewer objects per seed
# Split long sequences with more seed frames
```

### Slow Downloads
```bash
# Pre-download checkpoint manually
mkdir -p ~/.cache/sam2_ckpt
cd ~/.cache/sam2_ckpt
wget https://hf-mirror.com/facebook/sam2.1-hiera-small/resolve/main/sam2.1_hiera_small.pt
```

### Wrong Propagation Direction
```bash
# Use forward-only when new objects appear
labelme-sam2-propagate --dir images --forward-only
```

### Drift Accumulation
```bash
# Add more seed frames (every 15-20 frames)
# Shorter propagation distances = less drift
```

## Integration Examples

### Custom Python Script

```python
from labelme_sam2_propagate import run_propagation

# Basic
run_propagation(
    data_dir="images",
    preview=True,
)

# Advanced
run_propagation(
    data_dir="warehouse_video",
    preview=True,
    static_labels=["wall", "floor", "ceiling"],
    model_size="small",
    max_points=150,
    forward_only=False,
)
```

### Batch Script with Logging

```bash
#!/bin/bash
LOG_DIR="logs"
mkdir -p "$LOG_DIR"

for sequence in data/*/; do
    name=$(basename "$sequence")
    echo "Processing $name..."
    
    labelme-sam2-propagate \
        --dir "$sequence" \
        --preview \
        --static wall,floor \
        2>&1 | tee "$LOG_DIR/$name.log"
done
```

### Quality Assurance Pipeline

```bash
#!/bin/bash
# 1. Generate annotations
labelme-sam2-propagate --dir images --preview

# 2. Create demo GIF for quick review
python create_demo_gifs.py images/_preview docs/

# 3. Open GIF for visual QA
xdg-open docs/demo_fast.gif
```
