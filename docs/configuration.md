# Configuration Guide

## Static Labels

Static labels are objects that don't move in the video sequence. For these labels, the tool skips GPU-intensive tracking and simply copies the polygon from the nearest seed frame.

### When to Use Static Labels

Use `--static` for:
- Fixed camera setups with static background
- Walls, floors, ceilings
- Equipment that doesn't move
- Docking boards, loading platforms
- Any object that appears in the same position across frames

### Performance Impact

Example scenario: 10 objects, 5 static, 30 frames
- **Without --static**: ~60 seconds (2 sec/frame)
- **With --static**: ~30 seconds (1 sec/frame)

### Configuration Examples

#### Warehouse/Loading Dock
```bash
labelme-sam2-propagate --dir images/ \
  --static truck_wall,truck_roof,truck_floor,dock_board
```

#### Indoor Manufacturing
```bash
labelme-sam2-propagate --dir images/ \
  --static wall,floor,ceiling,equipment_frame,workbench
```

#### Outdoor Fixed Camera
```bash
labelme-sam2-propagate --dir images/ \
  --static road,sidewalk,building,pole
```

## Model Selection

| Model | VRAM | Speed | Quality | Use Case |
|-------|------|-------|---------|----------|
| `tiny` | ~2GB | Fast | Good | Low VRAM, many videos |
| `small` | ~4GB | Medium | Very Good | Default choice |
| `base_plus` | ~6GB | Slow | Excellent | High quality needed |
| `large` | ~8GB | Slowest | Best | Maximum accuracy |

### GPU Memory Troubleshooting

If you get CUDA out-of-memory errors:

1. Try a smaller model:
   ```bash
   labelme-sam2-propagate --dir images/ --model tiny
   ```

2. Close other GPU applications (browsers, other ML tools)

3. Reduce batch size by processing fewer frames at once (automatic in the tool)

## Integration with LabelMe

### Recommended LabelMe Settings

Edit `~/.labelmerc`:

```yaml
# Auto-carry previous frame's polygons
keep_prev: true

# Auto-save on frame change
auto_save: true

# Keep zoom level when switching frames
keep_prev_scale: true

# Toggle keep_prev with Ctrl+P
```

### Workflow Integration

1. **First-time setup**:
   ```bash
   # Create project directory
   mkdir my_annotation_project
   cd my_annotation_project
   
   # Copy images
   cp /source/images/*.png .
   ```

2. **Label seeds**:
   ```bash
   labelme .
   # Label first frame, save, close
   ```

3. **Propagate**:
   ```bash
   labelme-sam2-propagate --dir . --preview
   ```

4. **Review**:
   ```bash
   # Check _preview/ folder first
   # Then open in labelme for corrections
   labelme .
   ```

## Environment Setup

### Option 1: Separate from LabelMe (Recommended)

```bash
# Create dedicated sam2 environment
conda create -n sam2 python=3.10 -y
conda activate sam2
pip install labelme-sam2-propagate

# Use specific Python when calling
/path/to/sam2/python -m labelme_sam2_propagate.cli --dir images/
```

### Option 2: Integrated with LabelMe

```bash
# Activate labelme environment
conda activate labelme

# Install sam2 dependencies (large download!)
pip install torch torchvision sam2

# Now can use directly
labelme-sam2-propagate --dir images/
```

**Note**: Option 2 adds ~2.5GB to your LabelMe environment. Option 1 is recommended.

## Batch Processing

### Process Multiple Directories

```bash
# Unix/Linux/Mac
for dir in dataset/*/; do
  labelme-sam2-propagate --dir "$dir" --preview
done

# Windows (PowerShell)
Get-ChildItem -Directory dataset | ForEach-Object {
  labelme-sam2-propagate --dir $_.FullName --preview
}
```

### Parallel Processing

```bash
# GNU parallel (Linux/Mac)
ls -d dataset/*/ | parallel -j 2 labelme-sam2-propagate --dir {} --preview

# Windows (PowerShell)
Get-ChildItem -Directory dataset | ForEach-Object -Parallel {
  labelme-sam2-propagate --dir $_.FullName --preview
} -ThrottleLimit 2
```

## Quality Control

### Preview Image Interpretation

Check `_preview/` images for:
- **Mask color consistency**: Each object should maintain its color
- **Boundary tightness**: Edges should follow object contours
- **Occlusion handling**: Objects should reappear after being blocked
- **Area stability**: Object area shouldn't jump unexpectedly

### Common Issues and Fixes

| Issue | Cause | Fix |
|-------|-------|-----|
| Drift after 10+ frames | No intermediate seeds | Add seed every ~20 frames |
| Boundary creep | Wrong static label | Add to `--static` list |
| Mask explosion | Scene change | Add seed at discontinuity |
| Object lost | Too small in frame | Label manually, re-run |

### Seed Placement Strategy

**Good seed placement** (minimal manual work):
```
Frame:  0 [seed] ... 20 [seed] ... 40 [seed] ... 60
Result: ████████████████████████████████████████████
        Perfect quality throughout
```

**Poor seed placement** (requires many fixes):
```
Frame:  0 [seed] ............................ 60
Result: ████░░░░░░░░░░░░░░░░░▓▓▓▓▒▒▒▒▒▒▒▒▒▒▒▒
        Good ... OK ... Needs fix .........
```

## Troubleshooting

### Import Errors

```
ModuleNotFoundError: No module named 'sam2'
```

**Solution**: Make sure you're using the correct Python environment:
```bash
# Check which Python
which python  # or "where python" on Windows

# Use full path if needed
/path/to/sam2/env/python -m labelme_sam2_propagate.cli --dir images/
```

### CUDA Errors

```
RuntimeError: CUDA out of memory
```

**Solution**: Use smaller model or close other GPU apps:
```bash
labelme-sam2-propagate --dir images/ --model tiny
```

### No Propagation Output

**Check**:
1. Do seed frames exist? (`*.json` files with non-empty `shapes`)
2. Are there unlabeled frames? (`.png`/`.jpg` without matching `.json`)
3. Is filename format correct? (must match `prefix_timestamp_nanosec` pattern)

**Debug**:
```bash
# Run with Python directly to see detailed output
python -m labelme_sam2_propagate.cli --dir images/ --preview
```
