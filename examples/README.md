# Examples

This directory contains example workflows and use cases.

## Basic Usage

```bash
# Simple propagation
labelme-sam2-propagate --dir /path/to/images --preview

# With static labels (for fixed camera + static objects)
labelme-sam2-propagate --dir /path/to/images --static truck_wall,truck_roof,dock_board --preview
```

## Typical Workflow

1. **Label seed frames in LabelMe**
   - Open your image directory in LabelMe
   - Label the first frame completely
   - Label additional frames every ~20 frames (optional, but improves quality)
   - Save all annotations

2. **Run propagation**
   ```bash
   labelme-sam2-propagate --dir /path/to/images --preview
   ```

3. **Review results**
   - Check `_preview/` directory for overlay images
   - Quickly scan through to identify any problematic frames

4. **Fix and iterate**
   - Open the directory in LabelMe
   - Fix any incorrect annotations
   - Re-run the same command - fixed frames automatically become new seeds

5. **Done**
   - All frames now have annotations
   - Continue normal LabelMe workflow

## Advanced Examples

### Multiple Camera Sequences

The tool automatically groups frames by camera prefix:

```
images/
  N_camera_0_link_1788425812_781000000.png
  N_camera_0_link_1788425813_281000000.png
  P_camera_0_link_1788424303_631000000.png
  P_camera_0_link_1788424304_131000000.png
```

Will be processed as two separate sequences.

### Static Object Optimization

For warehouse/loading dock scenarios where walls, floors, and equipment are static:

```bash
labelme-sam2-propagate --dir images/ \
  --static truck_wall,truck_roof,truck_floor,dock_board,equipment \
  --preview
```

This skips GPU tracking for these labels and simply copies them from the nearest seed frame.

### Model Size Selection

Choose based on your GPU memory:

```bash
# For 4GB VRAM (default)
labelme-sam2-propagate --dir images/ --model small

# For 2GB VRAM
labelme-sam2-propagate --dir images/ --model tiny

# For 8GB+ VRAM (best quality)
labelme-sam2-propagate --dir images/ --model large
```

## Expected Performance

- **Speed**: ~1-2 seconds per frame on RTX 3050 (4GB)
- **Quality**: Very accurate for continuous motion, may need correction at discontinuities
- **Memory**: ~3.5GB VRAM for `small` model

## Tips

1. **Seed placement**: Put seeds at scene boundaries (forklift enters/exits, camera moves, lighting changes)
2. **Static labels**: Always mark truly static objects - saves time and improves stability
3. **Preview first**: Always use `--preview` to catch issues before manual review
4. **Iterative refinement**: Don't aim for perfection in first seed - fix and re-run
