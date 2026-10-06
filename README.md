# LabelMe SAM2 Propagate

<div align="center">

**Accelerate video annotation with SAM 2.1 cross-frame tracking**

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![SAM 2.1](https://img.shields.io/badge/SAM-2.1-green.svg)](https://github.com/facebookresearch/segment-anything-2)

[Features](#-features) • [Quick Start](#-quick-start) • [Usage](#-usage) • [Performance](#-performance)

</div>

---

## 💡 Why This Tool?

LabelMe's built-in SAM is a **per-frame model**—each annotation starts from scratch. This tool uses **SAM 2.1's video predictor** with cross-frame memory:

- ✅ **Annotate 1 frame, propagate to entire sequence** (tested: 1 seed → 28 frames in 35s)
- ✅ **Handles occlusion and viewpoint changes** (forklift entering/exiting containers)
- ✅ **Multi-target simultaneous tracking** (10+ instances at once)
- ✅ **Bidirectional propagation** (forward + backward between seeds, halving drift)
- ✅ **Static object shortcut** (container walls/floor: copy directly, save GPU time)

## 📊 Performance

| Metric | Value |
|--------|-------|
| Speed | ~1.3s/frame (RTX 3050 4GB) |
| VRAM | ~3.5GB (small model) |
| Accuracy | >95% for continuous motion |
| Capacity | 10+ simultaneous objects |

## 🚀 Quick Start

### 1. Install

```bash
# Clone repository
git clone https://github.com/hatmore/labelme-sam2-propagate.git
cd labelme-sam2-propagate

# Create environment (separate from labelme)
conda create -n sam2 python=3.10 -y
conda activate sam2

# Install package
pip install -e .
```

### 2. Annotate Seed Frames

Open your image sequence in LabelMe and manually annotate:
- **1 frame** for short sequences (<20 frames)
- **Every ~20 frames** for longer sequences
- **Extra frames** at scene changes (object enters/exits, camera moves)

### 3. Run Propagation

```bash
# Basic usage
labelme-sam2-propagate --dir /path/to/images --preview

# With static object optimization
labelme-sam2-propagate --dir /path/to/images --preview \
    --static wall,floor,ceiling
```

### 4. Review & Iterate

- Check `_preview/` directory for overlay visualizations
- Open failed frames in LabelMe, fix them, and save
- **Re-run the same command** — your edits automatically become seeds

## 📖 Usage

### Command Line Options

```bash
labelme-sam2-propagate [OPTIONS]

Required:
  --dir PATH              Image sequence directory

Optional:
  --preview              Generate overlay visualizations in _preview/
  --static LABELS        Comma-separated labels to copy (not track)
  --model {tiny,small,base_plus,large}
                         Model size (default: small)
  --overwrite            Overwrite existing manual annotations
  --forward-only         Only forward propagation (no backward)
  --max-points N         Max polygon vertices (default: 200)
```

### Common Workflows

**Basic propagation:**
```bash
labelme-sam2-propagate --dir my_sequence --preview
```

**Warehouse/logistics (static containers):**
```bash
labelme-sam2-propagate --dir my_sequence --preview \
    --static truck_wall,truck_roof,truck_floor,dock_board
```

**Low VRAM (<4GB):**
```bash
labelme-sam2-propagate --dir my_sequence --model tiny
```

**Force regeneration:**
```bash
labelme-sam2-propagate --dir my_sequence --overwrite
```

### Python API

```python
from labelme_sam2_propagate import run_propagation

written = run_propagation(
    data_dir="my_sequence",
    preview=True,
    static_labels=["wall", "floor"],
    model_size="small",
)
print(f"Generated {written} annotations")
```

## 🎯 Best Practices

### Seed Frame Placement

| Scenario | Strategy |
|----------|----------|
| Smooth motion | 1 seed every 20 frames |
| Scene changes | Add seed at transition |
| Occlusion events | Seed before and after |
| Static background | Use `--static` for fixed objects |

### Quality Tips

1. **Check drift early**: Use `--preview` to spot issues before manual review
2. **Seed at extremes**: Annotate the most different-looking frames (object closest/farthest)
3. **Static labels**: Identify non-moving objects and pass via `--static` (2-3× faster)
4. **Iterative refinement**: Fix bad frames → re-run → repeat (your fixes become seeds)

## 🔧 Advanced

### Multi-Camera Sequences

The tool auto-detects camera prefixes in filenames:
```
N_camera_0_link_1788425812_781000000.jpg  → "N_camera_0_link" sequence
P_camera_0_link_1788425812_781000000.jpg  → "P_camera_0_link" sequence
```

Process only one camera:
```bash
labelme-sam2-propagate --dir mixed_cameras --only N_camera_0_link
```

### How It Works

1. **Seed detection**: Any non-empty JSON = seed (unless auto-generated and unmodified)
2. **Target assignment**: Empty frames assigned to nearest seed
3. **Bidirectional split**: Gap between seeds split in half (forward + backward)
4. **Change tracking**: Content hash distinguishes manual edits from auto results
5. **Auto-upgrade**: Edited frames automatically become seeds on next run

## 📝 Known Limitations

| Issue | Symptom | Solution |
|-------|---------|----------|
| Semantic jump | Floor extends to entire room | Add seed at boundary frame |
| Tiny objects | <500px disappears 1-2 frames | Acceptable or add seed |
| Long single-direction | >30 frames, cumulative drift | Add intermediate seed every 20 frames |

## 🤝 Integration with LabelMe

### Quick Launcher (Windows)

Create `labelme_sam2.bat`:
```batch
@echo off
C:\path\to\miniforge3\envs\sam2\python.exe ^
    -m labelme_sam2_propagate.cli --dir %1 --preview
pause
```

Right-click folder → Send To → `labelme_sam2.bat`

### Quick Launcher (Linux/Mac)

Create `labelme_sam2.sh`:
```bash
#!/bin/bash
conda run -n sam2 labelme-sam2-propagate --dir "$1" --preview
```

```bash
chmod +x labelme_sam2.sh
./labelme_sam2.sh /path/to/images
```

## 📦 Project Structure

```
labelme-sam2-propagate/
├── src/
│   └── labelme_sam2_propagate/
│       ├── __init__.py       # Public API
│       ├── cli.py            # Command-line interface
│       ├── core.py           # Main propagation logic
│       ├── models.py         # SAM2 checkpoint management
│       ├── utils.py          # Mask/polygon conversion
│       ├── sequence.py       # Frame parsing and sorting
│       └── ledger.py         # Auto-generation tracking
├── pyproject.toml            # Package metadata
├── README.md                 # This file
└── LICENSE                   # MIT License
```

## 🐛 Troubleshooting

**CUDA out of memory:**
```bash
# Use smaller model
labelme-sam2-propagate --dir . --model tiny
```

**Download timeout:**
```bash
# Manual download (place in ~/.cache/sam2_ckpt/)
wget https://hf-mirror.com/facebook/sam2.1-hiera-small/resolve/main/sam2.1_hiera_small.pt
```

**Wrong propagation direction:**
```bash
# Force forward-only (when new objects appear mid-sequence)
labelme-sam2-propagate --dir . --forward-only
```

## 📄 License

MIT License - see [LICENSE](LICENSE) file

## 🙏 Acknowledgments

- [SAM 2.1](https://github.com/facebookresearch/segment-anything-2) by Meta AI
- [LabelMe](https://github.com/wkentaro/labelme) by Kentaro Wada

## 📮 Contact

Issues and PRs welcome at [github.com/hatmore/labelme-sam2-propagate](https://github.com/hatmore/labelme-sam2-propagate)

---

<div align="center">
<sub>Made with ❤️ for efficient video annotation</sub>
</div>
