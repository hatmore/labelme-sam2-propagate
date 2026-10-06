# labelme-sam2-propagate

LabelMe 的 SAM2 视频传播标注加速工具。用一帧种子标注，自动传播到整个视频序列。

[English](#english) | [中文](#中文)

---

## 中文

### 为什么需要这个工具？

LabelMe 内置的 SAM 是**单帧模型**——每次标注都从零开始，即使相邻帧几乎一样，也要重复操作。本工具使用 **SAM 2.1 的视频预测器**，具备跨帧记忆：

- ✅ **标注 1 帧，传播到全序列**（实测：1 个种子 → 28 帧 / 35 秒）
- ✅ **支持遮挡和视角变化**（叉车进出货柜、物体被遮挡后重新出现）
- ✅ **多目标同时跟踪**（一次跑 10+ 个实例）
- ✅ **双向传播**（两个种子之间前推+后推，漂移减半）
- ✅ **静态物体捷径**（货柜墙/地板直接复制，不浪费 GPU）

### 快速开始

#### 1. 安装环境

需要单独建一个 `sam2` 环境（不要和 labelme 混在一起）：

```bash
# 创建环境
conda create -n sam2 python=3.10 -y
conda activate sam2

# 安装依赖（国内用户建议用镜像源，见下方）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/facebookresearch/segment-anything-2.git
pip install opencv-python scipy
```

<details>
<summary>国内镜像源加速（点击展开）</summary>

```bash
# 清华源安装 torch（国内推荐）
pip install torch torchvision --index-url https://mirrors.tuna.tsinghua.edu.cn/pytorch/cu118

# SAM2 从清华源安装
pip install sam2 -i https://pypi.tuna.tsinghua.edu.cn/simple

# 其他依赖
pip install opencv-python scipy -i https://pypi.tuna.tsinghua.edu.cn/simple
```

</details>

#### 2. 下载脚本

```bash
git clone https://github.com/hatmore/labelme-sam2-propagate.git
cd labelme-sam2-propagate
```

#### 3. 使用流程

**第一步：在 LabelMe 里手标种子帧**
- 打开你的图像序列文件夹
- 每隔 ~20 帧标一帧（场景变化大的地方多标几帧）
- 保存（Ctrl+S）

**第二步：运行传播**

```bash
# Windows（改成你的实际路径）
C:\Users\Admin1\miniforge3\envs\sam2\python.exe sam2_propagate.py --dir <图像文件夹> --preview

# Linux/Mac
python sam2_propagate.py --dir <图像文件夹> --preview
```

**第三步：检查结果**
- 去 `<文件夹>/_preview/` 用看图软件快速刷一遍
- 挑出崩掉的帧

**第四步：迭代修正**
- 在 LabelMe 里打开那几帧，手动修正并保存
- **重新运行同样的命令**——脚本会自动把你改过的帧升级成新种子，只重算机器结果

### 集成到 LabelMe（可选）

不需要改 LabelMe 代码，只需配置一个外部工具：

**Windows 用户：** 创建 `labelme_sam2_tool.bat`：

```batch
@echo off
set SAM2_PYTHON=C:\Users\Admin1\miniforge3\envs\sam2\python.exe
set SCRIPT_PATH=E:\labelme-sam2-propagate\sam2_propagate.py

%SAM2_PYTHON% %SCRIPT_PATH% --dir %1 --preview
pause
```

**在 LabelMe 里调用：**
- 打开 LabelMe 的配置文件（`~/.labelmerc` 或项目目录下的 `.labelmerc`）
- 添加：

```yaml
# ... 其他配置 ...

# 外部工具（Windows 示例）
# 用法：在 LabelMe 里按 Ctrl+T 或菜单栏选择 Tools -> SAM2 Propagate
# 会自动传入当前文件夹路径
```

> 注：LabelMe 7.x 版本的外部工具配置方式可能不同，详见 LabelMe 官方文档。

### 命令参考

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--dir` | 图像序列文件夹（**必需**） | - |
| `--preview` | 生成叠加预览图到 `_preview/` | False |
| `--overwrite` | 覆盖已有标注（会先备份 `.bak`） | False |
| `--static` | 静态类别（逗号分隔），直接复制不跟踪 | `""` |
| `--checkpoint` | SAM2 权重路径 | 自动下载 |
| `--model-cfg` | SAM2 配置文件 | `sam2.1_hiera_b+.yaml` |
| `--forward-only` | 只正向传播，不回推 | False |
| `--max-points` | 输出多边形最大顶点数 | 200 |

**常见场景命令：**

```bash
# 1. 基础用法：传播 + 预览
python sam2_propagate.py --dir 20260904_148 --preview

# 2. 静态类别优化（货柜内壁、地板不动）
python sam2_propagate.py --dir 20260904_148 --preview \
    --static truck_wall,truck_roof,truck_floor,dock_board

# 3. 单向传播（已有头尾种子，中间只需正推）
python sam2_propagate.py --dir 20260904_131 --forward-only

# 4. 强制重算全部（测试用）
python sam2_propagate.py --dir 20260904_148 --overwrite
```

### 已知限制

| 场景 | 表现 | 解决办法 |
|------|------|----------|
| 语义跳变（叉车进/出货柜口） | `truck_floor` 蔓延到整个地面 | 在跳变帧附近补种子 |
| 目标缩到极小（< 500px） | 可能丢失 1-2 帧 | 不影响下游，或在该帧补种子 |
| 长序列单向传播（> 30 帧） | 累积漂移，边界逐渐偏移 | 每 20 帧补一个种子 |
| 多相机混在一个文件夹 | 脚本按文件名前缀自动分组 | 确保文件名格式统一 |

### 完整文档

详细工作流逻辑、失效模式分析、环境部署见 [标注加速方案.md](./标注加速方案.md)

---

## English

### Why This Tool?

LabelMe's built-in SAM is a **per-frame model**—each annotation starts from scratch, even when adjacent frames are nearly identical. This tool uses **SAM 2.1's video predictor** with cross-frame memory:

- ✅ **Annotate 1 frame, propagate to the whole sequence** (tested: 1 seed → 28 frames / 35s)
- ✅ **Handles occlusion and viewpoint changes** (forklift entering/exiting containers)
- ✅ **Multi-target simultaneous tracking** (10+ instances at once)
- ✅ **Bidirectional propagation** (forward + backward between seeds, halving drift)
- ✅ **Static object shortcut** (container walls/floor: copy directly, save GPU)

### Quick Start

#### 1. Install Environment

Create a separate `sam2` environment (don't mix with labelme):

```bash
# Create environment
conda create -n sam2 python=3.10 -y
conda activate sam2

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/facebookresearch/segment-anything-2.git
pip install opencv-python scipy
```

#### 2. Download Script

```bash
git clone https://github.com/hatmore/labelme-sam2-propagate.git
cd labelme-sam2-propagate
```

#### 3. Workflow

**Step 1: Manually annotate seed frames in LabelMe**
- Open your image sequence folder
- Annotate every ~20 frames (more at scene changes)
- Save (Ctrl+S)

**Step 2: Run propagation**

```bash
# Windows (adjust paths)
C:\Users\YourName\miniforge3\envs\sam2\python.exe sam2_propagate.py --dir <image_folder> --preview

# Linux/Mac
python sam2_propagate.py --dir <image_folder> --preview
```

**Step 3: Review results**
- Check `<folder>/_preview/` with an image viewer
- Find frames that failed

**Step 4: Iterate**
- Open those frames in LabelMe, fix, and save
- **Rerun the same command**—script auto-upgrades your edits to seeds and only recalculates machine results

### Labelme Integration (Optional)

No need to modify LabelMe code—just configure an external tool:

**Windows:** Create `labelme_sam2_tool.bat`:

```batch
@echo off
set SAM2_PYTHON=C:\Users\YourName\miniforge3\envs\sam2\python.exe
set SCRIPT_PATH=C:\path\to\labelme-sam2-propagate\sam2_propagate.py

%SAM2_PYTHON% %SCRIPT_PATH% --dir %1 --preview
pause
```

**In LabelMe:** Add to config file (`~/.labelmerc`):

```yaml
# External tool configuration
# Usage: Press Ctrl+T in LabelMe or select Tools -> SAM2 Propagate
# Will automatically pass the current folder path
```

> Note: External tool configuration may vary in LabelMe 7.x. See official LabelMe docs.

### Command Reference

| Option | Description | Default |
|--------|-------------|---------|
| `--dir` | Image sequence folder (**required**) | - |
| `--preview` | Generate overlay preview in `_preview/` | False |
| `--overwrite` | Overwrite existing annotations (backs up to `.bak`) | False |
| `--static` | Static categories (comma-separated), copy instead of track | `""` |
| `--checkpoint` | SAM2 checkpoint path | Auto-download |
| `--model-cfg` | SAM2 config file | `sam2.1_hiera_b+.yaml` |
| `--forward-only` | Only forward propagation, no backward | False |
| `--max-points` | Max polygon vertices | 200 |

**Common scenarios:**

```bash
# Basic: propagate + preview
python sam2_propagate.py --dir 20260904_148 --preview

# With static categories
python sam2_propagate.py --dir 20260904_148 --preview \
    --static truck_wall,truck_roof,truck_floor,dock_board

# Forward-only (when you have head & tail seeds)
python sam2_propagate.py --dir 20260904_131 --forward-only
```

### Known Limitations

| Scenario | Behavior | Solution |
|----------|----------|----------|
| Semantic jump (forklift entering/exiting) | `truck_floor` spreads to entire ground | Add seed near the jump |
| Target shrinks tiny (< 500px) | May lose 1-2 frames | Minor; or add seed at that frame |
| Long single-direction (> 30 frames) | Cumulative drift | Add seed every ~20 frames |
| Multi-camera in one folder | Script auto-groups by filename prefix | Ensure consistent naming |

### Full Documentation

See [标注加速方案.md](./标注加速方案.md) (Chinese) for detailed workflow logic, failure modes, and deployment guide.

---

## Citation

This tool uses [SAM 2.1](https://github.com/facebookresearch/segment-anything-2) and is designed for [LabelMe](https://github.com/wkentaro/labelme).

```bibtex
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  year={2024}
}
```

## License

MIT License

## Contributing

Issues and PRs welcome! For major changes, please open an issue first to discuss what you would like to change.
