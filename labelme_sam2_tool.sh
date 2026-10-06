#!/bin/bash
# LabelMe SAM2 Propagate 启动脚本（Linux/Mac）
# 使用方法：./labelme_sam2_tool.sh <图像文件夹路径>

# ============ 配置区（修改成你的实际路径）============
SAM2_PYTHON="${HOME}/miniforge3/envs/sam2/bin/python"
SCRIPT_PATH="$(dirname "$0")/sam2_propagate.py"
# ====================================================

if [ -z "$1" ]; then
    echo "错误：未提供图像文件夹路径"
    echo "使用方法：$0 <图像文件夹路径>"
    exit 1
fi

echo "========================================"
echo "LabelMe SAM2 视频传播工具"
echo "========================================"
echo ""
echo "目标文件夹: $1"
echo "Python环境: $SAM2_PYTHON"
echo "脚本路径: $SCRIPT_PATH"
echo ""
echo "开始处理..."
echo "----------------------------------------"
echo ""

"$SAM2_PYTHON" "$SCRIPT_PATH" --dir "$1" --preview

echo ""
echo "----------------------------------------"
echo "处理完成！"
echo "预览图已保存到：$1/_preview/"
echo ""
