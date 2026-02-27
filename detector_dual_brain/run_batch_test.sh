#!/bin/bash

# --- 配置 ---
INPUT_DIR="HomeSafe-Bench/DATASETS_TimeStamp"
OUTPUT_DIR="./test_results_json"
PYTHON_SCRIPT="dual_brain.py"

# 创建输出目录
mkdir -p "$OUTPUT_DIR"

# 检查输入目录是否存在
if [ ! -d "$INPUT_DIR" ]; then
    echo "Error: Input directory $INPUT_DIR does not exist."
    exit 1
fi

echo "=========================================="
echo "Starting Batch Test"
echo "Input: $INPUT_DIR"
echo "Output: $OUTPUT_DIR"
echo "=========================================="

# 遍历视频文件 (支持 mp4, avi, mkv, mov)
find "$INPUT_DIR" -type f \( -name "*.mp4" -o -name "*.avi" -o -name "*.mkv" -o -name "*.mov" \) | sort | while read -r video_file; do
    
    # 获取文件名（不带路径和后缀）
    filename=$(basename -- "$video_file")
    filename_no_ext="${filename%.*}"
    
    # 定义输出 JSON 路径
    output_json="$OUTPUT_DIR/${filename_no_ext}_result.json"
    
    echo ""
    echo "------------------------------------------"
    echo "Processing: $filename"
    echo "Saving to: $output_json"
    echo "------------------------------------------"
    
    # 运行 Python 脚本
    # 注意：这里是串行运行，防止显存爆炸。如果显存足够大，可以考虑后台运行但要小心管理。
    python3 "$PYTHON_SCRIPT" --video "$video_file" --output "$output_json"
    
    if [ $? -eq 0 ]; then
        echo "✅ Finished: $filename"
    else
        echo "❌ Error processing: $filename"
    fi
    
    # 可选：每跑完一个视频稍微休息一下，让 GPU 降温或释放显存
    sleep 1

done

echo "=========================================="
echo "All tests completed."
echo "=========================================="