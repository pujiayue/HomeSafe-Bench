#!/bin/bash

# 修改为你的实际路径
DATA_DIR="HomeSafe-Bench/test/results/severity"
# 注意：确保这里的文件名与磁盘上的一致 (anno_truth_final.json)
GT_FILE="$DATA_DIR/anno_truth_final.json"
EVAL_SCRIPT="./evaluate_severity.py"
OUTPUT_CSV="severity_evaluation_results.csv"

echo "Model_Name,Overall_Severity_Accuracy(%)" > "$OUTPUT_CSV"

echo "===================================================="
echo "正在评估危险等级标签准确率..."
echo "===================================================="

for pred_file in "$DATA_DIR"/*.json; do
    filename=$(basename "$pred_file")
    
    # 排除掉 GT 文件本身和汇总文件
    if [[ "$filename" == "anno_truth_final.json" || "$filename" == "severity_summary.json" ]]; then 
        continue 
    fi
    
    echo -n "处理中: $filename ... "
    
    # 执行并捕获输出
    output=$(python3 "$EVAL_SCRIPT" "$GT_FILE" "$pred_file")
    result_line=$(echo "$output" | grep "RESULT_SEV:")
    
    if [ ! -z "$result_line" ]; then
        clean_result=$(echo "$result_line" | sed 's/RESULT_SEV: //')
        echo "$clean_result" >> "$OUTPUT_CSV"
        echo "成功"
    else
        echo "失败 (数据不匹配或脚本报错)"
    fi
done

echo "----------------------------------------------------"
echo "完成！汇总结果已保存至: $OUTPUT_CSV"