import json
import os
import sys
import numpy as np

def evaluate(gt_path, pred_path):
    # 1. 加载 GT
    with open(gt_path, "r") as f:
        gt_data = json.load(f)
    
    severity_order = {"L1": 0, "L2": 1, "L3": 2, "L4": 3}
    gt_dict = {}
    for item in gt_data["annotations"]:
        if item.get("is_valid", True):
            filename = os.path.basename(item["file_path"])
            gt_dict[filename] = item["severity"]

    # 2. 加载 Pred
    with open(pred_path, "r") as f:
        pred_data = json.load(f)

    y_true, y_pred = [], []
    
    for video_name, info in pred_data.items():
        if video_name in gt_dict:
            gt_sev = gt_dict[video_name]
            pred_sev = info.get("severity", None)
            if pred_sev in severity_order:
                y_true.append(severity_order[gt_sev])
                y_pred.append(severity_order[pred_sev])

    if len(y_true) == 0:
        return None

    # 3. 计算准确率
    accuracy = np.mean(np.array(y_true) == np.array(y_pred))
    return accuracy * 100

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python evaluate_severity.py <gt_json> <pred_json>")
        sys.exit(1)
    
    gt_file = sys.argv[1]
    pred_file = sys.argv[2]
    
    acc = evaluate(gt_file, pred_file)
    
    if acc is not None:
        # 这里的输出必须包含 RESULT_SEV: 供 Shell 脚本捕获
        model_name = os.path.basename(pred_file)
        print(f"RESULT_SEV: {model_name},{acc:.2f}")
    else:
        sys.exit(1)