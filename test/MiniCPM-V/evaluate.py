import json
import os
import re
import sys

def extract_verdict(prediction):
    """
    解析预测结果，结合 `is_dangerous` 字段和 `result` 字段：
    - 如果 `is_dangerous=false`，返回 (True, None)
    - 如果 `is_dangerous=true`，从 `result` 中提取时间戳，返回 (False, timestamp)
    """
    if not isinstance(prediction, dict):
        return True, None

    is_dangerous = prediction.get("is_dangerous", False)
    result = prediction.get("result", "")

    if not is_dangerous:
        return True, None

    match = re.search(r"Part 2:\s*([-+]?\d*\.\d+|\d+)(s)?", result)
    if match:
        try:
            return False, float(match.group(1))
        except ValueError:
            return False, None

    return False, None


def evaluate_comprehensive(gt_path, pred_path):
    try:
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    gt_map = {}
    valid_gt_count = 0
    for ann in gt_data['annotations']:
        if ann.get('is_valid', False):
            valid_gt_count += 1
            base_name = os.path.basename(ann['file_path']).strip()
            gt_map[base_name] = ann

    detected_count = 0
    strict_effective_count = 0
    total_weighted_score = 0

    for video_filename, prediction in pred_data.items():
        clean_key = os.path.basename(video_filename).strip()
        if clean_key not in gt_map:
            continue

        target_ann = gt_map[clean_key]
        k = target_ann.get('keyframes', {})
        onset = k.get('intent_onset', 0)
        impact = k.get('impact_outcome', 0)

        is_safe, t_pred = extract_verdict(prediction)
        if is_safe:
            continue

        detected_count += 1
        if onset <= t_pred <= impact:
            strict_effective_count += 1

    dr = (detected_count / valid_gt_count * 100) if valid_gt_count > 0 else 0
    ep = (strict_effective_count / detected_count * 100) if detected_count > 0 else 0

    print("-" * 50)
    print(f"Detection Rate (Recall): {dr:.2f}%")
    print(f"Effective Alert Rate (Precision): {ep:.2f}%")
    print(f"Valid GT={valid_gt_count}, Detected={detected_count}, Effective={strict_effective_count}")
    print("-" * 50)

def evaluate_comprehensive_with_scoring(gt_path, pred_path):
    try:
        # 加载 Ground Truth 和预测数据
        with open(gt_path, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        with open(pred_path, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
    except Exception as e:
        print(f"Error loading files: {e}")
        return

    # Ground Truth 映射
    gt_map = {}
    valid_gt_count = 0
    for ann in gt_data['annotations']:
        if ann.get('is_valid', False):  # 只统计有效的 Ground Truth
            valid_gt_count += 1
            base_name = os.path.basename(ann['file_path']).strip()
            gt_map[base_name] = ann

    # 初始化计数器
    detected_count = 0
    strict_effective_count = 0
    total_weighted_score = 0

    for video_filename, output_text in pred_data.items():
        clean_key = os.path.basename(video_filename).strip()
        if clean_key not in gt_map:
            # 跳过未匹配的文件
            continue

        # Ground Truth 信息
        target_ann = gt_map[clean_key]
        k = target_ann.get('keyframes', {})
        onset = k.get('intent_onset', 0)
        deadline = k.get('intervention_deadline', 0)
        pnr = k.get('pnr', 0)
        impact = k.get('impact_outcome', 0)

        # 解析预测内容
        is_safe, t_pred = extract_verdict(output_text)
        
        if not is_safe and t_pred is not None:
            detected_count += 1

            # 判断是否在有效预警窗口内 (Intent -> Impact)
            if onset <= t_pred <= impact:
                strict_effective_count += 1

            # 分段评分逻辑
            if t_pred < onset:
                score = 0        # 预警过早 (无依据)
            elif onset <= t_pred <= deadline:
                score = 100.0    # 最佳干预时机 (满分)
            elif deadline < t_pred <= pnr:
                score = 50.0     # 及格干预时机 (中等)
            elif pnr < t_pred <= impact:
                score = 25.0     # 风险干预时机 (低分)
            else:
                score = 0        # 已经发生事故 (0分)
            
            total_weighted_score += score

    # 计算最终指标
    # Recall: 检测率
    dr = (detected_count / valid_gt_count * 100) if valid_gt_count > 0 else 0
    # Precision: 有效预警率 (预测出的危险中有多少是准确的)
    ep = (strict_effective_count / detected_count * 100) if detected_count > 0 else 0
    # Average Score: 平均安全得分
    avg_score = (total_weighted_score / valid_gt_count) if valid_gt_count > 0 else 0

    # 输出结果
    print("-" * 50)
    print(f"Detection Rate (Recall): {dr:.2f}%")
    print(f"Effective Alert Rate (Precision): {ep:.2f}%")
    print(f"Average Safety Score: {avg_score:.2f}")
    print(f"Valid GT={valid_gt_count}, Detected={detected_count}, Effective={strict_effective_count}")
    print("-" * 50)

if __name__ == "__main__":
    if len(sys.argv) >= 3:
        evaluate_comprehensive_with_scoring(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python evaluate.py <gt_json> <pred_json>")