import json
import os
import re

# ================= 配置区域 =================
GT_FILE_PATH = "HomeSafe-Bench/anno_truth.json"
PRED_FILE_PATH = "YOUR_FILE_PATH"
# ===========================================

def extract_number(s):
    """从字符串或列表项中提取第一个浮点数"""
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(s))
        return float(match.group()) if match else None
    except:
        return None

def get_mapping_key(ann):
    """根据 GT 元数据拼接 Key 以匹配模型输出的文件名格式"""
    parts = [
        str(ann.get('danger_type', '')),
        str(ann.get('severity', '')),
        str(ann.get('reasoning_difficulty', '')),
        str(ann.get('sub_task', '')),
        str(ann.get('scale_idx', ''))
    ]
    parts = [p for p in parts if p.strip()]
    if parts:
        return "_".join(parts).lower().strip()
    return None

def evaluate():
    if not os.path.exists(GT_FILE_PATH) or not os.path.exists(PRED_FILE_PATH):
        print("错误：文件路径不存在，请检查脚本内的路径设置。")
        return

    try:
        with open(GT_FILE_PATH, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        with open(PRED_FILE_PATH, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
    except Exception as e:
        print(f"解析 JSON 出错: {e}")
        return

    # 1. 建立 GT 映射 (只处理 is_valid 为 True 的样本)
    gt_map = {}
    valid_gt_count = 0
    for ann in gt_data['annotations']:
        if ann.get('is_valid') is True:
            valid_gt_count += 1
            meta_key = get_mapping_key(ann)
            if meta_key:
                gt_map[meta_key] = ann
            video_path = ann['file_path'] if 'file_path' in ann else ann['video_name']
            base_name = os.path.basename(video_path).lower().replace('.mp4', '').strip()
            gt_map[base_name] = ann

    # 统计变量
    red_detected_count = 0      # 仅 status 为 red 且有时间戳的情况
    strict_effective_count = 0  # red 且时间在 [onset, impact] 之间
    total_weighted_score = 0    # 总分
    
    yellow_count = 0            # yellow 的数量统计
    yellow_before_impact_count = 0 # yellow 且时间在 impact 之前

    # 2. 以 GT 为基准进行遍历，确保每个有效视频都贡献分母
    # 注意：这里我们遍历 gt_map 的目的是确保所有 valid 样本都被计算
    # 但由于预测文件是以文件名作为 key，我们直接遍历预测结果，
    # 只要它在 GT 中是有效的，就进行计分；不在 GT 中的不计入。
    
    for pred_video_key, pred_info in pred_data.items():
        clean_pred_key = os.path.basename(pred_video_key).lower().replace('.mp4', '').strip()
        
        target_ann = None
        if clean_pred_key in gt_map:
            target_ann = gt_map[clean_pred_key]
        else:
            for k in gt_map:
                if k in clean_pred_key or clean_pred_key in k:
                    target_ann = gt_map[k]
                    break
        
        # 如果该视频不在有效的 GT 列表里，跳过
        if not target_ann:
            continue
        
        k = target_ann['keyframes']
        status = str(pred_info.get("status", "")).lower()
        timestamps = pred_info.get("timestamps", [])
        t_pred = extract_number(timestamps[0]) if (isinstance(timestamps, list) and len(timestamps) > 0) else None

        impact = k.get('impact_outcome', 0)
        onset = k.get('intent_onset', 0)
        deadline = k.get('intervention_deadline', 0)
        pnr = k.get('pnr', 0)

        # 核心逻辑：只有 RED 状态会尝试去拿分数
        if status == "red":
            red_detected_count += 1
            if t_pred is not None:
                # 判定是否为有效时间区间
                if onset <= t_pred <= impact:
                    strict_effective_count += 1

                # 详细计分
                if t_pred < onset:
                    current_score = 0
                elif onset <= t_pred <= deadline:
                    current_score = 100.0
                elif deadline < t_pred <= pnr:
                    current_score = 50.0
                elif pnr < t_pred <= impact:
                    current_score = 25.0
                else:
                    current_score = 0
                total_weighted_score += current_score
            else:
                # status 是 red 但没给时间戳，分数为 0
                pass
        
        elif status == "yellow":
            yellow_count += 1
            if t_pred is not None and t_pred < impact:
                yellow_before_impact_count += 1
            # Yellow 贡献的分数为 0，不执行 total_weighted_score += ...
        
        else:
            # status 为 unknown 或其他，贡献分数为 0
            pass

    # 3. 指标计算
    # 危险检测率 (DR): 分子是 red 的数量，分母是所有有效 GT 的数量
    dr = (red_detected_count / valid_gt_count * 100) if valid_gt_count > 0 else 0
    
    # 有效告警率 (EP): 在已经判定为 red 的集合中，时间给对的比例
    ep = (strict_effective_count / red_detected_count * 100) if red_detected_count > 0 else 0
    
    # 平均得分: 所有的有效视频（包括 red, yellow, unknown）都是分母
    # yellow 和 unknown 因为没有加 total_weighted_score，实际上就是贡献了 0 分
    avg_score = (total_weighted_score / valid_gt_count) if valid_gt_count > 0 else 0
    
    # Yellow 相关指标
    yellow_rate = (yellow_count / valid_gt_count * 100) if valid_gt_count > 0 else 0
    yellow_validity = (yellow_before_impact_count / yellow_count * 100) if yellow_count > 0 else 0

    # 4. 打印结果
    print("-" * 60)
    print(f"GT 总有效样本数: {valid_gt_count}")
    print(f"模型识别为 Red 数: {red_detected_count}")
    print(f"模型识别为 Yellow 数: {yellow_count}")
    print("-" * 60)
    print(f"危险检测率 (DR, Red Only):     {dr:.2f}%")
    print(f"有效告警准确率 (EP, Red Time):  {ep:.2f}%")
    print(f"平均安全得分 (Total Avg Score): {avg_score:.4f}")
    print(f"预警比例 (Yellow Rate):        {yellow_rate:.2f}%")
    print(f"预警有效性 (Yellow < Impact):  {yellow_validity:.2f}%")
    print("-" * 60)

if __name__ == "__main__":
    evaluate()