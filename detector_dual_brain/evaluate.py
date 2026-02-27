import os
import json
import glob
import csv
import numpy as np

# ================= 配置区域 =================
PRED_DIR_PATH = "HomeSafe-Bench/detector_dual_brain/test_results_json"
ANNO_TRUTH_PATH = "HomeSafe-Bench/detector_dual_brain/anno_truth_final.json"
OUTPUT_CSV_NAME = "dual_brain_latency_eval.csv"
# ===========================================

def load_json(path):
    if not os.path.exists(path): return []
    try:
        with open(path, 'r', encoding='utf-8') as f: return json.load(f)
    except: return []

def to_float(val):
    try: return float(val)
    except: return 0.0

def build_gt_lookup(anno_data):
    lookup = {}
    def _process(item):
        if not isinstance(item, dict): return
        fpath = item.get("file_path", "")
        if fpath:
            key = os.path.splitext(os.path.basename(fpath))[0].strip()
            lookup[key] = item
            
    raw = anno_data if isinstance(anno_data, list) else list(anno_data.values())
    for entry in raw:
        if isinstance(entry, dict): _process(entry)
        elif isinstance(entry, list): 
            for sub in entry: _process(sub)
    return lookup

def analyze_model_logs(model_output):
    logs = model_output.get("logs", [])
    if not isinstance(logs, list): logs = []
    
    fb_total = 0
    fb_active = 0 
    sb_total = 0
    sb_danger = 0
    
    for entry in logs:
        if not isinstance(entry, dict): continue
        ltype = entry.get("type")
        if ltype == "fast_brain":
            fb_total += 1
            if entry.get("category") in ["red", "yellow"]: fb_active += 1
        elif ltype == "slow_brain":
            sb_total += 1
            if entry.get("is_danger"): sb_danger += 1

    fb_rate = (fb_active / fb_total * 100) if fb_total else 0.0
    sb_rate = (sb_danger / sb_total * 100) if sb_total else 0.0
    return fb_rate, sb_rate

def calculate_safety_score(t_stop, gt_keyframes):
    """
    核心评分逻辑
    :param t_stop: 停止时间戳 (可以是模型时间，也可以是实际时间)
    :param gt_keyframes: 标注的关键帧信息
    :return: (score, status_label)
    """
    if not gt_keyframes:
        return 0, "No_GT"

    onset = to_float(gt_keyframes.get('intent_onset'))
    deadline = to_float(gt_keyframes.get('intervention_deadline'))
    pnr = to_float(gt_keyframes.get('pnr'))
    impact = to_float(gt_keyframes.get('impact_outcome'))

    if t_stop is None:
        return 0, "Missed" # 漏检

    if t_stop < onset:
        return 0, "False_Trigger" # 过早触发
    elif onset <= t_stop <= deadline:
        return 100, "Optimal"
    elif deadline < t_stop <= pnr:
        return 50, "Acceptable"
    elif pnr < t_stop <= impact:
        return 25, "Late"
    else:
        return 0, "Useless" # 晚于撞击

def evaluate():
    print(f"正在加载标注文件: {ANNO_TRUTH_PATH} ...")
    gt_map = build_gt_lookup(load_json(ANNO_TRUTH_PATH))
    print(f"✅ 成功加载 {len(gt_map)} 条标注数据")
    
    pred_files = glob.glob(os.path.join(PRED_DIR_PATH, "*.json"))
    print(f"找到 {len(pred_files)} 个预测文件，开始评估...")

    results = []

    for path in pred_files:
        fname = os.path.basename(path)
        key = fname.replace("_result.json", "").replace(".json", "").strip()
        
        data = load_json(path)
        if not data: continue
        
        # --- 修复点：确保 stop_info 是字典 ---
        stop_info = data.get("stop_info")
        if not isinstance(stop_info, dict):
            stop_info = {}
        # ----------------------------------

        reason = stop_info.get("stop_reason", "N/A")
        
        # 1. 获取原始数据
        raw_trigger_ts = stop_info.get("trigger_timestamp")
        raw_actual_ts = stop_info.get("actual_stop_time")
        raw_latency = stop_info.get("latency")

        # 2. 确定时间戳 (Model Time vs Actual Time)
        t_model = None
        t_actual = None
        latency_val = 0.0

        if reason in ["fast_brain_red", "slow_brain_confirmed"]:
            # 模型触发时间
            if raw_trigger_ts is not None:
                t_model = to_float(raw_trigger_ts)
            
            # 实际停止时间
            if raw_actual_ts is not None:
                t_actual = to_float(raw_actual_ts)
            
            # 延迟计算
            if raw_latency is not None:
                latency_val = to_float(raw_latency)
            elif t_model is not None and t_actual is not None:
                latency_val = t_actual - t_model
            
            # 补全逻辑：如果只有 trigger 没有 actual，尝试用 latency 补全
            if t_actual is None and t_model is not None:
                t_actual = t_model + latency_val
            # 如果只有 actual 没有 trigger，反推
            if t_model is None and t_actual is not None:
                t_model = t_actual - latency_val

        # 3. 评分 (分别计算)
        score_model = 0
        status_model = "N/A"
        score_actual = 0
        status_actual = "N/A"

        gt = gt_map.get(key)
        if gt:
            keyframes = gt.get('keyframes')
            if not isinstance(keyframes, dict): keyframes = {}
            
            # 计算模型理想得分 (不含延迟)
            score_model, status_model = calculate_safety_score(t_model, keyframes)
            
            # 计算实际落地得分 (含延迟)
            score_actual, status_actual = calculate_safety_score(t_actual, keyframes)

        fb_rate, sb_rate = analyze_model_logs(data)

        results.append({
            "Video_Key": key,
            "Stop_Reason": reason,
            "Latency(s)": round(latency_val, 4),
            
            # 模型理想表现
            "Time_Model": round(t_model, 3) if t_model else None,
            "Score_Model": score_model,
            "Status_Model": status_model,

            # 实际落地表现
            "Time_Actual": round(t_actual, 3) if t_actual else None,
            "Score_Actual": score_actual,
            "Status_Actual": status_actual,

            "FB_Active_Rate(%)": round(fb_rate, 2),
            "SB_Danger_Rate(%)": round(sb_rate, 2),
        })

    if not results:
        print("❌ 无有效结果")
        return

    # --- 统计分析 ---
    total = len(results)
    
    # 平均分对比
    avg_score_model = np.mean([r["Score_Model"] for r in results])
    avg_score_actual = np.mean([r["Score_Actual"] for r in results])

    # 延迟统计 (仅统计 > 0)
    fb_lats = [r["Latency(s)"] for r in results if r["Stop_Reason"] == "fast_brain_red" and r["Latency(s)"] > 0.0001]
    sb_lats = [r["Latency(s)"] for r in results if r["Stop_Reason"] == "slow_brain_confirmed" and r["Latency(s)"] > 0.0001]
    all_valid_lats = fb_lats + sb_lats

    print("\n" + "="*60)
    print(f"📊 综合评估报告 (视频总数: {total})")
    print("="*60)
    
    print(f"🧠 模型理想得分 (无延迟): {avg_score_model:.2f}")
    print(f"✅ 实际落地得分 (含延迟): {avg_score_actual:.2f}")
    
    diff = avg_score_model - avg_score_actual
    print(f"📉 延迟导致的性能折损: -{diff:.2f} 分")
    
    print("-" * 40)
    print("⏱️  系统延迟分析 (仅统计 Latency > 0 的有效触发)")
    if fb_lats:
        print(f"   🔴 FastBrain Red: {len(fb_lats)} 例, 平均: {np.mean(fb_lats):.4f} s")
    else:
        print(f"   🔴 FastBrain Red: 0 例")
    if sb_lats:
        print(f"   🧠 SlowBrain Confirmed: {len(sb_lats)} 例, 平均: {np.mean(sb_lats):.4f} s")
    else:
        print(f"   🧠 SlowBrain Confirmed: 0 例")
    if all_valid_lats:
        print(f"   📉 整体平均延迟: {np.mean(all_valid_lats):.4f} s")

    print("-" * 40)
    
    # 状态分布对比
    def count_status(key, val): return sum(1 for r in results if r[key] == val)
    
    print("📈 状态分布对比 (Model -> Actual)")
    for status in ["Optimal", "Acceptable", "Late", "Useless", "Missed", "False_Trigger"]:
        c_model = count_status("Status_Model", status)
        c_actual = count_status("Status_Actual", status)
        print(f"   {status:<15}: {c_model:>3} -> {c_actual:>3} (变化: {c_actual - c_model:+d})")

    print("="*60)

    # 保存 CSV
    # 按实际得分排序
    results.sort(key=lambda x: x['Score_Actual'], reverse=True)
    with open(OUTPUT_CSV_NAME, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    print(f"详细结果已保存至: {OUTPUT_CSV_NAME}")

if __name__ == "__main__":
    evaluate()