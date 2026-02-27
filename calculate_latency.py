import json
import os
import glob
import numpy as np

MODEL_1_PATH = "HomeSafe-Bench/test/results/latency/results_streaming_detection_qwen3-omni-30B-A3B-instruct_delay_time.json"
MODEL_2_PATH = "HomeSafe-Bench/test/results/latencyresults_minicpm-o-4.5-streaming_delay_time.json"
MODEL_3_PATH = "HomeSafe-Bench/test/results/latency/results_zero_shot_minicpm-o-2.6-streaming_delay_time.json"
MODEL_4_DIR = "HomeSafe-Bench/test/results/latency/test_results_final_new_new_promptv21_json"
GT_PATH = "HomeSafe-Bench/detector_dual_brain_vllm/anno_truth_final.json"

def load_json(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}

def get_video_categories(gt_path):
    gt_data = load_json(gt_path)
    mapping = {}
    if 'annotations' not in gt_data: return mapping
    for item in gt_data['annotations']:
        file_path = item.get('file_path', '')
        filename = os.path.basename(file_path)
        difficulty = item.get('reasoning_difficulty', '')
        if difficulty == 'D3':
            mapping[filename] = 'case2'
        else:
            mapping[filename] = 'case1'
    return mapping

def process_model_1(path):
    data = load_json(path)
    latencies = {}
    for filename, content in data.items():
        frames = content.get('frames', [])
        for frame in frames:
            if frame.get('signal') == 'Red':
                latencies[filename] = frame.get('latency_seconds')
                break
    return latencies

def process_minicpm(path):
    data = load_json(path)
    latencies = {}
    for filename, content in data.items():
        if content.get('is_dangerous') is True:
            metrics = content.get('latency_metrics', {})
            lat = metrics.get('latency')
            if lat is not None:
                latencies[filename] = lat
    return latencies

def process_dual_brain(folder_path):
    latencies = {}
    if not os.path.exists(folder_path): return latencies
    json_files = glob.glob(os.path.join(folder_path, "*_result.json"))
    for file_path in json_files:
        try:
            content = load_json(file_path)
            video_path = content.get('video_path')
            if video_path:
                filename = os.path.basename(video_path)
            else:
                base_name = os.path.basename(file_path)
                filename = base_name.replace('_result.json', '.mp4')
            stop_info = content.get('stop_info', {})
            if stop_info and 'latency' in stop_info:
                lat = stop_info['latency']
                if lat is not None:
                    latencies[filename] = lat
        except: pass
    return latencies


def print_stats(name, latencies, category_map, video_subset=None):
    """
    计算统计信息
    video_subset: 如果提供，只计算在这个列表里的视频 (用于交集计算)
    """
    case1_vals = []
    case2_vals = []
    total_vals = []

    target_videos = video_subset if video_subset is not None else latencies.keys()

    for filename in target_videos:
        if filename not in latencies:
            continue
            
        val = latencies[filename]
        cat = category_map.get(filename)
        
        total_vals.append(val)
        if cat == 'case1':
            case1_vals.append(val)
        elif cat == 'case2':
            case2_vals.append(val)

    print(f"--- {name} ---")
    if len(total_vals) == 0:
        print("No data.")
        return

    print(f"  Total : Avg={np.mean(total_vals):.4f}s | Count={len(total_vals)}")
    print(f"  Case 1: Avg={np.mean(case1_vals):.4f}s | Count={len(case1_vals)}")
    print(f"  Case 2: Avg={np.mean(case2_vals):.4f}s | Count={len(case2_vals)}")
    print("")

# ================= 主程序 =================

def main():
    category_map = get_video_categories(GT_PATH)
    
    print("Loading data...")
    lat_1 = process_model_1(MODEL_1_PATH) if os.path.exists(MODEL_1_PATH) else {}
    lat_2 = process_minicpm(MODEL_2_PATH) if os.path.exists(MODEL_2_PATH) else {}
    lat_3 = process_minicpm(MODEL_3_PATH) if os.path.exists(MODEL_3_PATH) else {}
    lat_4 = process_dual_brain(MODEL_4_DIR) if os.path.exists(MODEL_4_DIR) else {}


    common_videos = set(lat_1.keys()) & set(lat_2.keys()) & set(lat_3.keys()) & set(lat_4.keys())
    
    print(f"\n==========================================")
    print(f"PART 1: 原始结果 (Count不同代表检出率不同)")
    print(f"==========================================")
    print_stats("Model 1 (Frame-based)", lat_1, category_map)
    print_stats("Model 2 (MiniCPM-o-4.5)", lat_2, category_map)
    print_stats("Model 3 (MiniCPM-o-2.6)", lat_3, category_map)
    print_stats("Model 4 (Dual Brain)", lat_4, category_map)

    print(f"==========================================")
    print(f"PART 2: 公平对比 (仅计算 4 个模型都检出的 {len(common_videos)} 个视频)")
    print(f"==========================================")
    
    if len(common_videos) > 0:
        print_stats("Model 1 (Common Set)", lat_1, category_map, common_videos)
        print_stats("Model 2 (Common Set)", lat_2, category_map, common_videos)
        print_stats("Model 3 (Common Set)", lat_3, category_map, common_videos)
        print_stats("Model 4 (Common Set)", lat_4, category_map, common_videos)
    else:
        print("没有找到四个模型都共同检出的视频，无法进行交集对比。")

if __name__ == "__main__":
    main()