import json
import os
import re

# ================= Configuration Area =================
GT_FILE_PATH = "anno_truth.json"
PRED_FILE_PATH = "results_streaming_detection_qwen3-omni-30B-A3B-instruct.json"
# ===========================================

def extract_number(s):
    """Extract the first floating-point number from a string or list item"""
    try:
        match = re.search(r"[-+]?\d*\.\d+|\d+", str(s))
        return float(match.group()) if match else None
    except:
        return None

def get_mapping_key(ann):
    """Construct a key based on GT metadata to match the filename format of the model output"""
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
        print("Error: File path does not exist. Please check the path settings inside the script.")
        return

    try:
        with open(GT_FILE_PATH, 'r', encoding='utf-8') as f:
            gt_data = json.load(f)
        with open(PRED_FILE_PATH, 'r', encoding='utf-8') as f:
            pred_data = json.load(f)
    except Exception as e:
        print(f"Error parsing JSON: {e}")
        return

    # 1. Build GT mapping (only process samples where is_valid is True)
    gt_map = {}
    valid_gt_count = 0
    for ann in gt_data['annotations']:
        if ann.get('is_valid') is True:
            valid_gt_count += 1
            meta_key = get_mapping_key(ann)
            if meta_key:
                gt_map[meta_key] = ann
            base_name = os.path.basename(ann['file_path']).lower().replace('.mp4', '').strip()
            gt_map[base_name] = ann

    # Statistics variables
    red_detected_count = 0      # Only cases where status is red and has timestamp
    strict_effective_count = 0  # red and timestamp is within [onset, impact]
    total_weighted_score = 0    # Total score
    
    yellow_count = 0            # Count of yellow
    yellow_before_impact_count = 0 # yellow and timestamp before impact

    # 2. Iterate using GT as the reference to ensure every valid video contributes to the denominator
    # Note: the purpose of iterating gt_map here is to ensure all valid samples are counted
    # However, since the prediction file uses filenames as keys, we directly iterate through predictions.
    # As long as it exists in valid GT, we score it; otherwise it is not counted.
    
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
        
        # If the video is not in the valid GT list, skip it
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

        # Core logic: only RED status attempts to obtain score
        if status == "red":
            red_detected_count += 1
            if t_pred is not None:
                # Determine whether it falls into the valid time interval
                if onset <= t_pred <= impact:
                    strict_effective_count += 1

                # Detailed scoring
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
                # status is red but no timestamp is provided, score is 0
                pass
        
        elif status == "yellow":
            yellow_count += 1
            if t_pred is not None and t_pred < impact:
                yellow_before_impact_count += 1
            # Yellow contributes 0 score, so no total_weighted_score addition
        
        else:
            # status is unknown or others, contributes 0 score
            pass

    # 3. Metric calculation
    # Danger Detection Rate (DR): numerator is number of red predictions, denominator is all valid GT samples
    dr = (red_detected_count / valid_gt_count * 100) if valid_gt_count > 0 else 0
    
    # Effective Prediction Rate (EP): among predicted red cases, proportion with correct time
    ep = (strict_effective_count / red_detected_count * 100) if red_detected_count > 0 else 0
    
    # Average score: all valid videos (including red, yellow, unknown) are the denominator
    # yellow and unknown contribute 0 because they never add to total_weighted_score
    avg_score = (total_weighted_score / valid_gt_count) if valid_gt_count > 0 else 0
    
    # Yellow-related metrics
    yellow_rate = (yellow_count / valid_gt_count * 100) if valid_gt_count > 0 else 0
    yellow_validity = (yellow_before_impact_count / yellow_count * 100) if yellow_count > 0 else 0

    # 4. Print results
    print("-" * 60)
    print(f"Total valid GT samples: {valid_gt_count}")
    print(f"Predicted as Red by model: {red_detected_count}")
    print(f"Predicted as Yellow by model: {yellow_count}")
    print("-" * 60)
    print(f"Danger Detection Rate (DR, Red Only):     {dr:.2f}%")
    print(f"Effective Prediction Accuracy (EP, Red Time):  {ep:.2f}%")
    print(f"Average Safety Score (Total Avg Score): {avg_score:.4f}")
    print(f"Warning Ratio (Yellow Rate):        {yellow_rate:.2f}%")
    print(f"Warning Validity (Yellow < Impact):  {yellow_validity:.2f}%")
    print("-" * 60)

if __name__ == "__main__":
    evaluate()