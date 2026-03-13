import os
import json
import cv2

assets_dir = "/Users/pujiayue/EmbodiedSafety/HomeSafe-Bench/assets"
json_path = "/Users/pujiayue/EmbodiedSafety/HomeSafe-Bench/anno_truth.json"

output_root = os.path.join(assets_dir, "keyframes")
os.makedirs(output_root, exist_ok=True)

with open(json_path, "r") as f:
    data = json.load(f)

annotations = data["annotations"]

for ann in annotations:

    if not ann["is_valid"]:
        continue

    video_name = ann["video_name"]
    video_path = os.path.join(assets_dir, video_name)

    if not os.path.exists(video_path):
        print(f"⚠️ Video not found: {video_name}")
        continue

    keyframes = ann["keyframes"]

    times = {
        "intent_onset": keyframes["intent_onset"],
        "intervention_deadline": keyframes["intervention_deadline"],
        "pnr": keyframes["pnr"],
        "impact_outcome": keyframes["impact_outcome"]
    }

    video_id = video_name.replace(".mp4", "")
    out_dir = os.path.join(output_root, video_id)
    os.makedirs(out_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)

    for name, t in times.items():

        frame_id = int(t * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)

        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed {video_name} {name}")
            continue

        # 在文件名加入秒数
        time_str = f"{t:.1f}s"
        save_name = f"{name}_{time_str}.jpg"

        save_path = os.path.join(out_dir, save_name)

        cv2.imwrite(save_path, frame)

        print(f"Saved: {save_path}")

    cap.release()

print("✅ Done extracting keyframes.")