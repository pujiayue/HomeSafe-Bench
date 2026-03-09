import json

input_file = "anno_truth_video_name.json"
output_file = "anno_truth_final.json"

with open(input_file, "r", encoding="utf-8") as f:
    data = json.load(f)

new_annotations = []

for item in data["annotations"]:
    new_item = {"video_name": item["video_name"]}
    
    for k, v in item.items():
        if k != "video_name":
            new_item[k] = v
            
    new_annotations.append(new_item)

data["annotations"] = new_annotations

with open(output_file, "w", encoding="utf-8") as f:
    json.dump(data, f, indent=2, ensure_ascii=False)

print("Done.")