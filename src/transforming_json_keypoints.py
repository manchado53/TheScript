import os
import json

# === CONFIG ===
json_path = "/data/ai_club/SoccerStats2024/key_points/Labels2ndDataSetVisited.json"
output_path = "/data/ai_club/SoccerStats2024/key_points/Labels2ndDataSetVisited_COCO.json"

keypoint_names = [
    "Left Center Circle",
    "Right Center Circle",
    "Top Center Circle",
    "Bottom Center Circle",
    "Top Midfield",
    "Bottom Midfield",
    "Left D",
    "Right D",
    "Left D Top Intersection",
    "Left D Bottom Intersection",
    "Right D Top Intersection",
    "Right D Bottom Intersection"
]
keypoint_index = {name: i for i, name in enumerate(keypoint_names)}
num_keypoints = len(keypoint_names)

# === Create COCO structure ===
coco = {
    "images": [],
    "annotations": [],
    "categories": [{
        "id": 1,
        "name": "fieldpoints",
        "supercategory": "keypoint",
        "keypoints": keypoint_names,
        "skeleton": []  # You can add connections here if needed
    }]
}

# === Load Label Studio JSON ===
with open(json_path, "r") as f:
    data = json.load(f)

ann_id = 1
for img_id, item in enumerate(data):
    filename = item["data"]["img"].split("/")[-1]
    annotation = item["annotations"][0]["result"]
    width = annotation[0]["original_width"]
    height = annotation[0]["original_height"]

    # Add image entry
    coco["images"].append({
        "id": img_id,
        "file_name": filename,
        "width": width,
        "height": height
    })

    # Build keypoint list
    keypoints = [0] * (num_keypoints * 3)
    xs, ys = [], []

    for kp in annotation:
        label = kp["value"]["keypointlabels"][0]
        if label not in keypoint_index:
            continue
        idx = keypoint_index[label]
        x_abs = (kp["value"]["x"] / 100) * width
        y_abs = (kp["value"]["y"] / 100) * height
        keypoints[idx * 3] = x_abs
        keypoints[idx * 3 + 1] = y_abs
        keypoints[idx * 3 + 2] = 2  # visible
        xs.append(x_abs)
        ys.append(y_abs)

    if xs and ys:
        x_min = min(xs)
        y_min = min(ys)
        box_w = max(xs) - x_min
        box_h = max(ys) - y_min
        bbox = [x_min, y_min, box_w, box_h]

        coco["annotations"].append({
            "id": ann_id,
            "image_id": img_id,
            "category_id": 1,
            "keypoints": keypoints,
            "num_keypoints": sum(1 for v in keypoints[2::3] if v > 0),
            "bbox": bbox,
            "area": box_w * box_h,
            "iscrowd": 0
        })
        ann_id += 1

# === Save COCO JSON ===
with open(output_path, "w") as f:
    json.dump(coco, f, indent=4)

print(f"✅ COCO JSON saved to: {output_path}")
