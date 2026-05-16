"""
Find the BEST synchronized frames from fish3+4 videos where:
- Exactly 2 fish are detected in BOTH top and front views
- No overlapping bounding boxes
- Both fish clearly visible and well-separated
- High confidence, high sharpness, maximum separation
"""

import cv2
import json
import numpy as np
from pathlib import Path
from ultralytics import YOLO

yolo_path = r"C:\Users\shain\Downloads\FISH DATASET\FISH-TRAIN\Fish Annotations\yolo_training_runs\fish_yolo12m_full\weights\best.pt"
yolo_model = YOLO(yolo_path)

dataset_root = Path(r"C:\Users\shain\Downloads\FISH DATASET\FISH DATASET")
top_vid = list((dataset_root / "fish3+4" / "top view").glob("*.mp4"))[0]
front_vid = list((dataset_root / "fish3+4" / "front view").glob("*.mp4"))[0]

cap_top = cv2.VideoCapture(str(top_vid))
cap_front = cv2.VideoCapture(str(front_vid))
total = min(int(cap_top.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap_front.get(cv2.CAP_PROP_FRAME_COUNT)))
print(f"Total synced frames: {total}")


def iou(a, b):
    x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
    x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter + 1e-6)


def blur_score(img):
    return cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def box_area(b):
    return (b[2] - b[0]) * (b[3] - b[1])


def separation(boxes):
    c1 = ((boxes[0][0] + boxes[0][2]) / 2, (boxes[0][1] + boxes[0][3]) / 2)
    c2 = ((boxes[1][0] + boxes[1][2]) / 2, (boxes[1][1] + boxes[1][3]) / 2)
    return np.sqrt((c1[0] - c2[0]) ** 2 + (c1[1] - c2[1]) ** 2)


candidates = []
stride = 5  # Check every 5th frame for speed

for idx in range(0, total, stride):
    cap_top.set(cv2.CAP_PROP_POS_FRAMES, idx)
    cap_front.set(cv2.CAP_PROP_POS_FRAMES, idx)
    ok_t, top_f = cap_top.read()
    ok_f, front_f = cap_front.read()
    if not ok_t or not ok_f:
        continue

    res_t = yolo_model(top_f, verbose=False, imgsz=640)
    res_f = yolo_model(front_f, verbose=False, imgsz=640)

    boxes_t = res_t[0].boxes.xyxy.cpu().numpy()
    confs_t = res_t[0].boxes.conf.cpu().numpy()
    boxes_f = res_f[0].boxes.xyxy.cpu().numpy()
    confs_f = res_f[0].boxes.conf.cpu().numpy()

    # STRICT: Exactly 2 fish detected in BOTH views
    if len(boxes_t) != 2 or len(boxes_f) != 2:
        continue

    # No overlapping in top view
    overlap_t = iou(boxes_t[0], boxes_t[1])
    if overlap_t > 0.05:
        continue

    # No overlapping in front view
    overlap_f = iou(boxes_f[0], boxes_f[1])
    if overlap_f > 0.05:
        continue

    # Both boxes must be reasonably sized (not tiny false detections)
    min_area = 500
    if any(box_area(b) < min_area for b in boxes_t) or any(box_area(b) < min_area for b in boxes_f):
        continue

    # Score: high confidence + high separation + high sharpness + low overlap
    avg_conf = (confs_t.mean() + confs_f.mean()) / 2
    sep_t = separation(boxes_t)
    sep_f = separation(boxes_f)
    blur_t = blur_score(top_f)
    blur_f = blur_score(front_f)

    score = (
        (avg_conf * 100)
        + (sep_t * 0.5)
        + (sep_f * 0.5)
        + (blur_t * 0.01)
        + (blur_f * 0.01)
        - (overlap_t * 500)
        - (overlap_f * 500)
    )

    candidates.append({
        "frame": idx,
        "score": float(score),
        "conf_avg": float(avg_conf),
        "sep_top": float(sep_t),
        "sep_front": float(sep_f),
        "overlap_top": float(overlap_t),
        "overlap_front": float(overlap_f),
        "blur_top": float(blur_t),
        "blur_front": float(blur_f),
    })

    if len(candidates) % 10 == 0:
        print(f"  Scanned {idx}/{total}, found {len(candidates)} candidates so far...")

cap_top.release()
cap_front.release()

# Sort by score and pick top 10
candidates.sort(key=lambda x: x["score"], reverse=True)
top10 = candidates[:10]

print(f"\n{'=' * 70}")
print(f"RESULTS")
print(f"{'=' * 70}")
print(f"Total frames scanned: {total // stride}")
print(f"Candidates (2 fish, no overlap, both views): {len(candidates)}")
print(f"\nTOP 10 BEST FRAMES:")
for i, c in enumerate(top10):
    fr = c["frame"]
    sc = c["score"]
    co = c["conf_avg"]
    st = c["sep_top"]
    sf = c["sep_front"]
    ot = c["overlap_top"]
    of_ = c["overlap_front"]
    bt = c["blur_top"]
    bf = c["blur_front"]
    print(f"  #{i+1}: Frame {fr:5d} | Score={sc:.1f} | Conf={co:.3f} | SepT={st:.0f} SepF={sf:.0f} | OvT={ot:.3f} OvF={of_:.3f} | BlurT={bt:.0f} BlurF={bf:.0f}")

# Save the list
out_path = dataset_root / "mask_comparison_results" / "best_fish34_frames.json"
with open(str(out_path), "w") as f:
    json.dump(top10, f, indent=2)
print(f"\nSaved frame list to: {out_path}")
