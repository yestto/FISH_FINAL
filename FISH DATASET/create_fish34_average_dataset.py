import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

def load_calibration(dataset_root: str):
    import os
    cm_ground_path = os.path.join(dataset_root, "fish01", "cm_ground.npy")
    cm_vertical_path = os.path.join(dataset_root, "fish01", "cm_vertical.npy")
    cm_ground = float(np.load(cm_ground_path))
    cm_vertical = float(np.load(cm_vertical_path))
    return cm_ground, cm_vertical

def mask_to_points(mask: np.ndarray):
    pts = cv2.findNonZero(mask)
    if pts is not None:
        return pts.squeeze(1)
    return None

def measure_top(mask: np.ndarray, cm_per_px: float):
    pts = mask_to_points(mask)
    if pts is None or len(pts) < 30:
        return float("nan"), float("nan"), float("nan"), float("nan")
    hull = cv2.convexHull(pts.reshape(-1, 1, 2))
    area_px = float(cv2.contourArea(hull))
    peri_px = float(cv2.arcLength(hull, True))
    rect = cv2.minAreaRect(hull)
    w, h = rect[1]
    L_px, W_px = (max(w, h), min(w, h))
    L_cm = L_px * cm_per_px
    W_cm = W_px * cm_per_px
    area_cm2 = area_px * (cm_per_px**2)
    peri_cm = peri_px * cm_per_px
    return L_cm, W_cm, area_cm2, peri_cm

def measure_height(mask: np.ndarray, cm_per_px: float) -> float:
    pts = mask_to_points(mask)
    if pts is None or len(pts) < 30:
        return float("nan")
    ys = pts[:, 1]
    return float((ys.max() - ys.min()) * cm_per_px)

def enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

def smart_sam_lowconf(predictor, frame, box):
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = [int(v) for v in box]
    bw, bh = x2-x1, y2-y1
    pad = max(bw, bh)
    cx1, cy1 = max(0, x1-pad), max(0, y1-pad)
    cx2, cy2 = min(W, x2+pad), min(H, y2+pad)
    crop = frame[cy1:cy2, cx1:cx2].copy()
    crop_enh = enhance(crop)
    rx1, ry1 = x1-cx1, y1-cy1
    rx2, ry2 = x2-cx1, y2-cy1
    rcx, rcy = (rx1+rx2)/2, (ry1+ry2)/2
    points = np.array([
        [rcx, rcy],
        [rcx, rcy - bh*0.15],
        [rcx, rcy + bh*0.15],
        [rx1, ry1],
        [rx2, ry1],
        [rx1, ry2],
        [rx2, ry2],
    ])
    labels = np.array([1, 1, 1, 0, 0, 0, 0])
    predictor.set_image(cv2.cvtColor(crop_enh, cv2.COLOR_BGR2RGB))
    masks, scores, _ = predictor.predict(
        point_coords=points,
        point_labels=labels,
        box=np.array([rx1, ry1, rx2, ry2]),
        multimask_output=True
    )
    best_idx = np.argmax(scores)
    crop_mask = (masks[best_idx] * 255).astype(np.uint8)
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[cy1:cy2, cx1:cx2] = crop_mask
    restricted = np.zeros_like(full_mask)
    p = 5
    ry1c, ry2c = max(0, y1-p), min(H, y2+p)
    rx1c, rx2c = max(0, x1-p), min(W, x2+p)
    restricted[ry1c:ry2c, rx1c:rx2c] = full_mask[ry1c:ry2c, rx1c:rx2c]
    contours, _ = cv2.findContours(restricted, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        result = np.zeros_like(restricted)
        cv2.drawContours(result, [c], -1, 255, cv2.FILLED)
        return result
    return restricted

def remove_fins_gentle(mask, bbox):
    x1, y1, x2, y2 = [int(v) for v in bbox]
    min_dim = min(x2-x1, y2-y1)
    pct = 0.04 if min_dim < 50 else (0.06 if min_dim < 100 else 0.10)
    k = max(3, int(min_dim * pct))
    if k % 2 == 0: k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        final = np.zeros_like(clean)
        cv2.drawContours(final, [c], -1, 255, cv2.FILLED)
        return final
    return clean

def main():
    dataset_root = r'C:\Users\shain\Downloads\FISH DATASET\FISH DATASET'
    yolo_path = r'C:\Users\shain\Downloads\FISH DATASET\FISH-TRAIN\Fish Annotations\yolo_training_runs\fish_yolo12m_full\weights\best.pt'
    sam_checkpoint = os.path.join(dataset_root, r'sam_vit_b_01ec64.pth')
    
    yolo_model = YOLO(yolo_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    cm_ground, cm_vertical = load_calibration(dataset_root)

    top_vid = list(Path(dataset_root).joinpath('fish3+4', 'top view').glob('*.mp4'))[0]
    front_vid = list(Path(dataset_root).joinpath('fish3+4', 'front view').glob('*.mp4'))[0]
    
    cap_top = cv2.VideoCapture(str(top_vid))
    cap_front = cv2.VideoCapture(str(front_vid))

    target_frames = [3265, 6225, 3280, 3315, 3275]
    
    fish3_measurements = []
    fish4_measurements = []
    
    for idx in target_frames:
        cap_top.set(cv2.CAP_PROP_POS_FRAMES, idx)
        cap_front.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok_t, top_f = cap_top.read()
        ok_f, front_f = cap_front.read()
        if not ok_t or not ok_f: continue

        # TOP VIEW
        res_top = yolo_model(top_f, verbose=False, imgsz=640)
        boxes_top = res_top[0].boxes.xyxy.cpu().numpy()
        confs_top = res_top[0].boxes.conf.cpu().numpy()
        
        top_metrics = []
        for j, box in enumerate(boxes_top):
            if j >= 2: break
            conf = confs_top[j]
            if conf < 0.5:
                clean_mask = smart_sam_lowconf(sam_predictor, top_f, box)
            else:
                sam_predictor.set_image(cv2.cvtColor(top_f, cv2.COLOR_BGR2RGB))
                cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                masks, scores, _ = sam_predictor.predict(
                    point_coords=np.array([[cx, cy]]), point_labels=np.array([1]),
                    box=np.array([int(x) for x in box]), multimask_output=True
                )
                raw_mask = (masks[np.argmax(scores)] * 255).astype(np.uint8)
                clean_mask = remove_fins_gentle(raw_mask, box)
                
            L, W, A, P = measure_top(clean_mask, cm_ground)
            top_metrics.append({'L': L, 'W': W, 'A': A, 'P': P, 'box': box})
            
        # Sort top metrics by Length (larger = Fish3, smaller = Fish4)
        top_metrics.sort(key=lambda x: x['L'], reverse=True)
        if len(top_metrics) < 2: continue
        
        # FRONT VIEW
        res_front = yolo_model(front_f, verbose=False, imgsz=640)
        boxes_front = res_front[0].boxes.xyxy.cpu().numpy()
        confs_front = res_front[0].boxes.conf.cpu().numpy()
        
        front_metrics = []
        for j, box in enumerate(boxes_front):
            if j >= 2: break
            conf = confs_front[j]
            if conf < 0.5:
                clean_mask = smart_sam_lowconf(sam_predictor, front_f, box)
            else:
                sam_predictor.set_image(cv2.cvtColor(front_f, cv2.COLOR_BGR2RGB))
                cx, cy = (box[0]+box[2])/2, (box[1]+box[3])/2
                masks, scores, _ = sam_predictor.predict(
                    point_coords=np.array([[cx, cy]]), point_labels=np.array([1]),
                    box=np.array([int(x) for x in box]), multimask_output=True
                )
                raw_mask = (masks[np.argmax(scores)] * 255).astype(np.uint8)
                clean_mask = remove_fins_gentle(raw_mask, box)
                
            H = measure_height(clean_mask, cm_vertical)
            # Use bbox width approximation to sort if needed, or sort by height/area
            # Since front view width corresponds to length, let's estimate length in front view
            front_L = (box[2] - box[0]) * cm_ground # rough estimation
            front_metrics.append({'H': H, 'front_L': front_L})
            
        # Sort front metrics by front_L (larger = Fish3, smaller = Fish4)
        front_metrics.sort(key=lambda x: x['front_L'], reverse=True)
        if len(front_metrics) < 2: continue
        
        # Aggregate for Fish 3
        fish3_measurements.append({
            'Frame': idx,
            'Length (cm)': top_metrics[0]['L'],
            'Width (cm)': top_metrics[0]['W'],
            'Height (cm)': front_metrics[0]['H'],
            'Area (cm²)': top_metrics[0]['A'],
            'Perimeter (cm)': top_metrics[0]['P']
        })
        
        # Aggregate for Fish 4
        fish4_measurements.append({
            'Frame': idx,
            'Length (cm)': top_metrics[1]['L'],
            'Width (cm)': top_metrics[1]['W'],
            'Height (cm)': front_metrics[1]['H'],
            'Area (cm²)': top_metrics[1]['A'],
            'Perimeter (cm)': top_metrics[1]['P']
        })

    cap_top.release()
    cap_front.release()
    
    # Calculate Averages
    def avg_metrics(measurements):
        df = pd.DataFrame(measurements)
        return df.mean().to_dict()
        
    fish3_avg = avg_metrics(fish3_measurements)
    fish4_avg = avg_metrics(fish4_measurements)
    
    # Add True Weights
    fish3_avg['FishID'] = 'fish3'
    fish3_avg['Weight (g)'] = 81.90
    fish3_avg['Frames_Averaged'] = 5
    
    fish4_avg['FishID'] = 'fish4'
    fish4_avg['Weight (g)'] = 61.25
    fish4_avg['Frames_Averaged'] = 5
    
    final_df = pd.DataFrame([fish3_avg, fish4_avg])
    cols = ['FishID', 'Frames_Averaged', 'Weight (g)', 'Length (cm)', 'Width (cm)', 'Height (cm)', 'Area (cm²)', 'Perimeter (cm)']
    final_df = final_df[cols]
    
    out_csv = os.path.join(dataset_root, 'fish34_combined_average.csv')
    final_df.to_csv(out_csv, index=False)
    
    print("=== Final Averaged Measurements ===")
    print(final_df.to_string(index=False))
    print(f"\nSaved to: {out_csv}")

if __name__ == '__main__':
    import os
    main()
