import cv2
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def mask_to_points(mask: np.ndarray):
    pts = cv2.findNonZero(mask)
    if pts is not None: return pts.squeeze(1)
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
    return L_px * cm_per_px, W_px * cm_per_px, area_px * (cm_per_px**2), peri_px * cm_per_px

def measure_height(mask: np.ndarray, cm_per_px: float) -> float:
    pts = mask_to_points(mask)
    if pts is None or len(pts) < 30: return float("nan")
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
    points = np.array([[rcx, rcy], [rcx, rcy - bh*0.15], [rcx, rcy + bh*0.15],
                       [rx1, ry1], [rx2, ry1], [rx1, ry2], [rx2, ry2]])
    labels = np.array([1, 1, 1, 0, 0, 0, 0])
    predictor.set_image(cv2.cvtColor(crop_enh, cv2.COLOR_BGR2RGB))
    masks, scores, _ = predictor.predict(
        point_coords=points, point_labels=labels,
        box=np.array([rx1, ry1, rx2, ry2]), multimask_output=True
    )
    crop_mask = (masks[np.argmax(scores)] * 255).astype(np.uint8)
    full_mask = np.zeros((H, W), dtype=np.uint8)
    full_mask[cy1:cy2, cx1:cx2] = crop_mask
    return full_mask

def add_derived_features(df):
    L = df["Length (cm)"]
    W = df["Width (cm)"]
    H = df["Height (cm)"]
    A = df["Area (cm²)"]
    P = df["Perimeter (cm)"]
    df["Volume (cm³)"] = (np.pi / 6.0) * L * W * H
    a, b, c = L / 2.0, W / 2.0, H / 2.0
    p = 1.6075
    df["Surface Area (cm²)"] = 4.0 * np.pi * ((a**p * b**p + a**p * c**p + b**p * c**p) / 3.0) ** (1.0 / p)
    df["Aspect Ratio"] = L / np.maximum(W, 1e-9)
    df["Elongation"] = 1.0 - (W / np.maximum(L, 1e-9))
    df["Compactness"] = (4.0 * np.pi * A) / np.maximum(P ** 2, 1e-9)
    df["Condition Factor (K)"] = 1.0 
    df["Rectangularity"] = A / np.maximum(L * W, 1e-9)
    df["Equivalent Diameter (cm)"] = np.sqrt(4.0 * A / np.pi)
    return df

def main():
    import os
    dataset_root = r'C:\Users\shain\Downloads\FISH DATASET\FISH DATASET'
    
    print("Training Ridge model on the full dataset...")
    train_df = pd.read_csv(os.path.join(dataset_root, "fish_frames_200_ENHANCED_clean_unique_no_repeat.csv"))
    feature_cols = [
        "Length (cm)", "Width (cm)", "Height (cm)",
        "Area (cm²)", "Perimeter (cm)",
        "Volume (cm³)", "Surface Area (cm²)",
        "Aspect Ratio", "Elongation", "Compactness",
        "Rectangularity", "Equivalent Diameter (cm)"
    ]
    X_train = train_df[feature_cols].values
    y_train = train_df["Weight (g)"].values
    model = Pipeline([("scaler", StandardScaler()), ("model", Ridge(alpha=1.0))])
    model.fit(X_train, y_train)

    yolo_path = os.path.join(dataset_root, r'..\FISH-TRAIN\Fish Annotations\yolo_training_runs\fish_yolo12m_full\weights\best.pt')
    sam_checkpoint = os.path.join(dataset_root, r'sam_vit_b_01ec64.pth')
    
    yolo_model = YOLO(yolo_path)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    cm_ground = 0.2757
    cm_vertical = 0.0924

    top_vid = list(Path(dataset_root).joinpath('fish3+4', 'top view').glob('*.mp4'))[0]
    front_vid = list(Path(dataset_root).joinpath('fish3+4', 'front view').glob('*.mp4'))[0]
    
    cap_top = cv2.VideoCapture(str(top_vid))
    cap_front = cv2.VideoCapture(str(front_vid))
    total_frames = min(int(cap_top.get(cv2.CAP_PROP_FRAME_COUNT)), int(cap_front.get(cv2.CAP_PROP_FRAME_COUNT)))
    
    # Sample 20 evenly spaced frames for faster processing
    target_frames = np.linspace(100, total_frames - 100, 20).astype(int)
    
    fish3_measurements = []
    fish4_measurements = []
    
    print(f"Extracting features from {len(target_frames)} frames across the video...", flush=True)
    valid_count = 0
    
    for idx in target_frames:
        cap_top.set(cv2.CAP_PROP_POS_FRAMES, idx)
        cap_front.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok_t, top_f = cap_top.read()
        ok_f, front_f = cap_front.read()
        if not ok_t or not ok_f: continue

        res_top = yolo_model(top_f, verbose=False, imgsz=640)
        boxes_top = res_top[0].boxes.xyxy.cpu().numpy()
        
        if len(boxes_top) < 2: continue
        
        top_metrics = []
        for j, box in enumerate(boxes_top[:2]):
            clean_mask = smart_sam_lowconf(sam_predictor, top_f, box)
            L, W, A, P = measure_top(clean_mask, cm_ground)
            if L > 0 and W > 0:
                top_metrics.append({'L': L, 'W': W, 'A': A, 'P': P})
                
        if len(top_metrics) < 2: continue
        top_metrics.sort(key=lambda x: x['L'], reverse=True)
        
        res_front = yolo_model(front_f, verbose=False, imgsz=640)
        boxes_front = res_front[0].boxes.xyxy.cpu().numpy()
        if len(boxes_front) < 2: continue
        
        front_metrics = []
        for j, box in enumerate(boxes_front[:2]):
            clean_mask = smart_sam_lowconf(sam_predictor, front_f, box)
            H = measure_height(clean_mask, cm_vertical)
            front_L = (box[2] - box[0]) * cm_ground
            if H > 0:
                front_metrics.append({'H': H, 'front_L': front_L})
                
        if len(front_metrics) < 2: continue
        front_metrics.sort(key=lambda x: x['front_L'], reverse=True)
        
        fish3_measurements.append({
            'Length (cm)': top_metrics[0]['L'], 'Width (cm)': top_metrics[0]['W'],
            'Height (cm)': front_metrics[0]['H'], 'Area (cm²)': top_metrics[0]['A'], 'Perimeter (cm)': top_metrics[0]['P']
        })
        fish4_measurements.append({
            'Length (cm)': top_metrics[1]['L'], 'Width (cm)': top_metrics[1]['W'],
            'Height (cm)': front_metrics[1]['H'], 'Area (cm²)': top_metrics[1]['A'], 'Perimeter (cm)': top_metrics[1]['P']
        })
        valid_count += 1
        if valid_count % 1 == 0:
            print(f"Processed {valid_count} valid frames...", flush=True)

    cap_top.release()
    cap_front.release()
    
    if valid_count == 0:
        print("No valid frames extracted.")
        return
        
    f3_df = pd.DataFrame(fish3_measurements)
    f3_df = add_derived_features(f3_df)
    f3_df['Predicted_Weight'] = model.predict(f3_df[feature_cols].values)
    
    f4_df = pd.DataFrame(fish4_measurements)
    f4_df = add_derived_features(f4_df)
    f4_df['Predicted_Weight'] = model.predict(f4_df[feature_cols].values)
    
    f3_avg_pred = f3_df['Predicted_Weight'].mean()
    f4_avg_pred = f4_df['Predicted_Weight'].mean()
    
    # Save to CSV
    f3_df['FishID'] = 'fish3'
    f4_df['FishID'] = 'fish4'
    final_df = pd.concat([f3_df, f4_df], ignore_index=True)
    out_csv = os.path.join(dataset_root, 'fish34_all_frames_predictions.csv')
    final_df.to_csv(out_csv, index=False)
    
    print("\n" + "="*50)
    print(f"ESTIMATED WEIGHTS (Averaged across {valid_count} frames)")
    print("="*50)
    print(f"FISH 3 (Larger Fish) True Weight: 81.90 g")
    print(f"FISH 3 Predicted Weight: {f3_avg_pred:.2f} g")
    print(f"   Absolute Error: {abs(f3_avg_pred - 81.90):.2f} g")
    print("-" * 50)
    print(f"FISH 4 (Smaller Fish) True Weight: 61.25 g")
    print(f"FISH 4 Predicted Weight: {f4_avg_pred:.2f} g")
    print(f"   Absolute Error: {abs(f4_avg_pred - 61.25):.2f} g")
    print("="*50)

if __name__ == '__main__':
    main()
