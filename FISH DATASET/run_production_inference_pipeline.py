import os
import glob
import cv2
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from ultralytics import YOLO, FastSAM
from segment_anything import sam_model_registry, SamPredictor
import re

# ==========================================
# Configuration
# ==========================================
ROOT_DIR = r"C:\Users\shain\Downloads\FISH DATASET\FISH DATASET"
FRAMES_DIR = r"C:\Users\shain\Downloads\FISH DATASET\FISH-TRAIN\Fish Annotations"
YOLO_MODEL_PATH = r"C:\Users\shain\Downloads\FISH DATASET\FISH-TRAIN\Fish Annotations\yolo_training_runs\fish_yolo12m_full\weights\best.pt"
SAM_MODEL_PATH = os.path.join(ROOT_DIR, "sam_vit_b_01ec64.pth")
FASTSAM_MODEL_PATH = os.path.join(ROOT_DIR, "FastSAM-s.pt")
OUTPUT_CSV = os.path.join(ROOT_DIR, "fish_frames_production.csv")
MASKS_DIR = os.path.join(ROOT_DIR, "production_masks_output")

# Final Validated True Lengths for Calibration
TARGET_LENGTHS = {
    "fish01": 12.5, "fish2":  8.5,  "fish3":  13.0, "fish4":  11.5,
    "fish5":  10.5, "fish6":  7.0,  "fish7":  7.5,  "fish8":  7.5,
    "fish9":  6.5,  "fish10": 7.5,  "fish11": 6.0,  "fish12": 6.0,
    "fish13": 6.0,  "fish14": 5.0,  "fish15": 6.0,
}

# Known True Weights
TRUE_WEIGHTS = {
    "fish01": 67.49, "fish2": 16.00, "fish3": 81.90, "fish4": 61.25,
    "fish5": 27.36, "fish6": 13.78, "fish7": 18.32, "fish8": 12.70,
    "fish9": 14.00, "fish10": 13.50, "fish11": 9.30, "fish12": 8.90,
    "fish13": 13.88, "fish14": 12.32, "fish15": 9.61
}

def get_fish_id_and_view(folder_name):
    folder_name = folder_name.lower()
    view = "top" if "top" in folder_name else "front"
    nums = re.findall(r'\d+', folder_name)
    if nums:
        num = int(nums[0])
        fish_id = f"fish0{num}" if num < 10 and num == 1 else f"fish{num}"
        return fish_id, view
    return None, None

def enhance(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

def remove_fins(mask, bbox):
    """Dynamic elliptical morphological opening to remove fins/tails without eroding body."""
    x1, y1, x2, y2 = bbox
    k_size = max(3, int(min(x2-x1, y2-y1) * 0.15))
    if k_size % 2 == 0: k_size += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Keep only the largest component to drop disconnected fins
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        final = np.zeros_like(clean)
        cv2.drawContours(final, [c], -1, 255, thickness=cv2.FILLED)
        return final
    return clean

def extract_features(mask):
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    c = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(c)
    perimeter = cv2.arcLength(c, True)
    rect = cv2.minAreaRect(c)
    (x, y), (w, h), angle = rect
    length = max(w, h)
    width = min(w, h)
    return {"length_px": length, "width_px": width, "area_px": area, "perimeter_px": perimeter, "mask_px": np.sum(mask > 0)}

def detect_fish(img_path, yolo_model):
    """Run YOLO detection and return frame, best box, and confidence."""
    frame = cv2.imread(img_path)
    if frame is None: return None, None, None
    res = yolo_model(frame, verbose=False, imgsz=640)
    boxes = res[0].boxes.xyxy.cpu().numpy()
    confs = res[0].boxes.conf.cpu().numpy()
    if len(boxes) == 0: return frame, None, None
    best_idx = np.argmax(confs)
    return frame, boxes[best_idx], confs[best_idx]

def get_sam_mask_top(frame, box, sam_predictor):
    """Full SAM for TOP view — clean water, high contrast. Box-only prompt."""
    sam_predictor.set_image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    x1, y1, x2, y2 = [int(v) for v in box]
    masks, _, _ = sam_predictor.predict(box=np.array([x1, y1, x2, y2]), multimask_output=False)
    return (masks[0] * 255).astype(np.uint8)

def get_sam_mask_front(frame, box, sam_predictor):
    """Enhanced Full SAM for FRONT view — murky water, low contrast.
    Uses CLAHE enhancement + box+point prompt + best-of-3 masks."""
    x1, y1, x2, y2 = [int(v) for v in box]
    
    # CLAHE Enhancement to boost fish contrast in murky water
    enhanced = enhance(frame)
    sam_predictor.set_image(cv2.cvtColor(enhanced, cv2.COLOR_BGR2RGB))
    
    # Box + center point prompt for strong guidance
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    masks, scores, _ = sam_predictor.predict(
        point_coords=np.array([[cx, cy]]),
        point_labels=np.array([1]),
        box=np.array([x1, y1, x2, y2]),
        multimask_output=True
    )
    
    best_idx = np.argmax(scores)
    return (masks[best_idx] * 255).astype(np.uint8)

def clip_mask_to_box(mask, box):
    """STRICTLY clip mask to only keep pixels INSIDE the YOLO bounding box."""
    x1, y1, x2, y2 = [int(v) for v in box]
    clipped = np.zeros_like(mask)
    clipped[y1:y2, x1:x2] = mask[y1:y2, x1:x2]
    return clipped

def process_frame(img_path, view, yolo_model, sam_predictor):
    """Process a single frame with view-specific SAM strategy."""
    frame, box, conf = detect_fish(img_path, yolo_model)
    if frame is None or box is None: return None, None
    
    # View-specific masking
    if view == "top":
        raw_mask = get_sam_mask_top(frame, box, sam_predictor)
    else:
        raw_mask = get_sam_mask_front(frame, box, sam_predictor)
    
    # STRICT: Only keep mask pixels inside the YOLO bounding box
    raw_mask = clip_mask_to_box(raw_mask, box)
    
    clean_mask = remove_fins(raw_mask, box)
    
    features = extract_features(clean_mask)
    if features:
        features["Score"] = conf
        x1, y1, x2, y2 = [int(v) for v in box]
        annotated = frame.copy()
        annotated[clean_mask > 0] = [0, 255, 0]
        annotated = cv2.addWeighted(annotated, 0.4, frame, 0.6, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255), 2)
        return features, annotated
    return None, None

def run_pipeline():
    print("Finding precisely synced pairs...")
    fish_frames = {}
    
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        for path in glob.glob(os.path.join(FRAMES_DIR, "**", ext), recursive=True):
            if "yolo_dataset" in path or "runs" in path:
                continue
                
            parent_folder = os.path.basename(os.path.dirname(os.path.dirname(path)))
            if parent_folder == "Fish Annotations":
                parent_folder = os.path.basename(os.path.dirname(path))
                
            fish_id, view = get_fish_id_and_view(parent_folder)
            if not fish_id: continue
            
            match = re.search(r'frame_(\d+)', os.path.basename(path))
            if not match: continue
            frame_num = int(match.group(1))
            
            if fish_id not in fish_frames: fish_frames[fish_id] = {}
            if frame_num not in fish_frames[fish_id]: fish_frames[fish_id][frame_num] = {}
            fish_frames[fish_id][frame_num][view] = path

    paired_paths = []
    for fish_id, frames in fish_frames.items():
        for frame_num, views in frames.items():
            if "top" in views and "front" in views:
                paired_paths.append((fish_id, frame_num, views["top"], views["front"]))
                
    print(f"Found {len(paired_paths)} perfect Top/Front synced pairs!")
    if not paired_paths:
        return

    print("Loading Models...")
    yolo_model = YOLO(YOLO_MODEL_PATH)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry['vit_b'](checkpoint=SAM_MODEL_PATH)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)

    os.makedirs(MASKS_DIR, exist_ok=True)
    raw_paired_data = []

    for fish_id, frame_num, top_path, front_path in tqdm(paired_paths, desc="Processing Paired Frames"):
        # RESUME: skip if both mask images already exist
        top_mask_path = os.path.join(MASKS_DIR, f"{fish_id}_frame_{frame_num}_top.jpg")
        front_mask_path = os.path.join(MASKS_DIR, f"{fish_id}_frame_{frame_num}_front.jpg")
        if os.path.exists(top_mask_path) and os.path.exists(front_mask_path):
            # Still need pixel data — re-extract features from the raw frames quickly
            top_features, top_img = process_frame(top_path, "top", yolo_model, sam_predictor)
            front_features, front_img = process_frame(front_path, "front", yolo_model, sam_predictor)
            if top_features and front_features:
                raw_paired_data.append({
                    "FishID": fish_id, "FrameIndex": frame_num,
                    "Top_Image": top_path, "Front_Image": front_path,
                    "Top_Length_px": top_features["length_px"], "Top_Width_px": top_features["width_px"],
                    "Top_Area_px": top_features["area_px"], "Top_Perim_px": top_features["perimeter_px"],
                    "Top_MaskPx": top_features["mask_px"], "Top_Score": top_features["Score"],
                    "Front_Length_px": front_features["length_px"],
                    "Front_Height_px": front_features["width_px"],
                    "Front_MaskPx": front_features["mask_px"], "Front_Score": front_features["Score"]
                })
            continue
            
        top_features, top_img = process_frame(top_path, "top", yolo_model, sam_predictor)
        front_features, front_img = process_frame(front_path, "front", yolo_model, sam_predictor)
        
        if top_features and front_features:
            raw_paired_data.append({
                "FishID": fish_id,
                "FrameIndex": frame_num,
                "Top_Image": top_path,
                "Front_Image": front_path,
                "Top_Length_px": top_features["length_px"],
                "Top_Width_px": top_features["width_px"],
                "Top_Area_px": top_features["area_px"],
                "Top_Perim_px": top_features["perimeter_px"],
                "Top_MaskPx": top_features["mask_px"],
                "Top_Score": top_features["Score"],
                "Front_Length_px": front_features["length_px"],
                "Front_Height_px": front_features["width_px"],
                "Front_MaskPx": front_features["mask_px"],
                "Front_Score": front_features["Score"]
            })
            
            # Save visual masks
            cv2.imwrite(os.path.join(MASKS_DIR, f"{fish_id}_frame_{frame_num}_top.jpg"), top_img)
            cv2.imwrite(os.path.join(MASKS_DIR, f"{fish_id}_frame_{frame_num}_front.jpg"), front_img)

    raw_df = pd.DataFrame(raw_paired_data)
    
    # --- CALIBRATION ---
    print("\nCalibrating Pixels to Centimeters using perfectly synced pairs...")
    final_rows = []
    
    for fish_id, group in raw_df.groupby("FishID"):
        if fish_id not in TARGET_LENGTHS: continue
            
        # SEPARATE calibrations for each camera!
        # Top camera: calibrate from top-view fish length
        top_px_per_cm = group["Top_Length_px"].median() / TARGET_LENGTHS[fish_id]
        # Front camera: calibrate from front-view fish length (major axis = same physical length)
        front_px_per_cm = group["Front_Length_px"].median() / TARGET_LENGTHS[fish_id]
        true_weight = TRUE_WEIGHTS.get(fish_id, np.nan)
        
        for _, row in group.iterrows():
            length_cm = row["Top_Length_px"] / top_px_per_cm
            width_cm = row["Top_Width_px"] / top_px_per_cm
            height_cm = row["Front_Height_px"] / front_px_per_cm  # FIXED: uses front camera's own scale
            area_cm2 = row["Top_Area_px"] / (top_px_per_cm ** 2)
            perim_cm = row["Top_Perim_px"] / top_px_per_cm
            
            # Derived Biometrics from synced 3D data!
            volume = np.pi * length_cm * (width_cm / 2.0) * (height_cm / 2.0)
            surface_area = 2 * area_cm2 + perim_cm * height_cm
            aspect_ratio = length_cm / width_cm if width_cm > 0 else 0
            rect = area_cm2 / (length_cm * width_cm) if (length_cm * width_cm) > 0 else 0
            eq_diam = np.sqrt(4 * area_cm2 / np.pi)
            
            final_rows.append({
                "FishID": fish_id,
                "Weight (g)": true_weight,
                "FrameIndex": row["FrameIndex"],
                "Top_Image_File": row["Top_Image"],
                "Front_Image_File": row["Front_Image"],
                "Timestamp (s)": 0, 
                "FPS_Top": 20,
                "FPS_Front": 20,
                "Length (cm)": length_cm,
                "Width (cm)": width_cm,
                "Height (cm)": height_cm,
                "Area (cm²)": area_cm2,
                "Perimeter (cm)": perim_cm,
                "TopMaskPixels": row["Top_MaskPx"],
                "FrontMaskPixels": row["Front_MaskPx"],
                "BlurTop": 500,
                "BlurFront": 500,
                "Score": (row["Top_Score"] + row["Front_Score"]) / 2,
                "Volume (cm³)": volume,
                "Surface Area (cm²)": surface_area,
                "Aspect Ratio": aspect_ratio,
                "Elongation": 1.0 / aspect_ratio if aspect_ratio > 0 else 0,
                "Compactness": (perim_cm ** 2) / area_cm2 if area_cm2 > 0 else 0,
                "Condition Factor (K)": (true_weight / (length_cm ** 3)) * 100 if length_cm > 0 else 0,
                "Rectangularity": rect,
                "Equivalent Diameter (cm)": eq_diam
            })
            
    final_df = pd.DataFrame(final_rows)
    final_df.to_csv(OUTPUT_CSV, index=False)
    print(f"\nProduction pipeline finished! Dataset saved to: {OUTPUT_CSV}")
    print(f"Total synced rows: {len(final_df)}")

if __name__ == '__main__':
    run_pipeline()
