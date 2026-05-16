"""
Generate Hybrid AI Masks for all 2538 frames.
- Top View frames -> Full SAM
- Front View frames -> FastSAM
Uses YOLO12m bounding boxes and applies dynamic fin/tail removal.
Extracts mask Area, Length, and Width in pixels to a CSV.
"""

import cv2
import re
import csv
import torch
import numpy as np
from pathlib import Path
from ultralytics import YOLO, FastSAM
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("segment_anything is not installed. Please install it.")
    exit()

# -- PATHS --
FISH_DATASET = Path(r"C:\Users\shain\Downloads\FISH DATASET\FISH DATASET")
FRAMES_DIR = Path(r"C:\Users\shain\Downloads\FISH DATASET\frames\fraes")
OUTPUT_MASKS_DIR = FISH_DATASET / "final_ai_masks"
OUTPUT_MASKS_DIR.mkdir(exist_ok=True)
CSV_OUTPUT = FISH_DATASET / "ai_pixel_measurements_hybrid.csv"

YOLO_MODEL_PATH = Path(r"C:\Users\shain\Downloads\FISH DATASET\FISH-TRAIN\Fish Annotations\yolo_training_runs\fish_yolo12m_full\weights\best.pt")
FASTSAM_PATH = FISH_DATASET / "FastSAM-s.pt"
SAM_CHECKPOINT = FISH_DATASET / "sam_vit_b_01ec64.pth"

# -- INIT MODELS --
print("Loading YOLO12m...")
yolo_model = YOLO(str(YOLO_MODEL_PATH))

print("Loading FastSAM...")
fastsam_model = FastSAM(str(FASTSAM_PATH))

print("Loading Full SAM (vit_b)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
sam.to(device=device)
sam_predictor = SamPredictor(sam)

def remove_fins_and_tails(mask, bbox):
    """Dynamic morphological opening to remove fins/tails."""
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    
    k_size = max(3, int(min(box_w, box_h) * 0.15))
    if k_size % 2 == 0: k_size += 1
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Keep largest contour
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        final_mask = np.zeros_like(clean_mask)
        cv2.drawContours(final_mask, [c], -1, 255, thickness=cv2.FILLED)
        return final_mask, c
    return clean_mask, None

def get_fastsam_mask(img_path, bbox):
    bbox_int = [int(x) for x in bbox]
    results = fastsam_model(str(img_path), bboxes=[bbox_int], verbose=False)
    if results and len(results) > 0 and results[0].masks is not None:
        mask = results[0].masks.data[0].cpu().numpy()
        mask = cv2.resize(mask, (results[0].orig_shape[1], results[0].orig_shape[0]))
        return (mask * 255).astype(np.uint8)
    return np.zeros((10, 10), np.uint8)

def get_full_sam_mask(img, bbox):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    sam_predictor.set_image(img_rgb)
    masks, _, _ = sam_predictor.predict(box=np.array(bbox), multimask_output=False)
    return (masks[0] * 255).astype(np.uint8)

print("Collecting frames...")
unique_frames = {}
for fish_dir in sorted(FRAMES_DIR.iterdir()):
    if not fish_dir.is_dir(): continue
    fish_id = fish_dir.name.replace("_frames", "")
    for img in sorted(fish_dir.glob("*.jpg")):
        fname = img.name
        view = "front" if "frontview" in fname else ("top" if "topview" in fname else "unknown")
        match = re.search(r"-frame-(\d+)\.jpg", fname)
        if match:
            frame_num = int(match.group(1))
            key = (fish_id, view, frame_num)
            if key not in unique_frames or len(fname) < len(unique_frames[key].name):
                unique_frames[key] = img

frames = list(unique_frames.items())
print(f"Found {len(frames)} unique frames to process.")

csv_data = []

# Process in batches to save memory
for i, ((fish_id, view, frame_num), img_path) in enumerate(frames):
    img = cv2.imread(str(img_path))
    if img is None: continue
    
    # 1. YOLO Bounding Box
    results = yolo_model(str(img_path), verbose=False, imgsz=640)
    if len(results[0].boxes) == 0:
        continue
    
    box = results[0].boxes.xyxy[0].cpu().numpy()
    
    # 2. Hybrid Masking
    if view == "top":
        raw_mask = get_full_sam_mask(img, box)
        method = "Full_SAM"
    else:
        raw_mask = get_fastsam_mask(img_path, box)
        method = "FastSAM"
        if raw_mask.shape[:2] != img.shape[:2]:
            raw_mask = cv2.resize(raw_mask, (img.shape[1], img.shape[0]))
    
    # 3. Clean Mask (remove fins)
    clean_mask, contour = remove_fins_and_tails(raw_mask, box)
    
    # 4. Extract Measurements
    area = 0
    length = 0
    width = 0
    perimeter = 0
    
    if contour is not None and len(contour) > 4:
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        # Rotated rect for length/width
        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (rect_w, rect_h), angle = rect
        length = max(rect_w, rect_h)
        width = min(rect_w, rect_h)
    
    # Save the binary mask image
    mask_fname = f"{fish_id}_{view}_frame_{frame_num}_mask.png"
    cv2.imwrite(str(OUTPUT_MASKS_DIR / mask_fname), clean_mask)
    
    csv_data.append({
        "FishID": fish_id,
        "View": view,
        "FrameIndex": frame_num,
        "Method": method,
        "MaskArea_px": area,
        "MaskLength_px": length,
        "MaskWidth_px": width,
        "MaskPerimeter_px": perimeter,
        "Bbox_w_px": box[2] - box[0],
        "Bbox_h_px": box[3] - box[1]
    })
    
    if (i + 1) % 50 == 0 or (i + 1) == len(frames):
        print(f"  Processed {i+1}/{len(frames)}...")

# Write CSV
with open(CSV_OUTPUT, "w", newline="") as f:
    if len(csv_data) > 0:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)

print(f"\n✅ Done! Saved {len(csv_data)} masks to: {OUTPUT_MASKS_DIR}")
print(f"✅ Saved pixel measurements to: {CSV_OUTPUT}")
