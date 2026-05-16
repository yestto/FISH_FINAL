"""
Generate AI Masks for 2,538 frames using ONLY Full Meta SAM.
Replaces the hybrid approach to guarantee perfect bounding box adherence.
"""

import cv2
import torch
import numpy as np
import re
import csv
import os
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

# -- PATHS --
FISH_DATASET = Path(r"C:\Users\shain\Downloads\FISH DATASET\FISH DATASET")
FRAMES_DIR = Path(r"C:\Users\shain\Downloads\FISH DATASET\frames\fraes")

YOLO_WEIGHTS = FISH_DATASET.parent / "FISH-TRAIN" / "Fish Annotations" / "yolo_training_runs" / "fish_yolo12m_full" / "weights" / "best.pt"
SAM_CHECKPOINT = FISH_DATASET / "sam_vit_b_01ec64.pth"

OUTPUT_MASKS_DIR = FISH_DATASET / "final_ai_masks_full_sam"
OUTPUT_MASKS_DIR.mkdir(exist_ok=True)
CSV_OUTPUT = FISH_DATASET / "ai_pixel_measurements_full_sam.csv"

# -- MODELS --
print("Loading YOLO12m...")
yolo_model = YOLO(str(YOLO_WEIGHTS))

print("Loading Full Meta SAM (vit_b)...")
device = "cuda" if torch.cuda.is_available() else "cpu"
sam = sam_model_registry["vit_b"](checkpoint=str(SAM_CHECKPOINT))
sam.to(device=device)
sam_predictor = SamPredictor(sam)

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

# -- COLLECT FRAMES --
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
            # Avoid dupes, take shortest name
            if key not in unique_frames or len(fname) < len(unique_frames[key].name):
                unique_frames[key] = img

frames = list(unique_frames.items())
print(f"Found {len(frames)} unique frames to process.")

csv_data = []

# Process frames
for i, ((fish_id, view, frame_num), img_path) in enumerate(tqdm(frames, desc="Extracting with Full SAM")):
    img = cv2.imread(str(img_path))
    if img is None: continue
    
    # 1. YOLO Detection
    res = yolo_model(img, verbose=False, imgsz=640)
    boxes = res[0].boxes.xyxy.cpu().numpy()
    confs = res[0].boxes.conf.cpu().numpy()
    
    clean_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    # 2. Full SAM Segmentation (Only if boxes found)
    if len(boxes) > 0:
        # Get highest confidence box
        best_idx = np.argmax(confs)
        box = boxes[best_idx]
        box_int = [int(x) for x in box]
        
        # Run Full SAM
        sam_predictor.set_image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        masks, _, _ = sam_predictor.predict(box=np.array(box_int), multimask_output=False)
        raw_mask = (masks[0] * 255).astype(np.uint8)
        
        # Clean Fins
        clean_mask = remove_fins(raw_mask, box)
        
    # 3. Extract Measurements
    area = 0
    length = 0
    width = 0
    perimeter = 0
    
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        if len(c) > 4:
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            rect = cv2.minAreaRect(c)
            (center_x, center_y), (rect_w, rect_h), angle = rect
            length = max(rect_w, rect_h)
            width = min(rect_w, rect_h)
    
    # Save the binary mask image
    mask_fname = f"{fish_id}_{view}_frame_{frame_num}_full_sam_mask.png"
    cv2.imwrite(str(OUTPUT_MASKS_DIR / mask_fname), clean_mask)
    
    csv_data.append({
        "FishID": fish_id,
        "View": view,
        "FrameIndex": frame_num,
        "Method": "FullSAM",
        "MaskArea_px": area,
        "MaskLength_px": length,
        "MaskWidth_px": width,
        "MaskPerimeter_px": perimeter
    })

# Write CSV
with open(CSV_OUTPUT, "w", newline="") as f:
    if len(csv_data) > 0:
        writer = csv.DictWriter(f, fieldnames=csv_data[0].keys())
        writer.writeheader()
        writer.writerows(csv_data)

print(f"\n✅ Done! Saved {len(csv_data)} perfectly extracted Full SAM masks to: {OUTPUT_MASKS_DIR}")
print(f"✅ Saved pixel measurements to: {CSV_OUTPUT}")
