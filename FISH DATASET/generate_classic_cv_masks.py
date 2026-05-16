"""
Generate Classic CV Masks for all 2538 frames.
Uses the original background subtraction logic (motion_mask) from the existing pipeline.
Applies it to the EXACT same frames that we used for the AI Masks.
Extracts Mask Area, Length, and Width in pixels.
"""

import cv2
import re
import csv
import os
import numpy as np
from pathlib import Path
from build_fish_dataset import build_background, static_exclude_mask, motion_mask, combine_large_components

def keep_largest_component(mask):
    num, labels, stats, _ = cv2.connectedComponentsWithStats((mask > 0).astype("uint8"), connectivity=8)
    if num <= 1:
        return (mask > 0).astype("uint8") * 0
    best = 1
    best_area = int(stats[1, cv2.CC_STAT_AREA])
    for i in range(2, num):
        a = int(stats[i, cv2.CC_STAT_AREA])
        if a > best_area:
            best = i
            best_area = a
    out = ((labels == best).astype("uint8")) * 255
    return out

# -- PATHS --
FISH_DATASET = Path(r"C:\Users\shain\Downloads\FISH DATASET\FISH DATASET")
FRAMES_DIR = Path(r"C:\Users\shain\Downloads\FISH DATASET\frames\fraes")
OUTPUT_MASKS_DIR = FISH_DATASET / "final_classic_masks"
OUTPUT_MASKS_DIR.mkdir(exist_ok=True)
CSV_OUTPUT = FISH_DATASET / "classic_cv_pixel_measurements_all.csv"

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
            if key not in unique_frames or len(fname) < len(unique_frames[key].name):
                unique_frames[key] = img

frames = list(unique_frames.items())
print(f"Found {len(frames)} unique frames to process.")

bg_cache = {}
csv_data = []

# Process frames
for i, ((fish_id, view, frame_num), img_path) in enumerate(frames):
    img = cv2.imread(str(img_path))
    if img is None: continue
    
    # 1. Ensure Background is built for this fish
    if fish_id not in bg_cache:
        bg_cache[fish_id] = {}
        
        # Build Top
        top_video_dir = FISH_DATASET / fish_id / "top view"
        if top_video_dir.exists():
            vids = list(top_video_dir.glob("*.mp4"))
            if vids:
                print(f"Building background for {fish_id} Top View...")
                bg = build_background(str(vids[0]))
                ex = static_exclude_mask(bg, "top")
                bg_cache[fish_id]["top_bg"] = bg
                bg_cache[fish_id]["top_ex"] = ex
                
        # Build Front
        front_video_dir = FISH_DATASET / fish_id / "front view"
        if front_video_dir.exists():
            vids = list(front_video_dir.glob("*.mp4"))
            if vids:
                print(f"Building background for {fish_id} Front View...")
                bg = build_background(str(vids[0]))
                ex = static_exclude_mask(bg, "front")
                bg_cache[fish_id]["front_bg"] = bg
                bg_cache[fish_id]["front_ex"] = ex

    # 2. Extract Classic CV Mask
    clean_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    
    if view == "top" and "top_bg" in bg_cache.get(fish_id, {}):
        diff_thresh = 12
        mask = motion_mask(img, bg_cache[fish_id]["top_bg"], bg_cache[fish_id]["top_ex"], diff_thresh=diff_thresh)
        mask = combine_large_components(mask, min_area_px=300)
        clean_mask = keep_largest_component(mask)
        
    elif view == "front" and "front_bg" in bg_cache.get(fish_id, {}):
        diff_thresh = 10
        mask = motion_mask(img, bg_cache[fish_id]["front_bg"], bg_cache[fish_id]["front_ex"], diff_thresh=diff_thresh)
        mask = combine_large_components(mask, min_area_px=250)
        clean_mask = keep_largest_component(mask)
    
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
    mask_fname = f"{fish_id}_{view}_frame_{frame_num}_classic_mask.png"
    cv2.imwrite(str(OUTPUT_MASKS_DIR / mask_fname), clean_mask)
    
    csv_data.append({
        "FishID": fish_id,
        "View": view,
        "FrameIndex": frame_num,
        "Method": "ClassicCV",
        "MaskArea_px": area,
        "MaskLength_px": length,
        "MaskWidth_px": width,
        "MaskPerimeter_px": perimeter
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
