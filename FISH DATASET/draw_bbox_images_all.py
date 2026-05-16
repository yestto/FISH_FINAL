"""
Run YOLO12m inference on unseen frames for ALL fishes.
"""

import cv2
import re
from pathlib import Path
from ultralytics import YOLO

MODEL_PATH = Path(r"C:\Users\shain\Downloads\FISH DATASET\FISH-TRAIN\Fish Annotations\yolo_training_runs\fish_yolo12m_full\weights\best.pt")
FRAMES_DIR = Path(r"C:\Users\shain\Downloads\FISH DATASET\frames\fraes")
OUTPUT_DIR = Path(r"C:\Users\shain\Downloads\FISH DATASET\FISH DATASET\model_comparison_results\bbox_images_all")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("Loading YOLO12m model...")
model = YOLO(str(MODEL_PATH))

# Collect unique frames to avoid duplicates
# Key: (fish_id, view, frame_num) -> image_path
unique_frames = {}

for fish_dir in sorted(FRAMES_DIR.iterdir()):
    if not fish_dir.is_dir():
        continue
        
    fish_id = fish_dir.name.replace("_frames", "")
    
    for img in sorted(fish_dir.glob("*.jpg")):
        fname = img.name
        
        view = "unknown"
        if "frontview" in fname:
            view = "front"
        elif "topview" in fname:
            view = "top"
            
        # Extract frame number using regex
        match = re.search(r"-frame-(\d+)\.jpg", fname)
        if match:
            frame_num = int(match.group(1))
            key = (fish_id, view, frame_num)
            
            # Prefer shorter filenames to avoid IP address duplicates if they exist
            if key not in unique_frames or len(fname) < len(unique_frames[key].name):
                unique_frames[key] = img

frames = list(unique_frames.values())
print(f"Found {len(frames)} unique frames across all fish")

saved = 0
for i, img_path in enumerate(frames):
    results = model(str(img_path), verbose=False, imgsz=640)
    annotated = results[0].plot()
    
    out_path = OUTPUT_DIR / f"{img_path.parent.name}_{img_path.name}"
    cv2.imwrite(str(out_path), annotated)
    saved += 1
    
    if (i + 1) % 100 == 0 or (i + 1) == len(frames):
        print(f"  Processed {i+1}/{len(frames)}...")

print(f"\nDone! Saved {saved} bbox images for all fish to: {OUTPUT_DIR}")
