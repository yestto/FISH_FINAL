"""
Run YOLO11m-seg (segmentation) model on unseen frames
and save mask-annotated images for comparison with detection models.
"""

import cv2
from pathlib import Path
from ultralytics import YOLO

FISH_TRAIN = Path(r"C:\Users\shain\Downloads\FISH DATASET\FISH-TRAIN\Fish Annotations")
FRAMES_DIR = Path(r"C:\Users\shain\Downloads\FISH DATASET\frames\fraes")
OUTPUT_ROOT = Path(r"C:\Users\shain\Downloads\FISH DATASET\FISH DATASET\model_comparison_results")

SEG_MODEL = FISH_TRAIN / "yolo_training_runs" / "fish_yolo11m_seg_1280" / "weights" / "best.pt"

# Collect frames (skip IP duplicates)
frames = []
for fish_dir in sorted(FRAMES_DIR.iterdir()):
    if not fish_dir.is_dir():
        continue
    for img in sorted(fish_dir.glob("*.jpg")):
        if "192.168" not in img.name:
            frames.append(img)

print(f"Found {len(frames)} frames")

# Run segmentation model
print(f"\nRunning: YOLO11m-seg")
print(f"Model: {SEG_MODEL}")

out_dir = OUTPUT_ROOT / "bbox_YOLO11m_seg"
out_dir.mkdir(parents=True, exist_ok=True)

model = YOLO(str(SEG_MODEL))

for i, img_path in enumerate(frames):
    results = model(str(img_path), verbose=False, imgsz=640)
    annotated = results[0].plot()  # draws mask + bbox + confidence
    
    out_path = out_dir / f"{img_path.parent.name}_{img_path.name}"
    cv2.imwrite(str(out_path), annotated)
    
    if (i + 1) % 50 == 0:
        print(f"  {i+1}/{len(frames)}...")

print(f"\nDone! Saved {len(frames)} segmentation images to: {out_dir}")
