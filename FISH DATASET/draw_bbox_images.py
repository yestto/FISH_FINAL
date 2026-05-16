"""
Run ALL 3 detection models on unseen frames and save bbox images
in separate folders for visual comparison.
"""

import cv2
from pathlib import Path
from ultralytics import YOLO

FISH_TRAIN = Path(r"C:\Users\shain\Downloads\FISH DATASET\FISH-TRAIN\Fish Annotations")
FRAMES_DIR = Path(r"C:\Users\shain\Downloads\FISH DATASET\frames\fraes")
OUTPUT_ROOT = Path(r"C:\Users\shain\Downloads\FISH DATASET\FISH DATASET\model_comparison_results")

MODELS = {
    "YOLO11m": FISH_TRAIN / "yolo_training_runs" / "fish_yolo11m_full" / "weights" / "best.pt",
    "YOLO12m": FISH_TRAIN / "yolo_training_runs" / "fish_yolo12m_full" / "weights" / "best.pt",
    "YOLOv8m": FISH_TRAIN / "yolo_training_runs" / "fish_yolov8m_1280" / "weights" / "best.pt",
}

# Collect frames (skip IP duplicates)
frames = []
for fish_dir in sorted(FRAMES_DIR.iterdir()):
    if not fish_dir.is_dir():
        continue
    for img in sorted(fish_dir.glob("*.jpg")):
        if "192.168" not in img.name:
            frames.append(img)

print(f"Found {len(frames)} frames\n")

for model_name, model_path in MODELS.items():
    print(f"{'='*50}")
    print(f"Running: {model_name}")
    print(f"{'='*50}")
    
    out_dir = OUTPUT_ROOT / f"bbox_{model_name}"
    out_dir.mkdir(parents=True, exist_ok=True)
    
    model = YOLO(str(model_path))
    
    for i, img_path in enumerate(frames):
        results = model(str(img_path), verbose=False, imgsz=640)
        annotated = results[0].plot()
        
        out_path = out_dir / f"{img_path.parent.name}_{img_path.name}"
        cv2.imwrite(str(out_path), annotated)
        
        if (i + 1) % 50 == 0:
            print(f"  {i+1}/{len(frames)}...")
    
    print(f"  Saved {len(frames)} images to: {out_dir}\n")

print("All 3 models done! Compare the folders:")
for model_name in MODELS:
    print(f"  bbox_{model_name}/")
