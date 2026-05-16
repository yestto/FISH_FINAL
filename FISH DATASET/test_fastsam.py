import cv2
from ultralytics import YOLO, FastSAM
import os

import glob

YOLO_MODEL_PATH = r"C:\Users\shain\Downloads\FISH DATASET\FISH-TRAIN\Fish Annotations\yolo_training_runs\fish_yolo12m_full\weights\best.pt"
OUTPUT_DIR = r"C:\Users\shain\Downloads\FISH DATASET\FISH DATASET\production_masks_output"

# Find a front image
images = glob.glob(r"C:\Users\shain\Downloads\FISH DATASET\FISH-TRAIN\Fish Annotations\fish 11 front\**\*.jpg", recursive=True)
IMG_PATH = images[0]
print(f"Using image: {IMG_PATH}")

# Detect Box
yolo_model = YOLO(YOLO_MODEL_PATH)
res = yolo_model(IMG_PATH, imgsz=640, verbose=False)
box = res[0].boxes.xyxy[0].cpu().numpy().tolist()

# Load FastSAM
print("Loading FastSAM-s...")
fast_sam = FastSAM('FastSAM-s.pt')

# Run FastSAM with the YOLO Bounding Box prompt
results = fast_sam(IMG_PATH, bboxes=[box], imgsz=1024, verbose=False)

# Get the mask
frame = cv2.imread(IMG_PATH)
if results[0].masks is not None:
    mask = results[0].masks.data[0].cpu().numpy()
    mask = cv2.resize(mask, (frame.shape[1], frame.shape[0]))
    
    # Draw Green Mask
    annotated = frame.copy()
    annotated[mask > 0] = [0, 255, 0]
    annotated = cv2.addWeighted(annotated, 0.5, frame, 0.5, 0)
    
    # Draw Box
    cv2.rectangle(annotated, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), (0, 0, 255), 2)
    
    out_path = os.path.join(OUTPUT_DIR, "test_fastsam_front.jpg")
    cv2.imwrite(out_path, annotated)
    print(f"Saved FastSAM output to {out_path}")
else:
    print("FastSAM failed to find a mask inside the box.")
