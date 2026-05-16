"""
Compare Classic CV, FastSAM, and full SAM (vit_b) for fish body extraction.
Testing on specifically requested frames with improved dynamic morphology.
"""

import cv2
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
OUTPUT_DIR = FISH_DATASET / "mask_comparison_results"
OUTPUT_DIR.mkdir(exist_ok=True)

YOLO_MODEL_PATH = Path(r"C:\Users\shain\Downloads\FISH DATASET\FISH-TRAIN\Fish Annotations\yolo_training_runs\fish_yolo12m_full\weights\best.pt")
FASTSAM_PATH = FISH_DATASET / "FastSAM-s.pt"
SAM_CHECKPOINT = FISH_DATASET / "sam_vit_b_01ec64.pth"

TARGET_FRAMES = [
    Path(r"C:\Users\shain\Downloads\FISH DATASET\frames\fraes\fish01_frames\fish01-topview-frame-166.jpg"),
    Path(r"C:\Users\shain\Downloads\FISH DATASET\frames\fraes\fish01_frames\fish01-frontview-frame-3439.jpg")
]

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
    x1, y1, x2, y2 = bbox
    box_w = x2 - x1
    box_h = y2 - y1
    
    # Dynamic kernel size based on bounding box (around 15% of the shortest dimension)
    k_size = max(3, int(min(box_w, box_h) * 0.15))
    if k_size % 2 == 0: k_size += 1 # must be odd
    
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
    clean_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Keep only the largest contour to remove floating artifacts
    contours, _ = cv2.findContours(clean_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        final_mask = np.zeros_like(clean_mask)
        cv2.drawContours(final_mask, [c], -1, 255, thickness=cv2.FILLED)
        return final_mask
    return clean_mask

def get_classic_cv_mask(img, bbox):
    """Simple HSV thresholding restricted to bounding box as a fallback Classic CV"""
    x1, y1, x2, y2 = map(int, bbox)
    mask = np.zeros(img.shape[:2], np.uint8)
    
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Generic threshold that often works for dark fish
    _, thresh = cv2.threshold(hsv[:,:,2], 100, 255, cv2.THRESH_BINARY_INV)
    
    # Apply bounding box
    mask[y1:y2, x1:x2] = thresh[y1:y2, x1:x2]
    return mask

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

def overlay_mask(img, mask, color):
    overlay = img.copy()
    if mask.shape[:2] != overlay.shape[:2]:
        mask = cv2.resize(mask, (overlay.shape[1], overlay.shape[0]))
    overlay[mask > 127] = overlay[mask > 127] * 0.5 + np.array(color) * 0.5
    return overlay

print(f"\nProcessing {len(TARGET_FRAMES)} specific target frames...")

for i, img_path in enumerate(TARGET_FRAMES):
    if not img_path.exists(): continue
    img = cv2.imread(str(img_path))
    if img is None: continue
    
    results = yolo_model(str(img_path), verbose=False, imgsz=640)
    if len(results[0].boxes) == 0: continue
    
    box = results[0].boxes.xyxy[0].cpu().numpy()
    
    classic_mask = get_classic_cv_mask(img, box)
    fastsam_mask = get_fastsam_mask(img_path, box)
    full_sam_mask = get_full_sam_mask(img, box)
    
    classic_clean = remove_fins_and_tails(classic_mask, box)
    fastsam_clean = remove_fins_and_tails(fastsam_mask, box)
    full_sam_clean = remove_fins_and_tails(full_sam_mask, box)
    
    vis_classic = overlay_mask(img, classic_clean, [255, 0, 0])
    vis_fastsam = overlay_mask(img, fastsam_clean, [0, 255, 0])
    vis_full_sam = overlay_mask(img, full_sam_clean, [0, 0, 255])
    
    cv2.putText(vis_classic, "Classic CV (HSV)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis_fastsam, "FastSAM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(vis_full_sam, "Full SAM", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    
    x1, y1, x2, y2 = map(int, box)
    for vis in [vis_classic, vis_fastsam, vis_full_sam]:
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 255), 2)
    
    combined = np.hstack((vis_classic, vis_fastsam, vis_full_sam))
    
    artifact_path = Path(r"C:\Users\shain\.gemini\antigravity\brain\87351874-ce08-4f2d-8799-83af16d672a0") / f"FIXED_{img_path.name}"
    cv2.imwrite(str(artifact_path), combined)
    print(f"Saved comparison to: {artifact_path}")
