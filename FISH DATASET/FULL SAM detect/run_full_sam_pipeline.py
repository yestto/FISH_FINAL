import cv2
import torch
import numpy as np
import os
import glob
from pathlib import Path
from ultralytics import YOLO
from segment_anything import sam_model_registry, SamPredictor

def enhance(img):
    """Enhance murky underwater images for better detection and segmentation."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4,4))
    l = clahe.apply(l)
    return cv2.cvtColor(cv2.merge([l,a,b]), cv2.COLOR_LAB2BGR)

def remove_fins_gentle(mask, bbox):
    """Applies morphological opening to gently remove fins while preserving the fish body."""
    x1, y1, x2, y2 = [int(v) for v in bbox]
    min_dim = min(x2-x1, y2-y1)
    
    # Adaptive kernel size based on fish size
    pct = 0.04 if min_dim < 50 else (0.06 if min_dim < 100 else 0.10)
    k = max(3, int(min_dim * pct))
    if k % 2 == 0: k += 1
        
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    clean = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    
    # Keep only the largest component to remove disconnected floating noise
    contours, _ = cv2.findContours(clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        c = max(contours, key=cv2.contourArea)
        final = np.zeros_like(clean)
        cv2.drawContours(final, [c], -1, 255, cv2.FILLED)
        return final
    return clean

def apply_mask_overlay(image, mask, color=(0, 255, 0), alpha=0.5):
    """Draws a semi-transparent mask over the original image."""
    overlay = image.copy()
    overlay[mask > 0] = color
    return cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))
    input_dir = os.path.join(base_dir, "input_frames")
    output_dir = os.path.join(base_dir, "output_frames")
    
    # Create directories if they don't exist
    os.makedirs(input_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all images in input folder
    image_files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        
    if not image_files:
        print(f"No images found in '{input_dir}'. Please add some frames to process.")
        return

    # Load Models
    print("Loading YOLOv8 Detection Model...")
    dataset_root = os.path.dirname(base_dir) # parent directory 'FISH DATASET'
    yolo_path = os.path.join(dataset_root, r'..\FISH-TRAIN\Fish Annotations\yolo_training_runs\fish_yolo12m_full\weights\best.pt')
    yolo_model = YOLO(yolo_path)
    
    print("Loading Full SAM (ViT-B) Segmentation Model...")
    sam_checkpoint = os.path.join(dataset_root, 'sam_vit_b_01ec64.pth')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    sam = sam_model_registry['vit_b'](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam_predictor = SamPredictor(sam)
    
    print(f"\nProcessing {len(image_files)} frames...")
    
    for img_path in image_files:
        filename = os.path.basename(img_path)
        print(f"  -> Processing {filename}...")
        
        # Read image
        frame = cv2.imread(img_path)
        if frame is None:
            print(f"     [Error] Could not read {filename}")
            continue
            
        # Detect with YOLO
        res = yolo_model(frame, verbose=False, imgsz=640)
        boxes = res[0].boxes.xyxy.cpu().numpy()
        confs = res[0].boxes.conf.cpu().numpy()
        
        if len(boxes) == 0:
            print(f"     [Warning] No fish detected in {filename}")
            cv2.imwrite(os.path.join(output_dir, filename), frame)
            continue
            
        # Optional: enhance image before passing to SAM for better edge detection
        # We pass the enhanced image to SAM, but draw the final results on the original frame
        enhanced_frame = enhance(frame)
        sam_predictor.set_image(cv2.cvtColor(enhanced_frame, cv2.COLOR_BGR2RGB))
        
        annotated_frame = frame.copy()
        
        # Process each detected fish
        for i, box in enumerate(boxes):
            conf = confs[i]
            x1, y1, x2, y2 = [int(v) for v in box]
            
            # Predict mask with SAM using Bounding Box and Center Point
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            
            masks, scores, _ = sam_predictor.predict(
                point_coords=np.array([[cx, cy]]), 
                point_labels=np.array([1]),
                box=np.array([x1, y1, x2, y2]), 
                multimask_output=True
            )
            
            # Take the mask SAM is most confident in
            best_idx = np.argmax(scores)
            raw_mask = (masks[best_idx] * 255).astype(np.uint8)
            
            # Clean up the mask (remove fins)
            clean_mask = remove_fins_gentle(raw_mask, box)
            
            # Draw Mask
            # Cycle through colors for different fish: Green, Cyan, Magenta, Yellow
            colors = [(0, 255, 0), (255, 255, 0), (255, 0, 255), (0, 255, 255)]
            color = colors[i % len(colors)]
            annotated_frame = apply_mask_overlay(annotated_frame, clean_mask, color=color, alpha=0.4)
            
            # Draw Bounding Box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(annotated_frame, f"Fish {i+1} (Conf: {conf:.2f})", (x1, max(10, y1 - 10)), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
        # Save output
        out_path = os.path.join(output_dir, filename)
        cv2.imwrite(out_path, annotated_frame)
        print(f"     [Success] Saved to output_frames/{filename}")
        
    print("\nPipeline complete! Check the 'output_frames' folder for your results.")

if __name__ == '__main__':
    main()
