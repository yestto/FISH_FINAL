import cv2
import os
import glob
import numpy as np
from ultralytics import FastSAM

# Configuration
INPUT_DIR = r"C:\Users\shain\Downloads\FISH DATASET\FISH DATASET\fish2"
OUTPUT_DIR = r"C:\Users\shain\Downloads\FISH DATASET\FISH DATASET\SAM_AI_DATASET"
FISH_ID = "fish2"
NUM_FRAMES_PER_CAMERA = 25

# Setup paths
fish_out_dir = os.path.join(OUTPUT_DIR, FISH_ID)
img_dir = os.path.join(fish_out_dir, "images")
lbl_dir = os.path.join(fish_out_dir, "labels")
vis_dir = os.path.join(fish_out_dir, "visuals")

for d in [img_dir, lbl_dir, vis_dir]:
    os.makedirs(d, exist_ok=True)

print(f"Initializing FastSAM Model...")
model = FastSAM('FastSAM-s.pt')

def extract_and_annotate(video_path, prefix):
    print(f"\nProcessing {video_path}...")
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    if total_frames <= 0:
        print("Could not read video.")
        return

    # Evenly spaced frame indices
    step = max(1, total_frames // NUM_FRAMES_PER_CAMERA)
    frame_indices = list(range(0, total_frames, step))[:NUM_FRAMES_PER_CAMERA]
    
    for count, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue
            
        base_name = f"{prefix}_{FISH_ID}_frame_{frame_idx:06d}"
        img_path = os.path.join(img_dir, base_name + ".jpg")
        lbl_path = os.path.join(lbl_dir, base_name + ".txt")
        vis_path = os.path.join(vis_dir, base_name + "_annotated.jpg")
        
        # Save raw image
        cv2.imwrite(img_path, frame)
        
        # Run FastSAM prediction
        results = model(img_path, conf=0.25, iou=0.9, verbose=False) # Zero-shot Auto-segmentation
        
        best_mask = None
        max_area = 0
        img_h, img_w = frame.shape[:2]
        total_area = img_h * img_w
        
        if results and len(results) > 0:
            result = results[0]
            if result.masks is not None:
                # Find the largest mask that isn't the entire background
                for segments in result.masks.xyn:
                    if len(segments) < 3:
                        continue
                        
                    # Calculate approximate area using shoelace formula
                    x = segments[:, 0]
                    y = segments[:, 1]
                    area_ratio = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                    
                    # We want the largest object that is less than 85% of the total screen
                    if area_ratio > max_area and area_ratio < 0.85:
                        max_area = area_ratio
                        best_mask = segments
        
        if best_mask is not None:
            # Format to YOLO format: <class> <x1> <y1> <x2> <y2> ...
            # class is 0 (fish)
            flat_coords = " ".join([f"{x:.6f} {y:.6f}" for x, y in best_mask])
            with open(lbl_path, "w") as f:
                f.write(f"0 {flat_coords}\n")
                
            # Draw visualization to verify
            vis_img = frame.copy()
            # De-normalize coordinates for drawing
            pts = (best_mask * np.array([img_w, img_h])).astype(np.int32)
            cv2.polylines(vis_img, [pts], True, (0, 255, 0), 2)
            cv2.imwrite(vis_path, vis_img)
            
            print(f"[{count+1}/{len(frame_indices)}] Processed {base_name} - Mask Found!")
        else:
            print(f"[{count+1}/{len(frame_indices)}] Processed {base_name} - No suitable mask found.")

    cap.release()

if __name__ == "__main__":
    top_videos = glob.glob(os.path.join(INPUT_DIR, "top view", "*.mp4"))
    front_videos = glob.glob(os.path.join(INPUT_DIR, "front view", "*.mp4"))
    
    if top_videos:
        extract_and_annotate(top_videos[0], "top")
    else:
        print("No top view video found.")
        
    if front_videos:
        extract_and_annotate(front_videos[0], "front")
    else:
        print("No front view video found.")
        
    print(f"\nFinished! All outputs saved to structurally isolated folder:\n{fish_out_dir}")
