import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import torch
import numpy as np

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("segment_anything is not installed.")
    exit(1)

def process_view(frame_dir, view, raw_file, mask_file, predictor):
    print(f"Processing {view} view...")
    sam_output_file = os.path.join(frame_dir, f"50_{view}_SAM_mask.png")
    raw_path = os.path.join(frame_dir, raw_file)
    mask_path = os.path.join(frame_dir, mask_file)
    
    if not os.path.exists(raw_path):
        print(f"Missing raw image: {raw_path}")
        return
    if not os.path.exists(mask_path):
        print(f"Missing mask: {mask_path}")
        return
        
    raw_img = cv2.imread(raw_path)
    classic_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    
    contours, _ = cv2.findContours(classic_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        print(f"No contours found for {view}")
        return
    c = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(c)
    bbox = np.array([x, y, x + w, y + h])
    
    raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
    
    print(f"  Running SAM prediction logic...")
    predictor.set_image(raw_rgb)
    masks, _, _ = predictor.predict(box=bbox, multimask_output=False)
    sam_mask = (masks[0] * 255).astype(np.uint8)
    
    classic_bool = classic_mask > 127
    sam_bool = sam_mask > 127

    print(f"  Saving output masks...")
    cv2.imwrite(sam_output_file, sam_mask)
    
    vis_file = os.path.join(frame_dir, f"51_SAM_vs_Classic_Vis_{view}.jpg")
    vis = raw_rgb.copy()
    vis[classic_bool] = vis[classic_bool] * 0.5 + np.array([255, 0, 0]) * 0.5
    vis[sam_bool] = vis[sam_bool] * 0.5 + np.array([0, 255, 0]) * 0.5
    cv2.imwrite(vis_file, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
    
    print(f"Successfully processed {view} view.")

def main():
    # Auto-detect: uses CUDA on RTX, falls back to CPU
    if torch.cuda.is_available():
        DEVICE = "cuda"
        print(f"Loading SAM on GPU: {torch.cuda.get_device_name(0)} - FAST MODE")
    else:
        DEVICE = "cpu"
        print("Loading SAM Model onto CPU (no GPU detected)...")
    checkpoint_path = "sam_vit_b_01ec64.pth"
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)

    frame_dir = r"C:\Users\shain\Downloads\FISH DATASET\FISH DATASET\final stages clean no repeat\fish01\frame_001728"
    
    print(f"\nProcessing target frame: {frame_dir}")
    process_view(frame_dir, "top", "01_top_raw.jpg", "40_top_final_mask.png", predictor)
    process_view(frame_dir, "front", "21_front_raw.jpg", "41_front_final_mask.png", predictor)
    print("\nAll done!")

if __name__ == "__main__":
    main()
