# =============================================================
# REQUIREMENTS:
#   pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
#   pip install segment-anything opencv-python tqdm
#   Download checkpoint: sam_vit_b_01ec64.pth (auto-downloaded if missing)
# RTX GPU: Will automatically use CUDA for fast processing (~2hrs)
# CPU only: Will fall back to CPU (very slow, ~15hrs)
# =============================================================
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import cv2
import torch
import numpy as np
from tqdm import tqdm

try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("segment_anything is not installed. Run: pip install segment-anything")
    exit(1)

def main():
    print("="*60)
    print("  SAM ACADEMIC BENCHMARK - CLEAN NO REPEAT DATASET")
    print("="*60)
    
    checkpoint_path = "sam_vit_b_01ec64.pth"
    if not os.path.exists(checkpoint_path):
        os.system(f"powershell -c \"Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth' -OutFile '{checkpoint_path}'\"")

    # Auto-detect GPU — uses CUDA if available (RTX), otherwise CPU
    if torch.cuda.is_available():
        DEVICE = "cuda"
        gpu_name = torch.cuda.get_device_name(0)
        vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Loading SAM on GPU: {gpu_name} ({vram_gb:.1f}GB VRAM) - FAST MODE")
    else:
        DEVICE = "cpu"
        print("No GPU detected. Loading SAM on CPU (slow mode)...")
    
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    
    root_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "final stages clean no repeat")
    
    if not os.path.exists(root_dir):
        print(f"Could not find directory: {root_dir}")
        return

    fish_folders = [f for f in os.listdir(root_dir) if f.startswith("fish") and os.path.isdir(os.path.join(root_dir, f))]
    
    # We will log IOUs to a CSV to give you quantitative benchmark data for the academic paper!
    iou_csv_path = os.path.join(root_dir, "SAM_IOU_Benchmark_All.csv")
    if not os.path.exists(iou_csv_path):
        with open(iou_csv_path, "w") as f:
            f.write("Fish,Frame,View,ClassicArea,SAMArea,IOU\n")
            
    # Function to process one view
    def process_view(frame_dir, fish_id, frame_id, view, raw_file, mask_file):
        sam_output_file = os.path.join(frame_dir, f"50_{view}_SAM_mask.png")
        if os.path.exists(sam_output_file):
            return # Already done (this allows you to stop and resume without losing progress)
            
        raw_path = os.path.join(frame_dir, raw_file)
        mask_path = os.path.join(frame_dir, mask_file)
        
        if not os.path.exists(raw_path) or not os.path.exists(mask_path):
            return
            
        raw_img = cv2.imread(raw_path)
        classic_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        
        # 1. Get Bounding Box exactly from the Classic CV Mask generated previously
        contours, _ = cv2.findContours(classic_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            return
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        bbox = np.array([x, y, x + w, y + h])
        
        # 2. SAM zero-shot prediction
        raw_rgb = cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB)
        predictor.set_image(raw_rgb)
        masks, _, _ = predictor.predict(box=bbox, multimask_output=False)
        sam_mask = (masks[0] * 255).astype(np.uint8)
        
        # 3. Calculate Academic IoU Metric
        classic_bool = classic_mask > 127
        sam_bool = sam_mask > 127
        intersection = np.logical_and(classic_bool, sam_bool).sum()
        union = np.logical_or(classic_bool, sam_bool).sum()
        iou = intersection / union if union > 0 else 0
        
        # 4. Save isolated SAM mask inside the frame folder
        cv2.imwrite(sam_output_file, sam_mask)
        
        # 5. Save an overlay visualization (Red = Classic, Green = SAM)
        vis_file = os.path.join(frame_dir, f"51_SAM_vs_Classic_Vis_{view}.jpg")
        vis = raw_rgb.copy()
        vis[classic_bool] = vis[classic_bool] * 0.5 + np.array([255, 0, 0]) * 0.5
        vis[sam_bool] = vis[sam_bool] * 0.5 + np.array([0, 255, 0]) * 0.5
        cv2.imwrite(vis_file, cv2.cvtColor(vis, cv2.COLOR_RGB2BGR))
        
        # 6. Log IoU
        with open(iou_csv_path, "a") as f:
            f.write(f"{fish_id},{frame_id},{view},{classic_bool.sum()},{sam_bool.sum()},{iou:.4f}\n")
            
    print(f"Starting processing... (This will dynamically save to SAM_IOU_Benchmark_All.csv)")
    for fish_id in tqdm(fish_folders, desc="Fishes"):
        fish_path = os.path.join(root_dir, fish_id)
        frame_folders = [f for f in os.listdir(fish_path) if f.startswith("frame_")]
        
        for frame_folder in tqdm(frame_folders, desc=f"{fish_id} frames", leave=False):
            frame_dir = os.path.join(fish_path, frame_folder)
            
            # Top
            process_view(frame_dir, fish_id, frame_folder, "top", "01_top_raw.jpg", "40_top_final_mask.png")
            # Front
            process_view(frame_dir, fish_id, frame_folder, "front", "21_front_raw.jpg", "41_front_final_mask.png")

if __name__ == "__main__":
    main()
