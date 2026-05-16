"""
SAM Benchmark Comparison Script
===============================
This script directly compares the Classic Computer Vision mask
with Meta's Segment Anything Model (SAM), using the Classic mask's
bounding box as a spatial prompt for SAM on enhanced frames.

It produces side-by-side visual comparisons.
"""

import os
import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

from build_fish_dataset import (
    build_background,
    static_exclude_mask,
    combine_large_components,
    find_first_video
)
from image_enhancements import enhance_frame, motion_mask_enhanced

# Import SAM
try:
    from segment_anything import sam_model_registry, SamPredictor
except ImportError:
    print("ERROR: segment-anything is not installed. Run: pip install segment-anything")
    exit(1)


def compare_frames(fish_id: str, view: str, video_path: str, bg_bgr: np.ndarray, exclude_mask: np.ndarray,
                   sam_predictor: SamPredictor, output_dir: str, num_frames=5):
    """
    Extracts a few frames, computes classical mask -> bbox -> SAM mask,
    and saves side-by-side plots for publication benchmarking.
    """
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stride = max(1, total_frames // (num_frames + 2))
    
    # Pre-process background to match motion_mask pipeline
    # Note: apply_gamma is explicitly disabled to avoid background subtraction failure
    bg_enhanced = enhance_frame(bg_bgr, low_visibility=False, apply_gamma=False)
    
    os.makedirs(output_dir, exist_ok=True)
    
    extracted = 0
    i = 0
    
    print(f"Sampling {num_frames} frames from {fish_id} {view}...")
    
    while extracted < num_frames and i < total_frames:
        ok, frame_bgr = cap.read()
        if not ok:
            break
            
        if i % stride != 0 or i == 0:  # Skip frame 0 as it's often the background itself
            i += 1
            continue
            
        # 1. Classical Motion Mask (from raw pixels to guarantee robust background subtraction)
        classic_mask = motion_mask_enhanced(
            frame_bgr, bg_bgr, exclude_mask, diff_thresh=8 if view == 'top' else 6
        )
        min_area = 200 if view == 'top' else 150
        classic_mask = combine_large_components(classic_mask, min_area_px=min_area)
        
        # Check if classic CV found anything
        if cv2.countNonZero(classic_mask) < min_area:
            i += 1
            continue
            
        # 2. Extract Bounding Box Prompt for SAM
        contours, _ = cv2.findContours(classic_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not contours:
            i += 1
            continue
        c = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        bbox = np.array([x, y, x + w, y + h])
        
        # 3. Enhanced frame for SAM Prompting (Better contrast)
        frame_visual_enh = enhance_frame(frame_bgr, low_visibility=True, apply_gamma=True)
        frame_rgb = cv2.cvtColor(frame_visual_enh, cv2.COLOR_BGR2RGB)
        
        # 4. SAM Prediction
        sam_predictor.set_image(frame_rgb)
        masks, scores, _ = sam_predictor.predict(
            box=bbox,
            multimask_output=False,
        )
        sam_mask = masks[0]  # First (and only) mask
        
        # Calculate IoU
        classic_bool = classic_mask > 0
        sam_bool = sam_mask > 0
        intersection = np.logical_and(classic_bool, sam_bool).sum()
        union = np.logical_or(classic_bool, sam_bool).sum()
        iou = intersection / union if union > 0 else 0
        
        # 5. Save Visualization
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # A) Original Dark/Murky Frame
        axes[0].imshow(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        axes[0].set_title(f"A. Raw Frame ({view})")
        axes[0].axis('off')
        
        # B) Classic CV Mask Result
        vis_classic = frame_rgb.copy()
        vis_classic[classic_bool] = vis_classic[classic_bool] * 0.5 + np.array([255, 0, 0]) * 0.5
        # Draw bbox
        cv2.rectangle(vis_classic, (x, y), (x + w, y + h), (0, 255, 0), 2)
        axes[1].imshow(vis_classic)
        axes[1].set_title(f"B. Classic CV Mask + BBox Prompt")
        axes[1].axis('off')
        
        # C) SAM Mask Result
        vis_sam = frame_rgb.copy()
        vis_sam[sam_bool] = vis_sam[sam_bool] * 0.5 + np.array([0, 255, 0]) * 0.5
        axes[2].imshow(vis_sam)
        axes[2].set_title(f"C. SAM Mask (IoU: {iou:.3f})")
        axes[2].axis('off')
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, f"{fish_id}_{view}_{extracted:02d}.jpg")
        plt.savefig(out_path, dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        extracted += 1
        i += 1
        
    cap.release()
    print(f"  -> Saved {extracted} frames successfully.")

def main():
    print("=" * 60)
    print(" SAM vs Classic CV - Benchmark Generator")
    print("=" * 60)
    
    checkpoint_path = "sam_vit_b_01ec64.pth"
    if not os.path.exists(checkpoint_path):
        print(f"Downloading SAM Model ({checkpoint_path}) ~360MB ...")
        # Lightweight powershell download
        os.system(f"powershell -c \"Invoke-WebRequest -Uri 'https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth' -OutFile '{checkpoint_path}'\"")
        
    print("Loading SAM Model...")
    DEVICE = "cpu"
    print(f"Using device: {DEVICE} (Forced CPU safely due to 4GB VRAM limit)")
    
    sam = sam_model_registry["vit_b"](checkpoint=checkpoint_path)
    sam.to(device=DEVICE)
    predictor = SamPredictor(sam)
    
    dataset_root = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(dataset_root, "SAM_Comparisons")
    
    test_fish = ["fish01", "fish2"]
    
    for fish_id in test_fish:
        folder = os.path.join(dataset_root, fish_id)
        if not os.path.isdir(folder):
            continue
            
        print(f"\nProcessing {fish_id}...")
        
        top_dir = os.path.join(folder, "top view")
        front_dir = os.path.join(folder, "front view")

        top_video = find_first_video(top_dir) if os.path.isdir(top_dir) else None
        front_video = find_first_video(front_dir) if os.path.isdir(front_dir) else None

        if top_video:
            bg_top = build_background(top_video)
            ex_top = static_exclude_mask(bg_top, "top")
            compare_frames(fish_id, "top", top_video, bg_top, ex_top, predictor, output_dir)
            
        if front_video:
            bg_front = build_background(front_video)
            ex_front = static_exclude_mask(bg_front, "front")
            compare_frames(fish_id, "front", front_video, bg_front, ex_front, predictor, output_dir)
            
    print(f"\nDone! Benchmark images saved to: {output_dir}")

if __name__ == "__main__":
    main()
