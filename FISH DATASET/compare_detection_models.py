"""
Compare Detection Models on Unseen Frames
==========================================
Tests all 3 detection models (YOLO11m, YOLO12m, YOLOv8m) on the
unseen frames from the frames/fraes folder.

Picks the best model based on confidence scores and detection rate,
then saves all bounding box detections to a CSV.

Usage:
    python compare_detection_models.py
"""

import os
import re
import csv
import numpy as np
from pathlib import Path
from collections import defaultdict

# -- PATHS --
FRAMES_ROOT = Path(r"C:\Users\shain\Downloads\FISH DATASET\frames\fraes")
FISH_TRAIN  = Path(r"C:\Users\shain\Downloads\FISH DATASET\FISH-TRAIN\Fish Annotations")
FISH_DATASET = Path(r"C:\Users\shain\Downloads\FISH DATASET\FISH DATASET")
OUTPUT_DIR  = FISH_DATASET / "model_comparison_results"
OUTPUT_DIR.mkdir(exist_ok=True)

# -- TRAINED MODELS --
MODELS = {
    "YOLO11m":  FISH_TRAIN / "yolo_training_runs" / "fish_yolo11m_full"  / "weights" / "best.pt",
    "YOLO12m":  FISH_TRAIN / "yolo_training_runs" / "fish_yolo12m_full"  / "weights" / "best.pt",
    "YOLOv8m":  FISH_TRAIN / "yolo_training_runs" / "fish_yolov8m_1280"  / "weights" / "best.pt",
}


def collect_unseen_frames():
    """
    Collect all unseen frame images from the frames folder.
    Returns list of (image_path, fish_id, view, frame_number) tuples.
    
    Filename patterns:
      fish01-frontview-frame-144.jpg      -> (fish01, front, 144)
      fish01-topview-frame-144.jpg        -> (fish01, top, 144)
      fish01-topview-192.168...-frame-144.jpg -> SKIP (duplicate of short-name top)
    """
    frames = []
    
    for fish_dir in sorted(FRAMES_ROOT.iterdir()):
        if not fish_dir.is_dir():
            continue
        
        # Extract fish ID from folder name: "fish01_frames" -> "fish01"
        fish_id = fish_dir.name.replace("_frames", "")
        
        for img_file in sorted(fish_dir.glob("*.jpg")):
            fname = img_file.name
            
            # Skip the long-name duplicates (they contain IP addresses)
            if "192.168" in fname:
                continue
            
            # Parse: fish01-frontview-frame-144.jpg
            #    or: fish01-topview-frame-144.jpg
            if "-frontview-frame-" in fname:
                view = "front"
                frame_num = int(fname.split("-frame-")[1].replace(".jpg", ""))
            elif "-topview-frame-" in fname:
                view = "top"
                frame_num = int(fname.split("-frame-")[1].replace(".jpg", ""))
            else:
                continue
            
            frames.append((img_file, fish_id, view, frame_num))
    
    return frames


def run_model_on_frames(model_path, frames, model_name):
    """Run a YOLO model on all frames and return detection stats."""
    from ultralytics import YOLO
    
    print(f"\n{'='*60}")
    print(f"  Running: {model_name}")
    print(f"  Model:   {model_path}")
    print(f"  Frames:  {len(frames)}")
    print(f"{'='*60}")
    
    model = YOLO(str(model_path))
    
    detections = []
    detected_count = 0
    total_conf = 0.0
    
    batch_size = 4
    for i in range(0, len(frames), batch_size):
        batch = frames[i:i+batch_size]
        img_paths = [str(f[0]) for f in batch]
        
        results = model(img_paths, verbose=False, imgsz=640)
        
        for j, result in enumerate(results):
            img_path, fish_id, view, frame_num = batch[j]
            
            if len(result.boxes) > 0:
                confs = result.boxes.conf.cpu().numpy()
                best_idx = confs.argmax()
                best_conf = float(confs[best_idx])
                box = result.boxes.xyxy[best_idx].cpu().numpy()
                
                x1, y1, x2, y2 = box
                box_w = x2 - x1
                box_h = y2 - y1
                
                detected_count += 1
                total_conf += best_conf
                
                detections.append({
                    "model": model_name,
                    "fish_id": fish_id,
                    "view": view,
                    "frame": frame_num,
                    "img_path": str(img_path),
                    "confidence": round(best_conf, 4),
                    "x1": round(float(x1), 1),
                    "y1": round(float(y1), 1),
                    "x2": round(float(x2), 1),
                    "y2": round(float(y2), 1),
                    "box_w_px": round(float(box_w), 1),
                    "box_h_px": round(float(box_h), 1),
                    "img_w": result.orig_shape[1],
                    "img_h": result.orig_shape[0],
                    "num_detections": len(result.boxes),
                })
            else:
                detections.append({
                    "model": model_name,
                    "fish_id": fish_id,
                    "view": view,
                    "frame": frame_num,
                    "img_path": str(img_path),
                    "confidence": 0.0,
                    "x1": 0, "y1": 0, "x2": 0, "y2": 0,
                    "box_w_px": 0, "box_h_px": 0,
                    "img_w": result.orig_shape[1],
                    "img_h": result.orig_shape[0],
                    "num_detections": 0,
                })
        
        done = min(i + batch_size, len(frames))
        if done % (batch_size * 5) == 0 or done == len(frames):
            print(f"  Processed {done}/{len(frames)} frames...")
    
    detection_rate = detected_count / len(frames) * 100
    avg_conf = total_conf / max(detected_count, 1)
    
    print(f"\n  Results for {model_name}:")
    print(f"    Detection rate: {detected_count}/{len(frames)} ({detection_rate:.1f}%)")
    print(f"    Avg confidence: {avg_conf:.4f}")
    
    return detections, detection_rate, avg_conf


def main():
    print("Collecting unseen frames from:", FRAMES_ROOT)
    frames = collect_unseen_frames()
    print(f"Found {len(frames)} total frames (excluding IP-address duplicates)")
    
    # Per-fish breakdown
    fish_counts = defaultdict(lambda: {"top": 0, "front": 0})
    for _, fish_id, view, _ in frames:
        fish_counts[fish_id][view] += 1
    
    print(f"\nPer-fish frame counts:")
    for fish_id in sorted(fish_counts.keys()):
        c = fish_counts[fish_id]
        print(f"  {fish_id}: top={c['top']}, front={c['front']}")
    
    # Run each model
    all_results = {}
    for model_name, model_path in MODELS.items():
        if not model_path.exists():
            print(f"\n  WARNING: Model not found: {model_path}")
            continue
        
        detections, det_rate, avg_conf = run_model_on_frames(
            model_path, frames, model_name
        )
        all_results[model_name] = {
            "detections": detections,
            "detection_rate": det_rate,
            "avg_confidence": avg_conf,
        }
    
    # Compare
    print("\n" + "=" * 60)
    print("MODEL COMPARISON SUMMARY")
    print("=" * 60)
    print(f"{'Model':<15} {'Detection Rate':>15} {'Avg Confidence':>15}")
    print("-" * 45)
    
    best_model = None
    best_score = -1
    
    for model_name, result in all_results.items():
        score = result["detection_rate"] * result["avg_confidence"]
        print(f"{model_name:<15} {result['detection_rate']:>14.1f}% {result['avg_confidence']:>14.4f}")
        if score > best_score:
            best_score = score
            best_model = model_name
    
    print(f"\nBEST MODEL: {best_model}")
    
    # Save best model detections
    best_detections = all_results[best_model]["detections"]
    csv_path = OUTPUT_DIR / f"best_detections_{best_model}.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=best_detections[0].keys())
        writer.writeheader()
        writer.writerows(best_detections)
    
    print(f"\nSaved {len(best_detections)} detections to: {csv_path}")
    
    # Save ALL model detections for comparison
    for model_name, result in all_results.items():
        csv_all = OUTPUT_DIR / f"detections_{model_name}.csv"
        with open(csv_all, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=result["detections"][0].keys())
            writer.writeheader()
            writer.writerows(result["detections"])
    
    # Save summary
    summary_path = OUTPUT_DIR / "model_comparison_summary.txt"
    with open(summary_path, "w") as f:
        f.write("FISH DETECTION MODEL COMPARISON\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total unseen frames tested: {len(frames)}\n")
        f.write(f"Frames source: {FRAMES_ROOT}\n\n")
        for model_name, result in all_results.items():
            f.write(f"{model_name}:\n")
            f.write(f"  Detection Rate: {result['detection_rate']:.1f}%\n")
            f.write(f"  Avg Confidence: {result['avg_confidence']:.4f}\n\n")
        f.write(f"\nBest Model: {best_model}\n")
    
    print(f"Saved summary to: {summary_path}")
    print(f"\nDone! Next: use {best_model} detections for measurement extraction.")


if __name__ == "__main__":
    main()
