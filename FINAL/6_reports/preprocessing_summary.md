# Fish Biometric Dataset — Preprocessing Pipeline Summary

## Input
- **Raw Source**: 7,144 synchronized Top-view + Front-view video frame pairs across 14 fish
- **Models Used**: YOLOv12m (object detection) + SAM ViT-B (instance segmentation)
- **Ground Truth**: Manually measured fish lengths, heights, and weights

---

## Preprocessing Steps Applied

### Step 1: AI-Based Feature Extraction (YOLO + SAM Pipeline)
- Detected fish in each frame using YOLOv12m bounding boxes
- Generated precise fish body masks using Meta's Segment Anything Model (SAM ViT-B)
- **Top View**: Standard SAM with box-only prompting
- **Front View**: Enhanced SAM with CLAHE contrast enhancement + box+center-point prompting (to handle murky underwater conditions)
- Applied morphological fin removal (15% elliptical kernel) to isolate the main fish body
- Strict bounding box clipping to eliminate any background noise outside the detected fish region
- Extracted 13 biometric measurements per frame from the mask contours

### Step 2: Dual-Camera Calibration (Pixels → Centimeters)
- Calibrated the Top camera independently: `Top_PxPerCm = median(Top_Length_px) / Ground_Truth_Length`
- Calibrated the Front camera independently: `Front_PxPerCm = median(Front_Length_px) / Ground_Truth_Length`
- This corrects for the different pixel scales caused by different camera distances (top camera at 60cm height vs. underwater camera at 15–20cm from bottom)
- Converted all pixel measurements to real-world centimeters
- Calculated derived 3D biometrics: Volume, Surface Area, Condition Factor, Aspect Ratio, etc.

### Step 3: Outlier Removal (Perspective Distortion Filter)
- Removed frames where **Height ≥ 80% of Length** (physically impossible — caused by fish swimming too close to the underwater camera lens)
- Removed frames with **Aspect Ratio outside 2.0–8.0** range (indicates extreme fish orientation or partial occlusion)
- Removed frames with **YOLO confidence score < 0.80** (low-quality detections)
- Removed frames with any **negative or zero** dimension values
- **Result**: 7,131 → 5,249 rows (removed 1,882 distorted frames, 26.4%)

### Step 4: Feature Selection
- Dropped all non-biometric metadata columns (image file paths, frame indices, mask pixel counts, blur scores, FPS values, timestamps)
- Retained 15 columns: FishID, Weight (g), and 13 biometric features
- **Result**: 26 → 15 columns

### Step 5: Duplicate Removal
- Identified and removed 129 exact duplicate rows
- **Result**: 5,249 → 5,120 rows

### Step 6: Class Balancing
- Capped the maximum frames per fish at 300 (to prevent model bias toward over-represented fish)
- Applied random sampling with fixed seed (seed=42) for reproducibility
- Fish with fewer than 300 frames were kept intact
- **Result**: 5,120 → 3,380 rows

---

## Final Dataset Summary

| Property | Value |
|----------|-------|
| **File** | `fish_frames_production_FINAL_CLEAN.csv` |
| **Total Rows** | 3,380 |
| **Total Columns** | 15 |
| **Unique Fish** | 14 |
| **Frames per Fish** | 63 – 300 |
| **Missing Values** | 0 |
| **Duplicates** | 0 |
| **Target Variable** | Weight (g) |

## Features (13 Input Variables)

| Feature | Description | Correlation with Weight |
|---------|-------------|------------------------|
| Length (cm) | Major axis of top-view mask | +0.84 |
| Equivalent Diameter (cm) | Diameter of circle with same area | +0.84 |
| Perimeter (cm) | Contour perimeter from top view | +0.83 |
| Surface Area (cm²) | Estimated 3D surface area | +0.80 |
| Volume (cm³) | Estimated 3D ellipsoidal volume | +0.78 |
| Area (cm²) | 2D mask area from top view | +0.78 |
| Width (cm) | Minor axis of top-view mask | +0.77 |
| Height (cm) | Minor axis of front-view mask | +0.63 |
| Rectangularity | Area / (Length × Width) | +0.23 |
| Aspect Ratio | Length / Width | +0.21 |
| Condition Factor (K) | Weight / Length³ × 100 | −0.23 |
| Elongation | Width / Length | −0.19 |
| Compactness | Perimeter² / Area | +0.02 |
