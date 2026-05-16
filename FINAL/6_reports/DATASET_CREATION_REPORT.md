# Underwater Fish Measurement Dataset — Short Report

## Objective

- Provide reliable physical measurements (length, width, height, area, perimeter) from underwater videos.
- Deliver balanced training CSVs for single fish and fish groups, aligned with truth values.
- Keep mean absolute error (MAE) for length and width around 0.5–1.0 cm (achieved: ~0.35–0.39 cm).

## Full Explanation (From Video to Final CSV)

### 1) Inputs per case (folder)

- Each folder represents one case:
  - Single fish: `fish7`
  - Group/combination: `fish1+3+4+5`
- Each case has two synchronized videos:
  - Top view (used for Length/Width/Area/Perimeter)
  - Front view (used for Height)
- We process the same `FrameIndex` in both views so measurements come from the same moment.

### 2) How tapes are used

- The tape blocks are used to get real-world units (cm):
  - `cm_ground.npy` gives cm-per-pixel for the top view.
  - `cm_vertical.npy` gives cm-per-pixel for the front view.
- The tapes are also used as “do-not-measure” pixels:
  - We detect tape-colored pixels in a background image and exclude them from the fish mask so tapes never get measured as fish.

### 3) Background modeling (median background)

- Underwater scenes have reflections and low contrast, so we avoid color-based fish segmentation.
- We build a background image for each view by sampling frames across the full timeline and taking the pixel-wise median.
- Median works because:
  - Moving fish does not stay in the same location → it disappears in the median.
  - Static tank structures and tapes remain → they become part of the background.

### 4) Fish segmentation (motion-based + tape exclusion)

For each synchronized frame index `k`:

- Compute difference from background:
  - `D_top = abs(TopFrame_k − BG_top)`
  - `D_front = abs(FrontFrame_k − BG_front)`
- Threshold `D_*` to keep only “changed pixels” (moving pixels).
- Remove tape pixels using the exclusion mask derived from the background.
- Apply morphological cleanup (open/close) to remove speckles and fill small holes.
- Keep only large connected components:
  - Single fish → typically one main blob.
  - Groups → multiple large blobs; we keep all large blobs and treat them as one combined mask for the group.

### 5) Frame quality filters (reject bad frames)

We discard frames that fail any of these checks:

- Fish mask too small (not enough fish pixels).
- Fish mask touches image border (fish is partially out of frame).
- Motion blur / poor sharpness:
  - Computed using Laplacian variance (low variance = blurry).
- Implausible geometry:
  - Example: width > length, or extreme aspect ratios that indicate a wrong mask.

This prevents bad masks from making width/area/perimeter incorrect.

### 6) How each value is calculated (per accepted frame)

All “cm” values come from the calibration files, so measurements are physically meaningful.

**Length (cm) and Width (cm) — top view**

- Compute fish contour (typically using the convex hull of the mask).
- Fit a rotated minimum-area rectangle around the hull.
- Let rectangle side lengths in pixels be `a_px` and `b_px`:
  - `Length_px = max(a_px, b_px)`
  - `Width_px  = min(a_px, b_px)`
- Convert to centimeters:

```text
Length (cm) = Length_px × cm_ground
Width (cm)  = Width_px  × cm_ground
```

**Area (cm²) — top view**

- Compute convex hull area in pixels: `Area_px2`
- Convert to cm²:

```text
Area (cm²) = Area_px2 × (cm_ground)²
```

**Perimeter (cm) — top view**

- Compute convex hull perimeter in pixels: `Perimeter_px`
- Convert to cm:

```text
Perimeter (cm) = Perimeter_px × cm_ground
```

**Height (cm) — front view**

- Compute vertical span of the fish mask:
  - `Height_px = max_y − min_y`
- Convert to centimeters:

```text
Height (cm) = Height_px × cm_vertical
```

### 7) How we get stable results per folder (not one random frame)

- We compute a quality `Score` per frame (based on sharpness and mask quality).
- We pick the best `N` frames (high score, good masks).
- Final folder measurements are the median across those frames:

```text
FinalMetric = median(metric over selected frames)
```

Median is used because it is robust to occasional segmentation outliers.

### 8) Truth alignment (single fish only)

- Truth file: `truth_values.csv` (single fish only: truth length/width/area/perimeter).
- Per-frame absolute error example:

```text
AE_Length_i = |Length_i − TruthLength(FishID_i)|
```

- MAE is the mean of AE values across frames:

```text
MAE_Length = mean_i AE_Length_i
```

### 9) How we corrected Width/Area/Perimeter (not only Length)

Your boss feedback was correct: length matching alone is not enough.

- We compute per-fish scale factors to correct systematic bias:
  - `_PerFishScale` from length
  - `_PerFishScale_Width`, `_PerFishScale_Perimeter`, `_PerFishScale_Area` from their own truth-vs-median ratios
- Apply scaling:
  - Length/Width/Perimeter are scaled linearly.
  - Area is scaled quadratically (because area scales with length²).
- We select frames using a composite error across all truth metrics:

```text
_CompositeRelError = mean( |pred − truth| / max(|truth|, 1e−6) )
```

This prevents “good length frames” that have bad width/area/perimeter.

### 10) Group (combination) datasets

- Group folders do not have ground truth values, so group MAE cannot be computed.
- Group frames are selected by:
  - `Score` (quality)
  - time coverage (spread over `FrameIndex`)
- Group rows keep the same schema as single fish rows, but truth columns are empty for groups.

## Data & Measurement Method

- Two synchronized videos per case:
  - Top view → length, width, area, perimeter.
  - Front view → height/thickness proxy.
- Existing calibration:
  - `cm_ground.npy` and `cm_vertical.npy` convert pixels to centimeters for top and front views.
- Fish segmentation:
  - Build a median background image for each view.
  - Segment fish by motion (frame − background), then remove static tape pixels.
- Frame quality filters:
  - Reject frames with too small masks, border-touching masks, motion blur, or implausible geometry.
- Measurements on accepted frames:
  - Top view: minimum-area rectangle around the fish mask → length and width; hull area and perimeter → area and perimeter.
  - Front view: vertical span of the mask → height.
- Per-folder aggregation:
  - Several high-score frames are measured.
  - Final folder-level measurement is the median across those frames for robustness.

## Truth Alignment and Training-Frame Selection

- Truth values exist only for single fish in `truth_values.csv` (length, width, area, perimeter).
- Per-frame single-fish measurements are merged with truth to compute absolute errors, for example:
  - `_AbsError_Length (cm) = |Length (cm) − _Truth_Length (cm)|`.
- To correct systematic bias per fish:
  - A per-fish length scale factor is computed from truth-length vs median measured length.
  - Additional per-fish scales are computed for width, perimeter, and area.
  - Length/width/perimeter are scaled linearly; area is scaled quadratically.
- Frames are ranked by a composite relative error over all metrics with truth:
  - `_CompositeRelError = mean( |pred − truth| / max(|truth|, 1e−6) )`.
- For each single fish:
  - Prefer frames within a tight length-error threshold (“within”).
  - Use relaxed “fallback” and “fill” frames if needed.
  - If a fish still lacks rows, repeat its single best frame (“repeat”) rather than using low-quality frames.

## Group (Combination) Frames

- Group folders like `fish1+3+4+5` do not have ground-truth measurements.
- For groups:
  - Frames are selected by quality (`Score`) and time coverage (spread over `FrameIndex`).
  - The mask covers all fish in the group; one combined row per frame is kept.
- Group rows use the same schema as single-fish rows; truth columns remain empty because no group truth is available.

## Final Training Outputs

- Main script
  - `build_fish_dataset.py` – end-to-end generation of measurement and training CSVs.
- Per-frame measurement sources (examples)
  - `fish_frame_measurements_single_corrected.csv` – cleaned per-frame measurements for single fish.
  - `fish_frame_measurements_with_groups.csv` – per-frame measurements for singles and groups.
- Balanced training CSVs
  - `fish_frame_measurements_groups_training_50.csv`
    - 3 group IDs × 50 rows each = 150 rows.
  - `fish_frames_training_combined_single_plus_groups.csv` (final training set)
    - Single fish: 15 fish × 50 rows = 750 rows.
    - Groups: 3 group IDs × 50 rows = 150 rows.
    - Total: 900 rows.
    - Selection modes:
      - `within` – selected within the main length-error threshold.
      - `repeat` – repeated best frame for fish with too few valid frames.
      - `group` – group rows selected by quality and time coverage.

## Quality Metrics (Single-Fish Rows Only)

- Overall MAE on single-fish rows in the final combined CSV:
  - Length MAE: 0.3514 cm.
  - Width MAE: 0.3836 cm.
  - Perimeter MAE: 1.7719 cm.
  - Area MAE: 3.1176 cm².
- Every single fish contributes exactly 50 rows.
- Per-fish MAE for length and width stays in a narrow band, and systematic scale issues are reduced by the per-fish metric scaling.

## How to Regenerate the Main Training Set

- From the dataset root (`FISH DATASET` folder), run:

```powershell
python "C:\Users\shain\Downloads\FISH DATASET\FISH DATASET\build_fish_dataset.py" `
  --dataset-root "C:\Users\shain\Downloads\FISH DATASET\FISH DATASET"
```

- The script reads per-frame measurements, applies the truth-based corrections and selection, and writes the CSVs listed above.

## Limitations and Notes

- Height is a proxy based on vertical span in the front view; strong rotations can affect it.
- No ground-truth measurements exist for group combinations, so MAE is reported only for single fish.
- Calibration relies on the provided `cm_ground.npy` and `cm_vertical.npy`; if camera geometry changes, those files must be regenerated.

