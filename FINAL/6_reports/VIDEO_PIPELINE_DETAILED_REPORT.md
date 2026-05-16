# Underwater Fish Measurement Dataset — Detailed Technical Report

This document explains, step-by-step, how measurements are computed from the videos (top + front), which techniques are used, how truth alignment is done for single fish, and what final datasets are produced.

Core implementation: [build_fish_dataset.py](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py)

## 1) Inputs and folder structure

Each case is a folder under the dataset root whose name starts with `fish`:

- Single fish example: `fish7`
- Group/combination example: `fish1+3+4+5`

Inside each case folder, the pipeline searches for:

- `top view/` → first `*.mp4` it finds
- `front view/` → first `*.mp4` it finds

Implementation detail: video search uses `find_first_video(...)` ([build_fish_dataset.py:L669-L679](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L669-L679)).

## 2) Calibration (pixels → centimeters)

All physical units are computed by converting pixels to centimeters using calibration values.

### 2.1 Video calibration (tape-based)

The script loads:

- `cm_ground.npy` → `cm_ground_per_px` for the top view
- `cm_vertical.npy` → `cm_vertical_per_px` for the front view

Load priority:

- Dataset root first
- Fallback to `fish01/` if not found in root

Implementation: `load_calibration(...)` ([build_fish_dataset.py:L34-L66](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L34-L66)).

### 2.2 Why tapes matter even if we do motion segmentation

The tapes are used in two ways:

- Calibration: they were used to create `cm_ground.npy` and `cm_vertical.npy`, which convert pixels to cm/cm² for every measurement.
- Safety: tape pixels are explicitly detected in the background and removed from the fish mask, so tapes are never measured as fish.

## 3) Per-frame processing (top+front synchronized)

Per-frame extraction outputs one CSV row per accepted frame (for single fish, and optionally for groups). This is the main “from video to numbers” stage.

Implementation: `process_pair_per_frame(...)` ([build_fish_dataset.py:L1298-L1407](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L1298-L1407)).

### 3.1 Background modeling (median background)

For each video, the script builds a background image by sampling frames across the whole timeline and taking the pixel-wise median.

- Samples: `sample_count = 120`
- Requires at least 10 readable frames

Implementation: `build_background(...)` ([build_fish_dataset.py:L949-L968](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L949-L968)).

Reason: fish is moving, so it disappears in the median; static structures (tank + tapes) remain.

### 3.2 Tape exclusion mask (static exclusion)

Tape pixels are detected from the background image using HSV thresholds:

- Red is detected in two HSV bands:
  - `(0, 80, 40) → (10, 255, 255)`
  - `(160, 80, 40) → (180, 255, 255)`
- Green (top view only):
  - `(35, 60, 40) → (90, 255, 255)`

The mask is then closed with a `(7×7)` rectangle kernel, 2 iterations.

Implementation: `static_exclude_mask(...)` ([build_fish_dataset.py:L971-L992](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L971-L992)).

### 3.3 Motion segmentation (background subtraction)

For each frame:

1) Blur both frame and background with a `(5×5)` Gaussian
2) Compute absolute difference
3) Convert to grayscale
4) Threshold to binary mask (diff threshold is configurable)
5) Remove tape pixels by `mask AND NOT exclude_mask`
6) Clean using morphology:
   - Open: ellipse `(5×5)`, 1 iteration
   - Close: ellipse `(9×9)`, 1 iteration
   - Erode 1 iteration + Dilate 2 iterations

Implementation: `motion_mask(...)` ([build_fish_dataset.py:L995-L1014](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L995-L1014)).

### 3.4 Keep only “large enough” connected components

The mask may contain small noisy blobs. We keep only connected components whose pixel area is at least a threshold (`min_area_px`).

Implementation: `combine_large_components(...)` ([build_fish_dataset.py:L1030-L1039](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L1030-L1039)).

### 3.5 Frame acceptance rules (quality filtering)

A frame is accepted only if all checks pass:

- Mask pixel count ≥ `min_area_top` and `min_area_front`
- Mask does not touch borders (margin `border_margin` pixels)
  - Implementation: `passes_border_margin(...)` ([build_fish_dataset.py:L1042-L1052](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L1042-L1052))
- Measurement geometry checks:
  - `Length > 0`, `Width > 0`, `Height > 0`
  - `Length >= Width`
  - `Length / Width >= min_aspect`
- Blur (sharpness) checks:
  - `blur_score(top_frame) >= min_blur`
  - `blur_score(front_frame) >= min_blur`
  - Implementation: `blur_score(...)` ([build_fish_dataset.py:L1017-L1019](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L1017-L1019))

### 3.6 Score used for ranking frames

For accepted frames:

```text
Score = (BlurTop + BlurFront) + 0.02 * TopMaskPixels
```

This prefers sharp frames and strong/clean top masks.

Implementation: score calculation appears in both aggregate and per-frame extraction:

- [build_fish_dataset.py:L1242-L1244](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L1242-L1244)
- [build_fish_dataset.py:L1382-L1399](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L1382-L1399)

## 4) How each measurement is computed (exact geometry)

All measurements are computed from the binary fish mask and converted using calibration.

### 4.1 Top view: Length, Width, Area, Perimeter

Technique:

- Convert mask to point set (`mask_to_points`)
- Compute convex hull
- Use convex hull for stable geometry (less noise sensitivity)
- Fit a minimum-area rotated rectangle to hull (handles rotated fish)

Implementation: `measure_top(...)` ([build_fish_dataset.py:L1127-L1143](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L1127-L1143)).

Formulas:

- Let hull rectangle side lengths in pixels be `w_px` and `h_px`:
  - `Length_px = max(w_px, h_px)`
  - `Width_px  = min(w_px, h_px)`
- Convert using `cm_ground_per_px`:

```text
Length (cm)    = Length_px * cm_ground_per_px
Width (cm)     = Width_px  * cm_ground_per_px
Area (cm²)     = Area_px   * (cm_ground_per_px)²
Perimeter (cm) = Perimeter_px * cm_ground_per_px
```

### 4.2 Front view: Height

Technique:

- Convert mask to point set
- Take the vertical extent: `max_y − min_y`

Implementation: `measure_height(...)` ([build_fish_dataset.py:L1146-L1151](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L1146-L1151)).

Formula:

```text
Height (cm) = (max_y - min_y) * cm_vertical_per_px
```

## 5) Aggregate per-folder measurements (one row per fish/group)

The “aggregate” dataset (`fish_measurements.csv`) is produced by:

- Extracting many candidate frames per folder
- Sorting by `Score`
- Taking the best `max_frames` frames
- Computing the median of each metric across those frames

Implementation: `process_pair(...)` ([build_fish_dataset.py:L1154-L1295](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L1154-L1295)).

This is robust because the median reduces the effect of outliers.

## 6) Per-frame CSV output (what columns mean)

Per-frame extraction rows contain:

- `FishID`, `FrameIndex`, `Timestamp (s)`
- `FPS_Top`, `FPS_Front`
- `Length (cm)`, `Width (cm)`, `Height (cm)`, `Area (cm²)`, `Perimeter (cm)`
- `TopMaskPixels`, `FrontMaskPixels`
- `BlurTop`, `BlurFront`
- `Score`

Example file: [fish_frame_measurements_single_corrected.csv](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/fish_frame_measurements_single_corrected.csv)

## 7) Truth extraction (single images on grid paper)

Truth values (Length_truth/Width_truth/Area_truth/Perimeter_truth) come from single still images taken on a grid, not from the tank videos.

Pipeline summary (truth images):

1) Segment fish on a grid using background removal + chroma threshold
2) Estimate grid period in pixels using Sobel gradients + 1D autocorrelation
3) Convert pixels to cm using either:
   - `--truth-px-per-cm` override, or
   - `--truth-grid-cm` (period_px / grid_cm), or
   - an automatic chooser that tries common grid sizes and picks plausible fish sizes
4) Compute truth Length/Width/Area/Perimeter from convex hull geometry

Implementations:

- Grid period: `estimate_grid_period_px(...)` ([build_fish_dataset.py:L773-L792](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L773-L792))
- Fish segmentation on grid: `segment_fish_on_grid(...)` ([build_fish_dataset.py:L794-L842](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L794-L842))
- Choosing px/cm: `choose_px_per_cm(...)` ([build_fish_dataset.py:L845-L869](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L845-L869))
- Truth extraction wrapper: `truth_from_single_image(...)` ([build_fish_dataset.py:L899-L946](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L899-L946))

Output file: [truth_values.csv](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/truth_values.csv)

## 8) Truth-aligned training dataset (single fish)

Boss feedback: “length is close but width/area/perimeter are far” means we must select and correct frames based on multiple metrics, not length alone.

Implementation: `select_training_frames_by_truth(...)` ([build_fish_dataset.py:L267-L563](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L267-L563)).

### 8.1 Absolute error and MAE

For a metric `M`:

```text
AE_i(M)  = | M_i - M_truth(FishID_i) |
MAE(M)   = mean over i of AE_i(M)
```

### 8.2 Per-fish scaling (systematic bias correction)

Optional length-based scaling (`--train-apply-per-fish-scale`):

- For each fish, compute a length scale:
  - `scale = truth_length / median(measured_length)` (with outlier bounding and clipping)
- Apply:
  - Length, Width, Height, Perimeter → multiplied by `scale`
  - Area → multiplied by `scale²`

Implementation: [build_fish_dataset.py:L325-L349](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L325-L349).

Optional per-metric scaling (`--train-apply-per-fish-metric-scales`):

- For each fish, compute separate scales:
  - `_PerFishScale_Width` from truth width vs median measured width
  - `_PerFishScale_Perimeter` from truth perimeter vs median measured perimeter
  - `_PerFishScale_Area` from truth area vs median measured area
- Apply:
  - Width → multiplied by `_PerFishScale_Width`
  - Perimeter → multiplied by `_PerFishScale_Perimeter`
  - Area → multiplied by `_PerFishScale_Area`

Implementation: [build_fish_dataset.py:L350-L397](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L350-L397).

### 8.3 Composite error ranking (multi-metric)

Optional ranking (`--train-rank-by-composite-error`):

For each frame, compute relative error for each available metric, then average:

```text
RelErr_Length    = |L - L_truth| / max(|L_truth|, 1e-6)
RelErr_Width     = |W - W_truth| / max(|W_truth|, 1e-6)
RelErr_Perimeter = |P - P_truth| / max(|P_truth|, 1e-6)
RelErr_Area      = |A - A_truth| / max(|A_truth|, 1e-6)

_CompositeRelError = mean(RelErr_* over available metrics)
```

Implementation: [build_fish_dataset.py:L398-L419](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L398-L419).

This directly targets the boss concern: frames are selected to be good across Width/Area/Perimeter too, not just Length.

### 8.4 Selection modes (balanced rows per fish)

Goal: select exactly `frames_per_fish` rows per single fish.

Selection modes used in output:

- `within`: frames with `_AbsError_Length (cm) <= train_ae_max`
- `fallback`: used if within is empty and `train_fallback_ae_max > 0`
- `fill`: extra rows (if enabled) pulled from a relaxed pool up to quota
- `repeat`: if still short, repeat the single best frame (by composite error when enabled, otherwise by length error)

Implementation details:

- Candidate choice and optional time binning: [build_fish_dataset.py:L420-L481](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L420-L481)
- Fill-to-quota: [build_fish_dataset.py:L485-L514](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L485-L514)
- Repeat-best: [build_fish_dataset.py:L515-L530](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L515-L530)

Output example: [fish_frames.csv](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/fish_frames.csv)

## 9) Group (combination) training dataset

Groups have no truth values, so we cannot compute MAE for groups unless a group-truth table is provided.

We still produce a balanced group dataset by selecting frames using:

- Quality (`Score`)
- Coverage over time (`FrameIndex` binning with `qcut`)

Implementation: `select_training_frames_for_groups(...)` ([build_fish_dataset.py:L566-L666](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L566-L666)).

Output example: [fish_frame_measurements_groups_training_50.csv](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/fish_frame_measurements_groups_training_50.csv)

## 10) Final datasets produced (what to deliver)

Main deliverables:

- [fish_frames.csv](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/fish_frames.csv)
  - Single fish training dataset (truth-aligned; includes truth columns and selection metadata)
- [fish_frame_measurements_groups_training_50.csv](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/fish_frame_measurements_groups_training_50.csv)
  - Group training dataset (no truth columns filled; selection based on score + time coverage)
- [fish_frames_training_combined_single_plus_groups.csv](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/fish_frames_training_combined_single_plus_groups.csv)
  - Combined training dataset (single + group rows under the same schema; group truth fields empty)
- [truth_values.csv](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/truth_values.csv)
  - Truth table used for truth alignment and MAE calculations (single fish only)
- [fish_measurements.csv](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/fish_measurements.csv)
  - Aggregate measurements (one row per folder; median of best frames)

## 11) Default parameter values (important for reproducibility)

Per-frame extraction defaults (CLI):

- `--per-frame-stride 1`
- `--per-frame-min-blur 10.0`
- `--per-frame-min-area-top 300`
- `--per-frame-min-area-front 250`
- `--per-frame-min-aspect 1.05`
- `--per-frame-border-margin 8`
- `--per-frame-diff-thresh-top 12`
- `--per-frame-diff-thresh-front 10`
- `--assume-fps 60.0` (used only if FPS is missing in video metadata)

Argument definitions: [build_fish_dataset.py:L1430-L1488](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L1430-L1488).

Aggregate (one-row-per-folder) internal defaults:

- `diff_thresh_top=12`, `diff_thresh_front=10`
- `min_area_top=500`, `min_area_front=400`
- `border_margin=8`
- `min_blur=20.0`
- `min_aspect=1.1`

Config setup: [build_fish_dataset.py:L1619-L1629](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L1619-L1629).

## 12) Debugging / visual verification

To visually verify that the mask matches fish (not tapes), the script can write:

- Background images: `bg_top.jpg`, `bg_front.jpg`
- Exclusion masks: `exclude_top.png`, `exclude_front.png`
- Overlay previews: stacked top+front frames with mask overlays + measured values + score

Overlay creation: `save_debug_preview(...)` ([build_fish_dataset.py:L1075-L1115](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L1075-L1115)).

## 13) Proof artifacts (saved frames, masks, annotations)

For publication-quality reproducibility, we also export “proof artifacts” from the final per-frame dataset so each CSV row can be traced back to the exact video frames used.

### 13.1 Proof extraction inputs

- Final per-frame dataset: [fish_frames_200.csv](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/fish_frames_200.csv)
  - 3000 rows total (rows may repeat the same `FishID + FrameIndex`)

### 13.2 Outputs produced

Generated by: [extract_proof_frames.py](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/extract_proof_frames.py)

Output root folder:

- [proof_frames_fish_frames_200](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/proof_frames_fish_frames_200)

Inside the folder:

- Per fish subfolders (example): [fish01](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/proof_frames_fish_frames_200/fish01)
  - `*_top.jpg`, `*_front.jpg` (raw extracted frames)
  - `*_top_mask.png`, `*_front_mask.png` (binary masks)
- Manifests:
  - [manifest_saved_frames.csv](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/proof_frames_fish_frames_200/manifest_saved_frames.csv)
    - 1 row per unique `(FishID, FrameIndex)` pair
  - [manifest_rows_all.csv](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/proof_frames_fish_frames_200/manifest_rows_all.csv)
    - 1 row per original CSV row index (all 3000 rows)
- Annotated visualizations:
  - [annotated_all](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/proof_frames_fish_frames_200/annotated_all)
    - annotated panels for unique `(FishID, FrameIndex)` pairs
  - [annotated_rows_all](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/proof_frames_fish_frames_200/annotated_rows_all)
    - annotated panels for all 3000 rows (filename includes the CSV row index)

### 13.3 Mask generation details for proof export

For proof export, masks are regenerated using the same core segmentation used in the pipeline:

- Background modeling + tape exclusion + motion segmentation:
  - [build_background](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L982-L1001)
  - [static_exclude_mask](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L1004-L1025)
  - [motion_mask](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/build_fish_dataset.py#L1028-L1047)
- Component filtering for cleaner masks:
  - keep only large components (`combine_large_components`)
  - then keep only the single largest component for a one-blob fish mask
  - implemented in [extract_proof_frames.py](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/extract_proof_frames.py)
