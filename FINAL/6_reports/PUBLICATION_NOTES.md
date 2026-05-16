# Publication Notes (Fish Video → Measurements → Datasets)

## What is safe to publish

This project produces multiple CSVs for different purposes. For publication, it is important to separate:

- **Measurement benchmarking vs truth** (needs truth columns)
- **ML training to predict weight** (must not use truth columns as inputs)

## Datasets in this folder

- **Per-frame training dataset (single fish, many frames)**
  - [fish_frames.csv](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/fish_frames.csv)
  - 50 frames per fish (15 fish ⇒ 750 rows)
  - Contains measured features + truth columns + selection metadata

- **Larger per-frame dataset (single fish, many frames)**
  - [fish_frames_200.csv](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/fish_frames_200.csv)
  - 200 frames per fish (15 fish ⇒ 3000 rows)
  - Same schema as fish_frames.csv

- **Benchmark: “best-1 frame per fish” (truth-informed selection)**
  - [fish_frames_best1.csv](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/fish_frames_best1.csv)
  - Exactly 1 frame per fish selected by lowest composite error vs truth
  - Use this for measurement benchmarking / qualitative examples
  - Do not treat it as a realistic “deployment” dataset (truth leakage in selection)

- **Fish-level dataset (one row per fish, for weight regression)**
  - [fish_single_fish_level_dataset.csv](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/fish_single_fish_level_dataset.csv)
  - 15 rows total (one per FishID, median of selected frames)
  - Recommended for weight prediction reporting with very small N fish

## Checks already performed (reproducibility)

- Truth values were re-extracted from the single images and compared with the existing truth table:
  - [truth_values_reextracted.csv](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/truth_values_reextracted.csv)
  - [truth_values_compare_full.csv](file:///c:/Users/shain/Downloads/FISH%20DATASET/FISH%20DATASET/truth_values_compare_full.csv)
  - Outcome: no large differences (>0.5 cm or >5 cm²).

## How to report results correctly (publication-safe)

### A) Measurement quality (measured vs truth)

Use MAE in the physical units:

- Length MAE (cm)
- Width MAE (cm)
- Area MAE (cm²)
- Perimeter MAE (cm)

Command:

```powershell
python evaluate_regression.py --mode truth_report --csv fish_frames.csv --truth-report-per-fish
python evaluate_regression.py --mode truth_report --csv fish_frames_best1.csv --truth-report-per-fish
```

Interpretation:
- fish_frames_best1 is expected to have lower MAE for the metrics it optimizes, because selection uses truth.

### B) Weight prediction (measured → Weight (g))

Do **not** use truth columns as inputs.

Because weight is per-fish and there are only 15 fish, do not claim “750 samples”.

Preferred reporting:
- LOFO (Leave-One-Fish-Out) evaluation on one-row-per-fish dataset
- MAE in grams

Command:

```powershell
python evaluate_regression.py --csv fish_frames.csv --mode weight_fish_level_svr --cv-splits 5 --random-state 42
```

## Key limitation to state explicitly

There are only **15 unique fish**. Per-frame rows are repeated measurements of the same fish and are not independent samples. This limits generalization for weight prediction and makes results sensitive to which fish are held out.

