import shutil
import os

ROOT = r"C:\Users\shain\Downloads\FISH DATASET\FISH DATASET"
FINAL = os.path.join(ROOT, "FINAL")
os.makedirs(FINAL, exist_ok=True)

# Create subfolders
os.makedirs(os.path.join(FINAL, "1_ground_truth"), exist_ok=True)
os.makedirs(os.path.join(FINAL, "2_pipeline_scripts"), exist_ok=True)
os.makedirs(os.path.join(FINAL, "3_preprocessing_scripts"), exist_ok=True)
os.makedirs(os.path.join(FINAL, "4_datasets"), exist_ok=True)
os.makedirs(os.path.join(FINAL, "5_sample_masks"), exist_ok=True)
os.makedirs(os.path.join(FINAL, "6_reports"), exist_ok=True)

files = {
    # 1. Ground Truth
    "1_ground_truth": [
        "truth_values_corrected.csv",
        "weights.csv",
    ],
    # 2. Pipeline Scripts (the core AI pipeline)
    "2_pipeline_scripts": [
        "run_production_inference_pipeline.py",
        "generate_corrected_truth_final.py",
        "generate_full_sam_masks.py",
        "generate_hybrid_masks.py",
    ],
    # 3. Preprocessing Scripts
    "3_preprocessing_scripts": [
        "preprocess_step1.py",
        "preprocess_step2.py",
        "preprocess_step3.py",
        "preprocess_step4_clean.py",
        "audit_final_dataset.py",
        "check_calibrations.py",
    ],
    # 4. Datasets (the full progression)
    "4_datasets": [
        "fish_frames_production.csv",
        "fish_frames_production_step1_filtered.csv",
        "fish_frames_production_step2_features.csv",
        "fish_frames_production_FINAL_CLEAN.csv",
    ],
    # 6. Reports
    "6_reports": [],
}

# Copy files
copied = 0
for folder, file_list in files.items():
    for f in file_list:
        src = os.path.join(ROOT, f)
        dst = os.path.join(FINAL, folder, f)
        if os.path.exists(src):
            shutil.copy2(src, dst)
            size_mb = os.path.getsize(dst) / (1024*1024)
            print(f"  Copied: {folder}/{f} ({size_mb:.1f} MB)")
            copied += 1
        else:
            print(f"  MISSING: {f}")

# Copy the preprocessing summary report
report_src = r"C:\Users\shain\.gemini\antigravity\brain\87351874-ce08-4f2d-8799-83af16d672a0\artifacts\preprocessing_summary.md"
if os.path.exists(report_src):
    shutil.copy2(report_src, os.path.join(FINAL, "6_reports", "preprocessing_summary.md"))
    print(f"  Copied: 6_reports/preprocessing_summary.md")
    copied += 1

# Copy sample masks (5 random fish, top + front)
masks_dir = os.path.join(ROOT, "production_masks_output")
sample_fish = ["fish01", "fish3", "fish5", "fish7", "fish12"]
sample_count = 0
if os.path.exists(masks_dir):
    for f in os.listdir(masks_dir):
        for fish in sample_fish:
            if f.startswith(fish + "_frame_") and sample_count < 20:
                shutil.copy2(
                    os.path.join(masks_dir, f),
                    os.path.join(FINAL, "5_sample_masks", f)
                )
                sample_count += 1
                break
    print(f"  Copied: {sample_count} sample mask images to 5_sample_masks/")
    copied += sample_count

print(f"\nTotal files copied: {copied}")
print(f"FINAL folder: {FINAL}")

# Print folder structure
print(f"\nFINAL Folder Structure:")
for folder in sorted(os.listdir(FINAL)):
    folder_path = os.path.join(FINAL, folder)
    if os.path.isdir(folder_path):
        count = len(os.listdir(folder_path))
        print(f"  {folder}/ ({count} files)")
        for f in sorted(os.listdir(folder_path))[:5]:
            print(f"    - {f}")
        if count > 5:
            print(f"    ... and {count-5} more")
