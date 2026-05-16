import shutil
import os

ROOT = r"C:\Users\shain\Downloads\FISH DATASET\FISH DATASET"
FINAL = r"C:\Users\shain\Downloads\FISH DATASET\FINAL"

# --- FOLDERS TO COPY ---
folders_to_copy = {
    # Mask comparison work
    "final_ai_masks": "5_mask_archives/final_ai_masks",
    "final_ai_masks_full_sam": "5_mask_archives/final_ai_masks_full_sam",
    "final_classic_masks": "5_mask_archives/final_classic_masks",
    "SAM_Comparisons": "5_mask_archives/SAM_Comparisons",
    "mask_comparison_results": "5_mask_archives/mask_comparison_results",
    "FULL SAM detect": "5_mask_archives/FULL_SAM_detect",
    
    # Truth debug/verification
    "truth_debug_previews_auto": "7_calibration/truth_debug_previews_auto",
    "truth_debug_previews_grid_0_5cm": "7_calibration/truth_debug_previews_grid_0_5cm",
    "truth_debug_previews_grid_1cm": "7_calibration/truth_debug_previews_grid_1cm",
    
    # Pipeline proof images
    "pipeline_stage_proof": "8_pipeline_proof",
}

for src_name, dst_name in folders_to_copy.items():
    src = os.path.join(ROOT, src_name)
    dst = os.path.join(FINAL, dst_name)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copytree(src, dst)
        count = len([f for f in os.listdir(dst) if os.path.isfile(os.path.join(dst, f))])
        print(f"  Copied folder: {src_name}/ -> {dst_name}/ ({count} files)")
    elif os.path.exists(dst):
        print(f"  Already exists: {dst_name}/")

# --- MISSING .py SCRIPTS ---
missing_scripts = [
    "build_fish_dataset.py",
    "generate_classic_cv_masks.py",
    "compare_mask_methods.py",
    "sam_benchmark_compare.py",
    "run_sam_test_one_frame.py",
    "run_sam_clean_dataset.py",
    "run_enhanced_extraction.py",
    "test_fastsam.py",
    "create_sam_dataset.py",
    "extract_proof_frames.py",
    "_compare_truth.py",
    # Regression scripts
    "train_weight_regression.py",
    "evaluate_regression.py",
    "regression_model_analysis.py",
    "regression_visualization_summary.py",
    "create_regression_plots.py",
    "visualize_regression_results.py",
    "visualize_regression_simple.py",
    "visualize_regression_demo.py",
]

script_dst = os.path.join(FINAL, "2_pipeline_scripts")
for f in missing_scripts:
    src = os.path.join(ROOT, f)
    dst = os.path.join(script_dst, f)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"  Copied script: {f}")

# --- MISSING TRUTH/CALIBRATION CSVs ---
missing_csvs = [
    "truth_values.csv",
    "truth_values_auto.csv",
    "truth_values_FIXED.csv",
    "truth_values_grid_0_5cm.csv",
    "truth_values_grid_1cm.csv",
    "truth_values_recalibrated.csv",
    "truth_values_recalibrated_v2.csv",
    "truth_values_reextracted.csv",
    "truth_values_compare_full.csv",
    "fish_truth_measurements.csv",
    "fish_measurements_publication_ready.csv",
]

truth_dst = os.path.join(FINAL, "1_ground_truth")
for f in missing_csvs:
    src = os.path.join(ROOT, f)
    dst = os.path.join(truth_dst, f)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"  Copied truth: {f}")

# --- MISSING REPORTS ---
reports = [
    "REGRESSION_ANALYSIS_REPORT.md",
    "DATASET_CREATION_REPORT.md",
    "PUBLICATION_DATASET_SUMMARY.md",
    "PUBLICATION_NOTES.md",
    "RESULTS_REPORT.md",
    "VIDEO_PIPELINE_DETAILED_REPORT.md",
    "publication_methodology_report.md",
]
report_dst = os.path.join(FINAL, "6_reports")
for f in reports:
    src = os.path.join(ROOT, f)
    dst = os.path.join(report_dst, f)
    if os.path.exists(src) and not os.path.exists(dst):
        shutil.copy2(src, dst)
        print(f"  Copied report: {f}")

# --- FINAL COUNT ---
print("\n" + "=" * 60)
print("FINAL FOLDER SUMMARY:")
print("=" * 60)
for folder in sorted(os.listdir(FINAL)):
    folder_path = os.path.join(FINAL, folder)
    if os.path.isdir(folder_path):
        count = sum(1 for _, _, files in os.walk(folder_path) for f in files)
        print(f"  {folder}/ ({count} files)")
