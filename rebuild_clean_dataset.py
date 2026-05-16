"""
REBUILD CLEAN DATASET - REMOVE ALL CONTAMINATION
This script creates a legitimate, publication-ready dataset
"""

import pandas as pd
import numpy as np

print("="*80)
print("🔧 REBUILDING CLEAN DATASET - REMOVING CONTAMINATION")
print("="*80)

# Load the contaminated dataset
df_contaminated = pd.read_csv('FISH DATASET/fish_frames.csv')
print(f"Original dataset: {len(df_contaminated)} samples, {len(df_contaminated.columns)} columns")

# Define legitimate features that are NOT derived from truth values
legitimate_features = [
    'FishID',
    'FrameIndex', 
    'Timestamp (s)',
    'FPS_Top',
    'FPS_Front',
    'Length (cm)',           # Raw measurement from image
    'Width (cm)',            # Raw measurement from image  
    'Height (cm)',           # Raw measurement from image
    'Area (cm²)',            # Raw measurement from image
    'Perimeter (cm)',        # Raw measurement from image
    'BlurTop',               # Image quality metric
    'BlurFront',             # Image quality metric
    'Score',                 # Detection confidence
    '_SelectionMode'         # Frame selection method
]

# Target variable (what we want to predict)
target = 'Weight (g)'

print(f"\nLegitimate features to keep: {len(legitimate_features)}")
for feature in legitimate_features:
    if feature in df_contaminated.columns:
        print(f"  ✅ {feature}")
    else:
        print(f"  ❌ {feature} - Missing")

# Create clean dataset
df_clean = df_contaminated[legitimate_features + [target]].copy()

print(f"\nClean dataset: {len(df_clean)} samples, {len(df_clean.columns)} columns")

# Show what we removed
removed_columns = set(df_contaminated.columns) - set(df_clean.columns)
print(f"\nRemoved {len(removed_columns)} contaminated columns:")
for col in sorted(removed_columns):
    print(f"  ❌ {col}")

# Save the clean dataset
df_clean.to_csv('fish_frames_CLEAN.csv', index=False)
print(f"\n✅ Clean dataset saved to: fish_frames_CLEAN.csv")

# Show basic statistics of clean features
print(f"\n📊 CLEAN FEATURE STATISTICS:")
print("-" * 50)
for col in ['Length (cm)', 'Width (cm)', 'Height (cm)', 'Area (cm²)', 'Weight (g)']:
    if col in df_clean.columns:
        print(f"{col}:")
        print(f"  Range: {df_clean[col].min():.2f} - {df_clean[col].max():.2f}")
        print(f"  Mean: {df_clean[col].mean():.2f} ± {df_clean[col].std():.2f}")

print(f"\n🎯 PUBLICATION-READY DATASET CREATED")
print("This dataset contains only legitimate morphometric measurements")
print("No truth values, no scaling factors, no derived contamination")
print("Ready for legitimate machine learning model development")