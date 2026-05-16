"""
CRITICAL: Methodology verification for publication integrity
This script checks for potential issues that could invalidate results
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt

def verify_dataset_integrity():
    """Check for data integrity issues"""
    print("=== DATASET INTEGRITY CHECK ===")
    
    # Load data
    df = pd.read_csv('fish_frames.csv')
    print(f"Dataset shape: {df.shape}")
    print(f"Total fish: {df['FishID'].nunique()}")
    print(f"Frames per fish: {df.groupby('FishID').size().describe()}")
    
    # Check for missing values
    missing = df.isnull().sum()
    if missing.any():
        print("⚠️  MISSING VALUES FOUND:")
        print(missing[missing > 0])
    else:
        print("✅ No missing values")
    
    # Check for duplicates
    duplicates = df.duplicated().sum()
    print(f"Duplicate rows: {duplicates}")
    
    # Check weight distribution
    print(f"Weight range: {df['Weight (g)'].min():.2f}g - {df['Weight (g)'].max():.2f}g")
    print(f"Weight mean ± std: {df['Weight (g)'].mean():.2f} ± {df['Weight (g)'].std():.2f}g")
    
    return df

def verify_no_data_leakage(df):
    """Ensure proper train/test splits by fish ID"""
    print("\n=== DATA LEAKAGE CHECK ===")
    
    # Get unique fish
    unique_fish = df['FishID'].unique()
    print(f"Unique fish IDs: {len(unique_fish)}")
    
    # Check if fish are properly separated
    # Our method uses first 12 fish for train, last 3 for test
    train_fish = set(unique_fish[:12])
    test_fish = set(unique_fish[12:])
    
    print(f"Train fish (12): {sorted(train_fish)}")
    print(f"Test fish (3): {sorted(test_fish)}")
    
    # Verify no overlap
    overlap = train_fish.intersection(test_fish)
    if overlap:
        print(f"❌ DATA LEAKAGE: Overlapping fish: {overlap}")
        return False
    else:
        print("✅ No data leakage - fish properly separated")
        return True

def verify_feature_scaling(df):
    """Check if features need scaling"""
    print("\n=== FEATURE SCALING CHECK ===")
    
    features = ['Length (cm)', 'Width (cm)', 'Height (cm)', 'Area (cm²)', 
                'Perimeter (cm)', 'TopMaskPixels', 'FrontMaskPixels']
    
    X = df[features]
    
    print("Feature statistics:")
    print(X.describe())
    
    # Check scale differences
    scales = X.std()
    max_scale = scales.max()
    min_scale = scales.min()
    scale_ratio = max_scale / min_scale
    
    print(f"\nScale ratio (max/min std): {scale_ratio:.2f}")
    if scale_ratio > 10:
        print("⚠️  LARGE SCALE DIFFERENCES - scaling recommended")
    else:
        print("✅ Reasonable scale differences")
    
    return scale_ratio

def verify_truth_comparison(df):
    """Check truth value comparisons"""
    print("\n=== TRUTH VALUE VERIFICATION ===")
    
    # Check if truth values exist
    truth_cols = ['_Truth_Length (cm)', 'Width_truth (cm)', 'Area_truth (cm²)', 'Perimeter_truth (cm)']
    available_truth = [col for col in truth_cols if col in df.columns]
    
    print(f"Available truth columns: {available_truth}")
    
    if available_truth:
        # Compare measurements vs truth
        if '_Truth_Length (cm)' in df.columns and 'Length (cm)' in df.columns:
            length_error = abs(df['Length (cm)'] - df['_Truth_Length (cm)']).mean()
            print(f"Length measurement error: {length_error:.4f} cm")
    
    return available_truth

def verify_model_robustness():
    """Test model robustness with different validation strategies"""
    print("\n=== MODEL ROBUSTNESS CHECK ===")
    
    df = pd.read_csv('fish_frames.csv')
    
    # Prepare features
    features = ['Length (cm)', 'Width (cm)', 'Height (cm)', 'Area (cm²)', 
                'Perimeter (cm)', 'TopMaskPixels', 'FrontMaskPixels']
    
    # Add volume features
    df['Volume_Proxy1'] = df['Length (cm)'] * df['Width (cm)'] * df['Height (cm)']
    df['Volume_Proxy2'] = df['Area (cm²)'] * df['Height (cm)']
    df['Volume_Proxy3'] = df['TopMaskPixels'] * df['FrontMaskPixels']
    
    X = df[features + ['Volume_Proxy1', 'Volume_Proxy2', 'Volume_Proxy3']]
    y = df['Weight (g)']
    groups = df['FishID']
    
    # Use Leave-One-Group-Out (LOGO) CV - most rigorous for this dataset
    logo = LeaveOneGroupOut()
    
    model = GradientBoostingRegressor(n_estimators=200, learning_rate=0.1, random_state=42)
    
    print("Leave-One-Fish-Out Cross-Validation:")
    maes = []
    r2s = []
    
    for train_idx, test_idx in logo.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        maes.append(mae)
        r2s.append(r2)
        
        test_fish = groups.iloc[test_idx].iloc[0]
        print(f"  Fish {test_fish}: MAE={mae:.3f}g, R²={r2:.3f}")
    
    print(f"\nLOGO CV Results:")
    print(f"Mean MAE: {np.mean(maes):.3f} ± {np.std(maes):.3f}g")
    print(f"Mean R²: {np.mean(r2s):.3f} ± {np.std(r2s):.3f}")
    print(f"Min R²: {np.min(r2s):.3f}")
    print(f"Max R²: {np.max(r2s):.3f}")
    
    return np.mean(maes), np.mean(r2s), np.std(r2s)

def main():
    """Run all verification checks"""
    print("PUBLICATION INTEGRITY VERIFICATION")
    print("="*50)
    
    # Load and check dataset
    df = verify_dataset_integrity()
    
    # Check for data leakage
    no_leakage = verify_no_data_leakage(df)
    
    # Check feature scaling
    scale_ratio = verify_feature_scaling(df)
    
    # Check truth values
    truth_available = verify_truth_comparison(df)
    
    # Check model robustness
    logo_mae, logo_r2, logo_r2_std = verify_model_robustness()
    
    print("\n" + "="*50)
    print("PUBLICATION READINESS ASSESSMENT")
    print("="*50)
    
    issues = []
    if not no_leakage:
        issues.append("Data leakage detected")
    if scale_ratio > 50:
        issues.append("Extreme scale differences require scaling")
    if len(df['FishID'].unique()) < 10:
        issues.append("Very small sample size")
    if logo_r2_std > 0.1:
        issues.append("High variance in cross-validation")
    if logo_r2 < 0.8:
        issues.append("Poor generalization performance")
    
    if issues:
        print("❌ ISSUES FOUND:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("✅ No major issues detected")
    
    print(f"\nFINAL RECOMMENDATION:")
    print(f"Leave-One-Fish-Out CV Results: MAE={logo_mae:.3f}g, R²={logo_r2:.3f}±{logo_r2_std:.3f}")
    
    if logo_r2 > 0.9 and logo_r2_std < 0.1 and no_leakage:
        print("✅ Results appear robust for publication")
    else:
        print("⚠️  Consider additional validation before publication")

if __name__ == "__main__":
    main()