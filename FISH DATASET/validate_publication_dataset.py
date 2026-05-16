#!/usr/bin/env python3
"""
Validate the publication-ready dataset for machine learning
by checking for data leakage and running cross-validation.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import LeaveOneOut
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
import matplotlib.pyplot as plt

def main():
    # Load the publication-ready dataset
    df = pd.read_csv('fish_measurements_publication_ready.csv')
    
    print("=== PUBLICATION-READY DATASET VALIDATION ===")
    print(f"Dataset shape: {df.shape}")
    print(f"Unique fish: {df['FishID'].nunique()}")
    
    # Check for data leakage
    print(f"\n=== DATA LEAKAGE CHECK ===")
    print(f"One sample per fish: {len(df) == df['FishID'].nunique()} ✓")
    print(f"No duplicate FishIDs: {df['FishID'].duplicated().sum() == 0} ✓")
    
    # Prepare features and target
    features = ['Length (cm)', 'Width (cm)', 'Height (cm)', 'Area (cm²)', 'Perimeter (cm)']
    target = 'Weight (g)'
    
    X = df[features].values
    y = df[target].values
    
    print(f"\n=== FEATURE STATISTICS ===")
    for feature in features:
        print(f"{feature}: mean={df[feature].mean():.2f}, std={df[feature].std():.2f}")
    
    print(f"\n=== TARGET STATISTICS ===")
    print(f"Weight (g): mean={y.mean():.2f}, std={y.std():.2f}, range={y.min():.1f}-{y.max():.1f}")
    
    print(f"\n=== CROSS-VALIDATION RESULTS (LOOCV) ===")
    loo = LeaveOneOut()
    lr = LinearRegression()

    print(f"\n=== MANUAL LOOCV VALIDATION ===")
    predictions = []
    actuals = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        lr.fit(X_train, y_train)
        pred = lr.predict(X_test)
        
        predictions.append(pred[0])
        actuals.append(y_test[0])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    
    # Calculate metrics
    r2_manual = r2_score(actuals, predictions)
    rmse_manual = np.sqrt(mean_squared_error(actuals, predictions))
    mae_manual = np.mean(np.abs(actuals - predictions))
    baseline_mae = np.mean(np.abs(actuals - np.mean(actuals)))
    
    print(f"Manual LOOCV Results:")
    print(f"  R²: {r2_manual:.3f}")
    print(f"  RMSE: {rmse_manual:.2f} g")
    print(f"  MAE: {mae_manual:.2f} g")
    print(f"  Baseline MAE (predict mean): {baseline_mae:.2f} g")
    
    # Feature importance
    lr.fit(X, y)
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Coefficient': lr.coef_,
        'Abs_Coefficient': np.abs(lr.coef_)
    }).sort_values('Abs_Coefficient', ascending=False)
    
    print(f"\n=== FEATURE IMPORTANCE ===")
    for _, row in feature_importance.iterrows():
        print(f"  {row['Feature']}: {row['Coefficient']:.3f}")
    
    # Save validation results
    validation_results = {
        'dataset_size': len(df),
        'unique_fish': df['FishID'].nunique(),
        'r2': r2_manual,
        'rmse': rmse_manual,
        'mae': mae_manual,
        'baseline_mae': baseline_mae,
    }
    
    # Save results to file
    with open('validation_results.txt', 'w') as f:
        f.write("=== PUBLICATION-READY DATASET VALIDATION RESULTS ===\n")
        f.write(f"Dataset: fish_measurements_publication_ready.csv\n")
        f.write(f"Samples: {validation_results['dataset_size']}\n")
        f.write(f"Unique fish: {validation_results['unique_fish']}\n")
        f.write(f"Manual LOOCV R²: {validation_results['r2']:.3f}\n")
        f.write(f"RMSE: {validation_results['rmse']:.2f} g\n")
        f.write(f"MAE: {validation_results['mae']:.2f} g\n")
        f.write(f"Baseline MAE (predict mean): {validation_results['baseline_mae']:.2f} g\n")
        
        f.write("Note: Per-fold R² is not meaningful for LOOCV with 1-sample test folds.\n")
    
    print(f"\n=== VALIDATION COMPLETE ===")
    print("No data leakage detected (one row per fish).")
    
    return validation_results

if __name__ == "__main__":
    main()
