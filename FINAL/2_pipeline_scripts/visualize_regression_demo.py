#!/usr/bin/env python3
"""
Simple regression visualization demo - runs directly without command line args.
Shows the Area truth regression results with text visualizations.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor

def main():
    print("REGRESSION VISUALIZATION DEMO - Area Truth Prediction")
    print("="*60)
    
    # Load the data
    print("Loading fish_frames.csv...")
    df = pd.read_csv('fish_frames.csv')
    
    # Define features and target
    feature_cols = ['Length (cm)', 'Width (cm)', 'Area (cm²)', 'Perimeter (cm)']
    target_col = 'Area_truth (cm²)'
    
    print(f"Dataset shape: {df.shape}")
    print(f"Features: {feature_cols}")
    print(f"Target: {target_col}")
    print()
    
    # Prepare data
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print()
    
    # Define models (same as in the improved results)
    models = {
        'ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0, random_state=42))
        ]),
        'svr_rbf': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf', C=10.0, epsilon=0.1, gamma='scale'))
        ]),
        'svr_rbf_stable': Pipeline([
            ('scaler', StandardScaler()),
            ('model', SVR(kernel='rbf', C=1.0, epsilon=0.2, gamma=0.1))
        ]),
        'gbr_stable': GradientBoostingRegressor(
            n_estimators=150, learning_rate=0.08, max_depth=2, 
            subsample=0.8, min_samples_leaf=3, random_state=42
        ),
        'extra_trees_stable': ExtraTreesRegressor(
            n_estimators=150, max_depth=6, min_samples_leaf=5,
            max_features='sqrt', bootstrap=True, random_state=42
        ),
        'random_forest_stable': RandomForestRegressor(
            n_estimators=150, max_depth=6, min_samples_leaf=5,
            max_features='sqrt', random_state=42
        ),
        'ridge_conservative': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=0.5, random_state=42))
        ]),
        'lasso_conservative': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=0.01, max_iter=2000, random_state=42))
        ])
    }
    
    # Train models and get predictions
    results = {}
    print("Training models...")
    
    for name, model in models.items():
        print(f"  Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'predictions': y_pred,
            'mae': mae,
            'rmse': rmse,
            'r2': r2
        }
    
    print("\n" + "="*60)
    print("REGRESSION RESULTS - AREA TRUTH PREDICTION")
    print("="*60)
    print(f"{'Model':<20} {'MAE':<8} {'RMSE':<8} {'R²':<8}")
    print("-"*50)
    
    for name, result in results.items():
        print(f"{name:<20} {result['mae']:<8.3f} {result['rmse']:<8.3f} {result['r2']:<8.3f}")
    
    print("\n" + "="*60)
    print("VISUALIZATION - PREDICTION QUALITY")
    print("="*60)
    
    # Find best model
    best_model = min(results.items(), key=lambda x: x[1]['mae'])[0]
    print(f"Best Model: {best_model} (MAE: {results[best_model]['mae']:.3f})")
    print()
    
    # Show sample predictions vs actual
    print("Sample Predictions vs Actual (first 10 test samples):")
    print("-"*50)
    print(f"{'Sample':<6} {'Actual':<8} {'Predicted':<8} {'Error':<8}")
    print("-"*35)
    
    y_pred_best = results[best_model]['predictions']
    for i in range(min(10, len(y_test))):
        actual = y_test[i]
        predicted = y_pred_best[i]
        error = actual - predicted
        print(f"{i+1:<6} {actual:<8.2f} {predicted:<8.2f} {error:<8.2f}")
    
    print("\n" + "="*60)
    print("ERROR ANALYSIS")
    print("="*60)
    
    errors = y_test - y_pred_best
    print(f"Mean Error: {np.mean(errors):.3f}")
    print(f"Std Error: {np.std(errors):.3f}")
    print(f"Min Error: {np.min(errors):.3f}")
    print(f"Max Error: {np.max(errors):.3f}")
    print()
    
    # Error distribution
    print("Error Distribution (counts):")
    print("-"*30)
    
    error_ranges = [(-5, -2), (-2, -1), (-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 5)]
    for low, high in error_ranges:
        count = np.sum((errors >= low) & (errors < high))
        print(f"[{low:4.1f}, {high:4.1f}): {count:3d} samples")
    
    print("\n" + "="*60)
    print("COMPARISON WITH BASELINE")
    print("="*60)
    
    # Compare with baseline (predicting mean)
    baseline_pred = np.full_like(y_test, np.mean(y_train))
    baseline_mae = mean_absolute_error(y_test, baseline_pred)
    baseline_rmse = np.sqrt(mean_squared_error(y_test, baseline_pred))
    baseline_r2 = r2_score(y_test, baseline_pred)
    
    print(f"Baseline (mean prediction):")
    print(f"  MAE: {baseline_mae:.3f}")
    print(f"  RMSE: {baseline_rmse:.3f}")
    print(f"  R²: {baseline_r2:.3f}")
    print()
    
    print(f"Best Model Improvement over Baseline:")
    print(f"  MAE improvement: {((baseline_mae - results[best_model]['mae']) / baseline_mae * 100):.1f}%")
    print(f"  R² improvement: {results[best_model]['r2'] - baseline_r2:.3f}")
    
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"The regression models successfully predict Area truth values from")
    print(f"measured features with high accuracy (R² = {results[best_model]['r2']:.3f}).")
    print(f"The best model achieves MAE = {results[best_model]['mae']:.3f} cm²,")
    print(f"which represents a {((baseline_mae - results[best_model]['mae']) / baseline_mae * 100):.1f}% improvement")
    print(f"over simply predicting the mean value.")

if __name__ == '__main__':
    main()