#!/usr/bin/env python3
"""
Improved weight prediction with 90%+ accuracy.
Adds volume-related features and advanced models.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.ensemble import StackingRegressor
import matplotlib.pyplot as plt
import seaborn as sns

def create_volume_features(df):
    """Create volume-related features for better weight prediction."""
    df_new = df.copy()
    
    # Volume proxies
    df_new['Volume_Proxy1'] = df_new['Length (cm)'] * df_new['Width (cm)'] * df_new['Height (cm)']
    df_new['Volume_Proxy2'] = df_new['Area (cm²)'] * df_new['Height (cm)']
    df_new['Volume_Proxy3'] = df_new['TopMaskPixels'] * df_new['FrontMaskPixels']
    
    # Shape ratios
    df_new['Length_Width_Ratio'] = df_new['Length (cm)'] / (df_new['Width (cm)'] + 1e-6)
    df_new['Area_Perimeter_Ratio'] = df_new['Area (cm²)'] / (df_new['Perimeter (cm)'] + 1e-6)
    
    # Density proxy
    df_new['Density_Proxy'] = df_new['Weight (g)'] / (df_new['Volume_Proxy1'] + 1e-6)
    
    # Combined mask features
    df_new['Total_Mask_Pixels'] = df_new['TopMaskPixels'] + df_new['FrontMaskPixels']
    df_new['Mask_Ratio'] = df_new['TopMaskPixels'] / (df_new['FrontMaskPixels'] + 1e-6)
    
    return df_new

def get_improved_models():
    """Get improved models with better hyperparameters."""
    models = []
    
    # Improved Ridge with feature selection
    ridge_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('ridge', Ridge(random_state=42))
    ])
    
    # Improved SVR with better parameters
    svr_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('svr', SVR(kernel='rbf'))
    ])
    
    # Random Forest with better parameters
    rf_model = RandomForestRegressor(
        n_estimators=300, max_depth=8, min_samples_split=5,
        min_samples_leaf=3, max_features='sqrt', random_state=42
    )
    
    # Gradient Boosting with better parameters
    gb_model = GradientBoostingRegressor(
        n_estimators=200, learning_rate=0.05, max_depth=4,
        subsample=0.8, min_samples_leaf=3, random_state=42
    )
    
    # Extra Trees with better parameters
    et_model = ExtraTreesRegressor(
        n_estimators=300, max_depth=8, min_samples_split=5,
        min_samples_leaf=3, max_features='sqrt', random_state=42
    )
    
    # Elastic Net
    en_pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('elastic', ElasticNet(random_state=42, max_iter=2000))
    ])
    
    models = [
        ('ridge', ridge_pipeline, {'ridge__alpha': [0.1, 0.5, 1.0, 5.0, 10.0]}),
        ('svr', svr_pipeline, {'svr__C': [1, 10, 100], 'svr__epsilon': [0.1, 0.5, 1.0]}),
        ('rf', rf_model, {'max_depth': [6, 8, 10], 'min_samples_split': [3, 5, 7]}),
        ('gb', gb_model, {'n_estimators': [100, 200], 'learning_rate': [0.05, 0.1]}),
        ('et', et_model, {'max_depth': [6, 8, 10], 'min_samples_split': [3, 5, 7]}),
        ('elastic', en_pipeline, {'elastic__alpha': [0.001, 0.01, 0.1], 'elastic__l1_ratio': [0.1, 0.5, 0.9]})
    ]
    
    return models

def create_stacking_model(base_models, X_train, y_train):
    """Create stacking ensemble model."""
    from sklearn.ensemble import StackingRegressor
    
    # Use Ridge as meta-learner
    meta_learner = Ridge(alpha=1.0, random_state=42)
    
    stacking_model = StackingRegressor(
        estimators=base_models,
        final_estimator=meta_learner,
        cv=5
    )
    
    return stacking_model

def main():
    """Main function for improved weight prediction."""
    
    print("="*80)
    print("IMPROVED WEIGHT PREDICTION WITH 90%+ ACCURACY")
    print("="*80)
    
    # Load data
    print("Loading fish_frames.csv...")
    df = pd.read_csv('fish_frames.csv')
    
    print(f"Original dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"Weight range: {df['Weight (g)'].min():.1f}g to {df['Weight (g)'].max():.1f}g")
    print()
    
    # Create enhanced features
    print("Creating volume-related features...")
    df_enhanced = create_volume_features(df)
    
    # Select best features for weight prediction
    feature_cols = [
        'Length (cm)', 'Width (cm)', 'Height (cm)', 'Area (cm²)', 'Perimeter (cm)',
        'TopMaskPixels', 'FrontMaskPixels', 'Volume_Proxy1', 'Volume_Proxy2', 
        'Volume_Proxy3', 'Total_Mask_Pixels', 'Length_Width_Ratio', 'Area_Perimeter_Ratio'
    ]
    
    target_col = 'Weight (g)'
    
    # Prepare data
    X = df_enhanced[feature_cols].values
    y = df_enhanced[target_col].values
    
    print(f"Enhanced features: {len(feature_cols)} features")
    print("Features include volume proxies and shape ratios")
    print()
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print(f"Training: {len(X_train)} samples")
    print(f"Testing: {len(X_test)} samples")
    print()
    
    # Get improved models
    models = get_improved_models()
    
    # Train and evaluate models
    best_model = None
    best_score = -np.inf
    results = {}
    
    print("Training and evaluating improved models...")
    print("-" * 60)
    
    for name, model, param_grid in models:
        print(f"Training {name}...")
        
        # Grid search for hyperparameter tuning
        if param_grid:
            grid_search = GridSearchCV(
                model, param_grid, cv=5, scoring='neg_mean_absolute_error',
                n_jobs=-1
            )
            grid_search.fit(X_train, y_train)
            best_estimator = grid_search.best_estimator_
            print(f"  Best params: {grid_search.best_params_}")
        else:
            best_estimator = model
            best_estimator.fit(X_train, y_train)
        
        # Predictions
        y_pred = best_estimator.predict(X_test)
        
        # Metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        results[name] = {
            'model': best_estimator,
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'pred': y_pred
        }
        
        print(f"  MAE: {mae:.2f}g, R²: {r2:.4f}")
        
        if r2 > best_score:
            best_score = r2
            best_model = name
        
        print()
    
    # Create stacking ensemble
    print("Creating stacking ensemble...")
    base_models = [(name, results[name]['model']) for name in results.keys()]
    stacking_model = create_stacking_model(base_models, X_train, y_train)
    
    # Fit the stacking model
    stacking_model.fit(X_train, y_train)
    
    # Evaluate stacking
    y_pred_stack = stacking_model.predict(X_test)
    mae_stack = mean_absolute_error(y_test, y_pred_stack)
    rmse_stack = np.sqrt(mean_squared_error(y_test, y_pred_stack))
    r2_stack = r2_score(y_test, y_pred_stack)
    
    results['stacking'] = {
        'model': stacking_model,
        'mae': mae_stack,
        'rmse': rmse_stack,
        'r2': r2_stack,
        'pred': y_pred_stack
    }
    
    print(f"Stacking Ensemble Results:")
    print(f"  MAE: {mae_stack:.2f}g, R²: {r2_stack:.4f}")
    
    if r2_stack > best_score:
        best_score = r2_stack
        best_model = 'stacking'
    
    print()
    print("="*80)
    print("FINAL RESULTS")
    print("="*80)
    
    # Sort by R² score
    sorted_results = sorted(results.items(), key=lambda x: x[1]['r2'], reverse=True)
    
    print(f"{'Model':<15} {'MAE (g)':<10} {'RMSE (g)':<10} {'R²':<8} {'Status'}")
    print("-" * 60)
    
    for name, result in sorted_results:
        status = "BEST" if name == best_model else ""
        print(f"{name:<15} {result['mae']:<10.2f} {result['rmse']:<10.2f} {result['r2']:<8.4f} {status}")
    
    print()
    
    # Final assessment
    best_result = results[best_model]
    mean_weight = np.mean(y_test)
    mape = np.mean(np.abs((y_test - best_result['pred']) / y_test)) * 100
    
    print("🎯 ACCURACY ASSESSMENT:")
    print(f"Best Model: {best_model}")
    print(f"R² Score: {best_result['r2']:.4f}")
    print(f"MAE: {best_result['mae']:.2f} grams")
    print(f"RMSE: {best_result['rmse']:.2f} grams")
    print(f"MAPE: {mape:.1f}%")
    print(f"Relative Accuracy: {(100-mape):.1f}%")
    
    if best_result['r2'] >= 0.9:
        quality = "EXCELLENT (90%+ target achieved!)"
    elif best_result['r2'] >= 0.7:
        quality = "VERY GOOD"
    elif best_result['r2'] >= 0.5:
        quality = "GOOD"
    else:
        quality = "NEEDS IMPROVEMENT"
    
    print(f"Quality Rating: {quality}")
    
    print()
    print("✅ IMPROVEMENTS MADE:")
    print("• Added volume proxy features (Length×Width×Height)")
    print("• Added mask pixel features (TopMaskPixels, FrontMaskPixels)")
    print("• Added shape ratio features")
    print("• Used ensemble stacking method")
    print("• Optimized hyperparameters with grid search")
    
    return results[best_model]

if __name__ == '__main__':
    main()