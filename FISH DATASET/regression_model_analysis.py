#!/usr/bin/env python3
"""
Comprehensive regression model analysis for the fish weight vs dimensions dataset.
Tests multiple regression approaches and provides detailed accuracy metrics.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import cross_val_score, LeaveOneOut, train_test_split
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

def load_data():
    """Load the publication-ready dataset."""
    df = pd.read_csv('fish_measurements_publication_ready.csv')
    print(f"Loaded dataset: {df.shape[0]} samples, {df.shape[1]} features")
    return df

def prepare_features(df):
    """Prepare feature sets for different model approaches."""
    
    # Basic morphometric features
    basic_features = ['Length (cm)', 'Width (cm)', 'Height (cm)']
    
    # All geometric features
    all_features = ['Length (cm)', 'Width (cm)', 'Height (cm)', 'Area (cm²)', 'Perimeter (cm)']
    
    # Engineered features (ratios and combinations)
    df_eng = df.copy()
    df_eng['Length_Width_Ratio'] = df_eng['Length (cm)'] / df_eng['Width (cm)']
    df_eng['Area_Perimeter_Ratio'] = df_eng['Area (cm²)'] / df_eng['Perimeter (cm)']
    df_eng['Volume_Proxy'] = df_eng['Length (cm)'] * df_eng['Width (cm)'] * df_eng['Height (cm)']
    df_eng['Surface_Area_Proxy'] = df_eng['Area (cm²)'] * 2  # Approximation
    
    engineered_features = ['Length (cm)', 'Width (cm)', 'Height (cm)', 'Length_Width_Ratio', 
                          'Area_Perimeter_Ratio', 'Volume_Proxy', 'Surface_Area_Proxy']
    
    feature_sets = {
        'basic': (basic_features, df),
        'all_geo': (all_features, df),
        'engineered': (engineered_features, df_eng)
    }
    
    return feature_sets

def evaluate_model(model, X, y, model_name, feature_set_name):
    """Comprehensive model evaluation using cross-validation."""
    
    # Leave-One-Out Cross-Validation (appropriate for small dataset)
    loo = LeaveOneOut()
    
    # Manual LOOCV for detailed metrics
    predictions = []
    actuals = []
    residuals = []
    
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Fit model
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        predictions.append(pred[0])
        actuals.append(y_test[0])
        residuals.append(y_test[0] - pred[0])
    
    predictions = np.array(predictions)
    actuals = np.array(actuals)
    residuals = np.array(residuals)
    
    # Calculate metrics
    r2 = r2_score(actuals, predictions)
    rmse = np.sqrt(mean_squared_error(actuals, predictions))
    mae = mean_absolute_error(actuals, predictions)
    mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
    
    # Relative RMSE (as percentage of mean actual value)
    mean_actual = np.mean(actuals)
    rmse_pct = (rmse / mean_actual) * 100
    
    results = {
        'model': model_name,
        'feature_set': feature_set_name,
        'r2': r2,
        'rmse': rmse,
        'mae': mae,
        'mape': mape,
        'rmse_pct': rmse_pct,
        'predictions': predictions,
        'actuals': actuals,
        'residuals': residuals,
        'mean_actual': mean_actual
    }
    
    return results

def compare_models(df, feature_sets):
    """Compare multiple regression models across different feature sets."""
    
    target = 'Weight (g)'
    all_results = []
    
    # Define models to test
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(alpha=1.0),
        'Lasso Regression': Lasso(alpha=1.0),
        'Elastic Net': ElasticNet(alpha=1.0, l1_ratio=0.5),
        'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
        'Support Vector Regression': SVR(kernel='rbf', C=1.0, gamma='scale')
    }
    
    print("\n" + "="*80)
    print("REGRESSION MODEL COMPARISON")
    print("="*80)
    
    for feature_set_name, (features, data) in feature_sets.items():
        print(f"\n--- Feature Set: {feature_set_name.upper()} ---")
        print(f"Features: {features}")
        
        X = data[features].values
        y = data[target].values
        
        # Standardize features for some models
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        for model_name, model in models.items():
            # Use scaled features for models that benefit from it
            if model_name in ['Ridge Regression', 'Lasso Regression', 'Elastic Net', 'Support Vector Regression']:
                results = evaluate_model(model, X_scaled, y, model_name, feature_set_name)
            else:
                results = evaluate_model(model, X, y, model_name, feature_set_name)
            
            all_results.append(results)
            
            print(f"{model_name:25} | R²: {results['r2']:6.3f} | RMSE: {results['rmse']:6.2f}g | MAE: {results['mae']:6.2f}g | RMSE%: {results['rmse_pct']:5.1f}%")
    
    return all_results

def find_best_model(all_results):
    """Identify the best performing model."""
    
    # Sort by R² score (descending)
    best_by_r2 = sorted(all_results, key=lambda x: x['r2'], reverse=True)
    
    print(f"\n{'='*80}")
    print("BEST MODEL SUMMARY")
    print(f"{'='*80}")
    
    best = best_by_r2[0]
    print(f"Best Model: {best['model']} with {best['feature_set']} features")
    print(f"R² Score: {best['r2']:.3f}")
    print(f"RMSE: {best['rmse']:.2f} g ({best['rmse_pct']:.1f}% of mean)")
    print(f"MAE: {best['mae']:.2f} g")
    print(f"MAPE: {best['mape']:.1f}%")
    
    return best

def detailed_analysis(best_results, df):
    """Provide detailed analysis of the best model."""
    
    print(f"\n{'='*80}")
    print("DETAILED ANALYSIS OF BEST MODEL")
    print(f"{'='*80}")
    
    # Prediction vs Actual scatter plot
    plt.figure(figsize=(10, 8))
    plt.scatter(best_results['actuals'], best_results['predictions'], alpha=0.7, s=100)
    
    # Perfect prediction line
    min_val = min(min(best_results['actuals']), min(best_results['predictions']))
    max_val = max(max(best_results['actuals']), max(best_results['predictions']))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
    
    plt.xlabel('Actual Weight (g)')
    plt.ylabel('Predicted Weight (g)')
    plt.title(f'{best_results["model"]} - Predicted vs Actual\nR² = {best_results["r2"]:.3f}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig('regression_predictions_vs_actual.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Residuals plot
    plt.figure(figsize=(10, 8))
    plt.scatter(best_results['predictions'], best_results['residuals'], alpha=0.7, s=100)
    plt.axhline(y=0, color='r', linestyle='--', lw=2)
    plt.xlabel('Predicted Weight (g)')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title(f'{best_results["model"]} - Residuals Plot')
    plt.grid(True, alpha=0.3)
    plt.savefig('regression_residuals.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Error distribution
    plt.figure(figsize=(10, 6))
    plt.hist(best_results['residuals'], bins=8, alpha=0.7, edgecolor='black')
    plt.xlabel('Residuals (g)')
    plt.ylabel('Frequency')
    plt.title(f'{best_results["model"]} - Residuals Distribution')
    plt.axvline(x=0, color='r', linestyle='--', lw=2)
    plt.grid(True, alpha=0.3)
    plt.savefig('regression_residuals_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Statistical summary
    print(f"\nPrediction Statistics:")
    print(f"  Mean Actual Weight: {best_results['mean_actual']:.2f} g")
    print(f"  Mean Prediction: {np.mean(best_results['predictions']):.2f} g")
    print(f"  Mean Residual: {np.mean(best_results['residuals']):.2f} g")
    print(f"  Std Residual: {np.std(best_results['residuals']):.2f} g")
    print(f"  Min Residual: {np.min(best_results['residuals']):.2f} g")
    print(f"  Max Residual: {np.max(best_results['residuals']):.2f} g")
    
    # Performance by weight range
    actuals = best_results['actuals']
    residuals = best_results['residuals']
    
    # Sort by actual weight
    sorted_indices = np.argsort(actuals)
    sorted_actuals = actuals[sorted_indices]
    sorted_residuals = residuals[sorted_indices]
    
    print(f"\nPerformance by Weight Range:")
    print(f"  Small fish (<20g): Mean abs error = {np.mean(np.abs(sorted_residuals[sorted_actuals < 20])):.2f} g")
    print(f"  Medium fish (20-50g): Mean abs error = {np.mean(np.abs(sorted_residuals[(sorted_actuals >= 20) & (sorted_actuals < 50)])):.2f} g")
    print(f"  Large fish (≥50g): Mean abs error = {np.mean(np.abs(sorted_residuals[sorted_actuals >= 50])):.2f} g")

def save_results(all_results, best_results, df):
    """Save detailed results to files."""
    
    # Save all model comparison results
    results_df = pd.DataFrame([
        {
            'Model': r['model'],
            'Feature_Set': r['feature_set'],
            'R2': r['r2'],
            'RMSE_g': r['rmse'],
            'MAE_g': r['mae'],
            'RMSE_Percent': r['rmse_pct'],
            'MAPE_Percent': r['mape']
        }
        for r in all_results
    ])
    
    results_df.to_csv('regression_model_comparison.csv', index=False)
    
    # Save best model predictions
    pred_df = pd.DataFrame({
        'FishID': df['FishID'],
        'Actual_Weight_g': best_results['actuals'],
        'Predicted_Weight_g': best_results['predictions'],
        'Residual_g': best_results['residuals'],
        'Abs_Error_g': np.abs(best_results['residuals']),
        'Percent_Error': (best_results['residuals'] / best_results['actuals']) * 100
    })
    
    pred_df.to_csv('regression_best_predictions.csv', index=False)
    
    print(f"\nResults saved to:")
    print(f"  - regression_model_comparison.csv")
    print(f"  - regression_best_predictions.csv")
    print(f"  - regression_predictions_vs_actual.png")
    print(f"  - regression_residuals.png")
    print(f"  - regression_residuals_distribution.png")

def main():
    """Main analysis function."""
    
    print("FISH WEIGHT REGRESSION MODEL ANALYSIS")
    print("="*80)
    
    # Load data
    df = load_data()
    
    # Prepare feature sets
    feature_sets = prepare_features(df)
    
    # Compare all models
    all_results = compare_models(df, feature_sets)
    
    # Find best model
    best_results = find_best_model(all_results)
    
    # Detailed analysis
    detailed_analysis(best_results, df)
    
    # Save results
    save_results(all_results, best_results, df)
    
    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

if __name__ == "__main__":
    main()