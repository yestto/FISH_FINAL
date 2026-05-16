#!/usr/bin/env python3
"""
Proper matplotlib visualizations for regression results.
Creates actual graphs and saves them as PNG files.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor

def setup_plotting_style():
    """Set up professional plotting style."""
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 14
    plt.rcParams['axes.labelsize'] = 12
    plt.rcParams['xtick.labelsize'] = 10
    plt.rcParams['ytick.labelsize'] = 10
    plt.rcParams['legend.fontsize'] = 10

def create_scatter_plot(y_true, y_pred, model_name, target_name, save_path):
    """Create scatter plot of predicted vs actual values."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Scatter plot
    ax.scatter(y_true, y_pred, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
    
    # Perfect prediction line
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect Prediction')
    
    # Calculate metrics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    # Formatting
    ax.set_xlabel('Actual Values', fontsize=12)
    ax.set_ylabel('Predicted Values', fontsize=12)
    ax.set_title(f'{model_name} - {target_name}\nMAE={mae:.3f}, RMSE={rmse:.3f}, R²={r2:.3f}', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add metrics text box
    textstr = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nR²: {r2:.3f}'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_residual_plot(y_true, y_pred, model_name, target_name, save_path):
    """Create residual plot."""
    fig, ax = plt.subplots(figsize=(10, 8))
    
    residuals = y_true - y_pred
    
    # Scatter plot of residuals
    ax.scatter(y_pred, residuals, alpha=0.6, s=60, edgecolors='black', linewidth=0.5)
    ax.axhline(y=0, color='r', linestyle='--', linewidth=2, label='Zero Error')
    
    # Add trend line
    z = np.polyfit(y_pred, residuals, 1)
    p = np.poly1d(z)
    ax.plot(y_pred, p(y_pred), "g--", alpha=0.8, linewidth=2, label='Trend Line')
    
    # Formatting
    ax.set_xlabel('Predicted Values', fontsize=12)
    ax.set_ylabel('Residuals (Actual - Predicted)', fontsize=12)
    ax.set_title(f'{model_name} - {target_name} Residuals', fontsize=14)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add statistics
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mean_res = np.mean(residuals)
    std_res = np.std(residuals)
    
    textstr = f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}\nMean Residual: {mean_res:.3f}\nStd Residual: {std_res:.3f}'
    props = dict(boxstyle='round', facecolor='lightblue', alpha=0.8)
    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_performance_comparison_plot(models_data, target_name, save_path):
    """Create performance comparison bar chart."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    
    models = list(models_data.keys())
    maes = [models_data[model]['mae'] for model in models]
    r2s = [models_data[model]['r2'] for model in models]
    
    # MAE comparison
    bars1 = ax1.bar(models, maes, alpha=0.8, color='skyblue', edgecolor='navy', linewidth=1)
    ax1.set_xlabel('Models', fontsize=12)
    ax1.set_ylabel('Mean Absolute Error (MAE)', fontsize=12)
    ax1.set_title(f'{target_name} - Model Performance (MAE)', fontsize=14)
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bar, mae in zip(bars1, maes):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                f'{mae:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    # R² comparison
    bars2 = ax2.bar(models, r2s, alpha=0.8, color='lightcoral', edgecolor='darkred', linewidth=1)
    ax2.set_xlabel('Models', fontsize=12)
    ax2.set_ylabel('R² Score', fontsize=12)
    ax2.set_title(f'{target_name} - Model Performance (R²)', fontsize=14)
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.set_ylim(0, 1)
    
    # Add value labels on bars
    for bar, r2 in zip(bars2, r2s):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{r2:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_error_distribution_plot(y_true, y_pred_dict, target_name, save_path):
    """Create error distribution histogram."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    axes = axes.flatten()
    
    model_names = list(y_pred_dict.keys())[:4]  # Show top 4 models
    
    for idx, model_name in enumerate(model_names):
        ax = axes[idx]
        y_pred = y_pred_dict[model_name]
        errors = y_true - y_pred
        
        # Histogram
        ax.hist(errors, bins=20, alpha=0.7, color='steelblue', edgecolor='black', linewidth=1)
        ax.axvline(x=0, color='red', linestyle='--', linewidth=2, label='Zero Error')
        
        # Add normal curve overlay
        mu, sigma = np.mean(errors), np.std(errors)
        x = np.linspace(errors.min(), errors.max(), 100)
        ax.plot(x, len(errors) * (1/(sigma * np.sqrt(2 * np.pi))) * np.exp(-0.5 * ((x - mu) / sigma) ** 2),
                'r-', linewidth=2, label=f'Normal (μ={mu:.3f}, σ={sigma:.3f})')
        
        # Formatting
        ax.set_xlabel('Error (Actual - Predicted)', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title(f'{model_name} - Error Distribution', fontsize=14)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mae = mean_absolute_error(y_true, y_pred)
        textstr = f'MAE: {mae:.3f}\nMean: {mu:.3f}\nStd: {sigma:.3f}'
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def create_feature_importance_plot(model, feature_names, target_name, save_path):
    """Create feature importance plot for tree-based models."""
    if not hasattr(model, 'feature_importances_'):
        return
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Create horizontal bar plot
    bars = ax.barh(range(len(importances)), importances[indices], alpha=0.8, color='forestgreen')
    ax.set_yticks(range(len(importances)))
    ax.set_yticklabels([feature_names[i] for i in indices])
    ax.set_xlabel('Feature Importance', fontsize=12)
    ax.set_title(f'Feature Importance - {target_name}', fontsize=14)
    ax.grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, importance) in enumerate(zip(bars, importances[indices])):
        ax.text(importance + 0.001, bar.get_y() + bar.get_height()/2,
                f'{importance:.3f}', va='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

def main():
    """Main function to create all visualizations."""
    setup_plotting_style()
    
    # Create output directory
    output_dir = 'regression_plots'
    os.makedirs(output_dir, exist_ok=True)
    
    print("Loading data...")
    df = pd.read_csv('fish_frames.csv')
    
    # Define features and target
    feature_cols = ['Length (cm)', 'Width (cm)', 'Area (cm²)', 'Perimeter (cm)']
    target_col = 'Area_truth (cm²)'
    
    # Prepare data
    X = df[feature_cols].values
    y = df[target_col].values
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Define models (same as in results)
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
        'ridge_conservative': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=0.5, random_state=42))
        ]),
        'lasso_conservative': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=0.01, max_iter=2000, random_state=42))
        ])
    }
    
    print("Training models and creating visualizations...")
    
    # Train models and create predictions
    predictions = {}
    models_data = {}
    
    for name, model in models.items():
        print(f"Training {name}...")
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        
        # Store metrics
        models_data[name] = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred)
        }
        
        # Create individual plots
        scatter_path = os.path.join(output_dir, f'scatter_{name}_{target_col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")}.png')
        residual_path = os.path.join(output_dir, f'residual_{name}_{target_col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")}.png')
        
        create_scatter_plot(y_test, y_pred, name, target_col, scatter_path)
        create_residual_plot(y_test, y_pred, name, target_col, residual_path)
        
        # Feature importance for tree models
        if hasattr(model, 'feature_importances_'):
            importance_path = os.path.join(output_dir, f'importance_{name}_{target_col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")}.png')
            create_feature_importance_plot(model, feature_cols, target_col, importance_path)
    
    # Create comparison plots
    print("Creating comparison plots...")
    comparison_path = os.path.join(output_dir, f'comparison_{target_col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")}.png')
    create_performance_comparison_plot(models_data, target_col, comparison_path)
    
    # Create error distribution plot
    error_dist_path = os.path.join(output_dir, f'error_distribution_{target_col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")}.png')
    create_error_distribution_plot(y_test, predictions, target_col, error_dist_path)
    
    print(f"\nAll visualizations saved to '{output_dir}' directory:")
    print("Files created:")
    for file in os.listdir(output_dir):
        if file.endswith('.png'):
            print(f"  - {file}")
    
    print(f"\nKey Results for {target_col}:")
    best_model = min(models_data.items(), key=lambda x: x[1]['mae'])
    print(f"Best Model: {best_model[0]}")
    print(f"MAE: {best_model[1]['mae']:.3f}")
    print(f"R²: {best_model[1]['r2']:.3f}")

if __name__ == '__main__':
    main()