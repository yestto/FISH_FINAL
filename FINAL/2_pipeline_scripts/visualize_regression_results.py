#!/usr/bin/env python3
"""
Visualization script for regression results.
Creates scatter plots, performance comparisons, and residual plots.
"""

import argparse
import os
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Ridge, Lasso
from sklearn.svm import SVR
from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor

# Set style
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_data(csv_path):
    """Load the fish frames dataset."""
    df = pd.read_csv(csv_path)
    return df

def prepare_regression_data(df, target_col, feature_cols):
    """Prepare data for regression."""
    X = df[feature_cols].values
    y = df[target_col].values
    groups = df['FishID'].values
    return X, y, groups

def train_models(X_train, y_train, groups_train, random_state=42):
    """Train multiple regression models with proper regularization."""
    models = {
        'ridge': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=1.0, random_state=random_state))
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
            subsample=0.8, min_samples_leaf=3, random_state=random_state
        ),
        'extra_trees_stable': ExtraTreesRegressor(
            n_estimators=150, max_depth=6, min_samples_leaf=5,
            max_features='sqrt', bootstrap=True, random_state=random_state
        ),
        'random_forest_stable': RandomForestRegressor(
            n_estimators=150, max_depth=6, min_samples_leaf=5,
            max_features='sqrt', random_state=random_state
        ),
        'ridge_conservative': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Ridge(alpha=0.5, random_state=random_state))
        ]),
        'lasso_conservative': Pipeline([
            ('scaler', StandardScaler()),
            ('model', Lasso(alpha=0.01, max_iter=2000, random_state=random_state))
        ])
    }
    
    trained_models = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        trained_models[name] = model
    
    return trained_models

def create_scatter_plots(df, target_col, feature_cols, models_dict, save_dir):
    """Create scatter plots showing predicted vs actual values."""
    X, y, groups = prepare_regression_data(df, target_col, feature_cols)
    
    # Split data for visualization
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=0.2, random_state=42, stratify=groups
    )
    
    # Create figure with subplots
    n_models = len(models_dict)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (model_name, model) in enumerate(models_dict.items()):
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Create scatter plot
        ax = axes[idx]
        ax.scatter(y_test, y_pred, alpha=0.6, s=50)
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        # Formatting
        ax.set_xlabel('Actual Values')
        ax.set_ylabel('Predicted Values')
        ax.set_title(f'{model_name}\nMAE={mae:.3f}, R²={r2:.3f}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Remove empty subplots
    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'scatter_plots_{target_col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_performance_comparison(df, target_col, feature_cols, models_dict, save_dir):
    """Create performance comparison charts."""
    X, y, groups = prepare_regression_data(df, target_col, feature_cols)
    
    # Use GroupKFold for cross-validation
    cv = GroupKFold(n_splits=5)
    
    results = []
    for model_name, model in models_dict.items():
        cv_scores = []
        for train_idx, val_idx in cv.split(X, y, groups):
            X_train_cv, X_val_cv = X[train_idx], X[val_idx]
            y_train_cv, y_val_cv = y[train_idx], y[val_idx]
            
            model.fit(X_train_cv, y_train_cv)
            y_pred_cv = model.predict(X_val_cv)
            
            mae = mean_absolute_error(y_val_cv, y_pred_cv)
            cv_scores.append(mae)
        
        results.append({
            'Model': model_name,
            'CV_MAE_mean': np.mean(cv_scores),
            'CV_MAE_std': np.std(cv_scores)
        })
    
    results_df = pd.DataFrame(results)
    
    # Create bar plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # MAE comparison
    bars = ax1.bar(results_df['Model'], results_df['CV_MAE_mean'], 
                   yerr=results_df['CV_MAE_std'], capsize=5, alpha=0.8)
    ax1.set_xlabel('Models')
    ax1.set_ylabel('Cross-Validation MAE')
    ax1.set_title(f'Model Performance Comparison - {target_col}')
    ax1.tick_params(axis='x', rotation=45)
    ax1.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for bar, mean_val, std_val in zip(bars, results_df['CV_MAE_mean'], results_df['CV_MAE_std']):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + std_val + 0.01,
                f'{mean_val:.3f}', ha='center', va='bottom', fontsize=9)
    
    # R² comparison (using test set)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    r2_scores = []
    model_names = []
    
    for model_name, model in models_dict.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2 = r2_score(y_test, y_pred)
        r2_scores.append(r2)
        model_names.append(model_name)
    
    bars2 = ax2.bar(model_names, r2_scores, alpha=0.8)
    ax2.set_xlabel('Models')
    ax2.set_ylabel('R² Score (Test Set)')
    ax2.set_title(f'R² Scores - {target_col}')
    ax2.tick_params(axis='x', rotation=45)
    ax2.grid(True, alpha=0.3)
    
    # Add value labels
    for bar, score in zip(bars2, r2_scores):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{score:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'performance_comparison_{target_col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_residual_plots(df, target_col, feature_cols, models_dict, save_dir):
    """Create residual plots to show prediction errors."""
    X, y, groups = prepare_regression_data(df, target_col, feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Create figure with subplots
    n_models = len(models_dict)
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()
    
    for idx, (model_name, model) in enumerate(models_dict.items()):
        # Make predictions
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        residuals = y_test - y_pred
        
        # Create residual plot
        ax = axes[idx]
        ax.scatter(y_pred, residuals, alpha=0.6, s=50)
        ax.axhline(y=0, color='r', linestyle='--', lw=2, label='Zero Error')
        
        # Add trend line
        z = np.polyfit(y_pred, residuals, 1)
        p = np.poly1d(z)
        ax.plot(y_pred, p(y_pred), "g--", alpha=0.8, label='Trend')
        
        # Formatting
        ax.set_xlabel('Predicted Values')
        ax.set_ylabel('Residuals (Actual - Predicted)')
        ax.set_title(f'{model_name} - Residuals')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        # Add statistics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        ax.text(0.05, 0.95, f'MAE: {mae:.3f}\nRMSE: {rmse:.3f}', 
                transform=ax.transAxes, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    # Remove empty subplots
    for idx in range(n_models, len(axes)):
        fig.delaxes(axes[idx])
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'residual_plots_{target_col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def create_feature_importance_plot(df, target_col, feature_cols, models_dict, save_dir):
    """Create feature importance plots for tree-based models."""
    tree_models = {k: v for k, v in models_dict.items() 
                   if k in ['extra_trees_stable', 'random_forest_stable', 'gbr_stable']}
    
    if not tree_models:
        return
    
    X, y, groups = prepare_regression_data(df, target_col, feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    fig, axes = plt.subplots(1, len(tree_models), figsize=(5*len(tree_models), 6))
    if len(tree_models) == 1:
        axes = [axes]
    
    for idx, (model_name, model) in enumerate(tree_models.items()):
        model.fit(X_train, y_train)
        
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        else:
            # For pipeline models
            importances = model.named_steps['model'].feature_importances_
        
        # Create bar plot
        ax = axes[idx]
        bars = ax.bar(feature_cols, importances, alpha=0.8)
        ax.set_xlabel('Features')
        ax.set_ylabel('Importance')
        ax.set_title(f'{model_name} - Feature Importance')
        ax.tick_params(axis='x', rotation=45)
        ax.grid(True, alpha=0.3)
        
        # Add value labels
        for bar, importance in zip(bars, importances):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.001,
                   f'{importance:.3f}', ha='center', va='bottom', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f'feature_importance_{target_col.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_")}.png'), dpi=300, bbox_inches='tight')
    plt.close()

def main():
    parser = argparse.ArgumentParser(description='Visualize regression results for fish dataset')
    parser.add_argument('--csv', required=True, help='Path to fish_frames.csv')
    parser.add_argument('--target', required=True, choices=['Length_truth (cm)', 'Width_truth (cm)', 'Area_truth (cm²)', 'Perimeter_truth (cm)'], 
                       help='Target column for regression')
    parser.add_argument('--output-dir', default='regression_plots', help='Directory to save plots')
    parser.add_argument('--random-state', type=int, default=42, help='Random state for reproducibility')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    df = load_data(args.csv)
    
    # Define feature columns
    feature_cols = ['Length (cm)', 'Width (cm)', 'Area (cm²)', 'Perimeter (cm)']
    
    # Prepare data
    X, y, groups = prepare_regression_data(df, args.target, feature_cols)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=args.random_state)
    
    # Train models
    models_dict = train_models(X_train, y_train, groups, args.random_state)
    
    # Create all visualizations
    print(f"Creating scatter plots for {args.target}...")
    create_scatter_plots(df, args.target, feature_cols, models_dict, args.output_dir)
    
    print(f"Creating performance comparison for {args.target}...")
    create_performance_comparison(df, args.target, feature_cols, models_dict, args.output_dir)
    
    print(f"Creating residual plots for {args.target}...")
    create_residual_plots(df, args.target, feature_cols, models_dict, args.output_dir)
    
    print(f"Creating feature importance plots for {args.target}...")
    create_feature_importance_plot(df, args.target, feature_cols, models_dict, args.output_dir)
    
    print(f"All plots saved to {args.output_dir}")
    print(f"Files created:")
    for file in os.listdir(args.output_dir):
        if args.target.replace(" ", "_").replace("(", "").replace(")", "").replace("/", "_") in file:
            print(f"  - {file}")

if __name__ == '__main__':
    main()