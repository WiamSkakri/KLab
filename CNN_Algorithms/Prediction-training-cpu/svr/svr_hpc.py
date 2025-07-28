import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Sklearn imports

# PyTorch imports (keeping minimal for compatibility)


def setup_logging():
    """Setup logging configuration"""
    print("=" * 80)
    print("SVR HYPERPARAMETER OPTIMIZATION AND CROSS-VALIDATION")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Script: {os.path.basename(__file__)}")
    print()


def load_and_preprocess_data(csv_file='combined.csv'):
    """Load and preprocess the dataset"""
    print("📊 LOADING AND PREPROCESSING DATA")
    print("-" * 50)

    try:
        # Load data
        print(f"Loading data from: {csv_file}")
        df = pd.read_csv(csv_file)
        print(
            f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

        # Display basic info
        print(f"Original columns: {list(df.columns)}")
        print(
            f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # One-hot encode the Algorithm column
        print("Encoding categorical variables...")
        df_encoded = pd.get_dummies(
            df, columns=['Algorithm'], prefix='Algorithm', dtype=int)
        print(
            f"After encoding: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} columns")

        # Update df to use encoded version
        df = df_encoded

        # Scaling features
        numerical_cols = ['Batch_Size', 'Input_Size', 'In_Channels',
                          'Out_Channels', 'Kernel_Size', 'Stride', 'Padding']
        print(f"Scaling numerical columns: {numerical_cols}")

        # Apply Standard Scaling to numerical columns only
        scaler = StandardScaler()
        df_scaled = df.copy()
        df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])

        # Define feature columns
        feature_cols = [
            col for col in df_scaled.columns if col != 'Execution_Time_ms']
        target_col = 'Execution_Time_ms'

        print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
        print(f"Target column: {target_col}")

        # Create features and target arrays
        X = df_scaled[feature_cols].values
        y = df_scaled[target_col].values

        print(f"✅ Feature matrix shape: {X.shape}")
        print(f"✅ Target vector shape: {y.shape}")
        print(
            f"Target statistics: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}, std={y.std():.4f}")

        return X, y, scaler, feature_cols, target_col

    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        sys.exit(1)


def setup_kfold_splits(X, y, k=5):
    """Setup K-Fold cross validation splits"""
    print(f"\n🔄 SETTING UP {k}-FOLD CROSS VALIDATION")
    print("-" * 50)

    # K-Fold Cross Validation Setup
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Store results for each fold
    fold_data = []

    # Prepare data splits for each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(
            f"Fold {fold}: Training={len(train_idx)}, Validation={len(val_idx)}")

        # Simple array slicing
        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        # Store for SVR training
        fold_data.append({
            'fold': fold,
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'train_size': len(X_train),
            'val_size': len(X_val)
        })

    print(f"✅ {k}-Fold splits prepared successfully")
    return fold_data


def setup_hyperparameter_grid():
    """Setup hyperparameter grids for SVR"""
    print(f"\n⚙️ SETTING UP HYPERPARAMETER GRIDS")
    print("-" * 50)

    # Define hyperparameter grid for SVR
    param_grid = {
        'kernel': ['rbf', 'linear'],  # Start with RBF and Linear
        'C': [0.1, 1, 10, 100],       # Regularization strength
        # Kernel coefficient (RBF only)
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'epsilon': [0.01, 0.1, 0.2]   # Tolerance
    }

    # For faster search with large dataset, use smaller grid
    param_grid_small = {
        'kernel': ['rbf'],
        'C': [1, 10, 100],
        'gamma': ['scale', 0.01, 0.1],
        'epsilon': [0.1]
    }

    full_combinations = len(param_grid['kernel']) * len(param_grid['C']) * len(
        param_grid['gamma']) * len(param_grid['epsilon'])
    small_combinations = len(param_grid_small['kernel']) * len(param_grid_small['C']) * len(
        param_grid_small['gamma']) * len(param_grid_small['epsilon'])

    print(f"Full grid combinations: {full_combinations}")
    print(f"Small grid combinations: {small_combinations}")
    print("Using small grid for faster computation")

    return param_grid_small


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def create_svr_with_gridsearch(param_grid, cv_folds=3):
    """Create SVR with GridSearchCV"""
    # Create base SVR model
    svr = SVR()

    # Create GridSearchCV
    grid_search = GridSearchCV(
        estimator=svr,
        param_grid=param_grid,
        cv=cv_folds,           # Internal CV for hyperparameter tuning
        scoring='neg_mean_squared_error',  # Use MSE for tuning
        n_jobs=-1,             # Use all available cores
        verbose=1              # Show progress
    )

    return grid_search


def evaluate_svr(model, X, y):
    """Evaluate SVR model"""
    # Make predictions
    predictions = model.predict(X)

    # Calculate metrics
    mape = calculate_mape(y, predictions)
    mae = mean_absolute_error(y, predictions)
    mse = mean_squared_error(y, predictions)
    r2 = r2_score(y, predictions)

    return {
        'mape': mape,
        'mae': mae,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2,
        'predictions': predictions,
        'actuals': y
    }


def print_metrics(metrics, title="Results"):
    """Print evaluation metrics"""
    print(f"{title}:")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")


def train_svr_cross_validation(fold_data, param_grid):
    """Train SVR with K-Fold Cross Validation"""
    print(f"\n🚀 STARTING SVR CROSS VALIDATION TRAINING")
    print("=" * 80)

    # Store results from all folds
    fold_metrics = []
    trained_models = []
    all_fold_results = []  # Store detailed results for plotting

    # Start timing
    start_time = time.time()

    # Loop through each fold
    for fold_info in fold_data:
        fold = fold_info['fold']
        X_train = fold_info['X_train']
        X_val = fold_info['X_val']
        y_train = fold_info['y_train']
        y_val = fold_info['y_val']

        print(f"\n🔄 FOLD {fold}")
        print("-" * 30)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")

        # Create SVR with GridSearchCV
        print("Running hyperparameter search...")
        svr_grid = create_svr_with_gridsearch(param_grid, cv_folds=3)

        # Fit the model (this does hyperparameter search + training)
        fold_start_time = time.time()
        svr_grid.fit(X_train, y_train)
        fold_training_time = time.time() - fold_start_time

        # Get the best model
        best_model = svr_grid.best_estimator_
        best_params = svr_grid.best_params_
        best_score = svr_grid.best_score_

        print(f"✅ Training complete in {fold_training_time:.1f} seconds")
        print(f"Best parameters: {best_params}")
        print(f"Best CV score (neg MSE): {best_score:.4f}")

        # Evaluate on training and validation sets
        train_metrics = evaluate_svr(best_model, X_train, y_train)
        val_metrics = evaluate_svr(best_model, X_val, y_val)

        # Store detailed results for plotting
        fold_details = {
            'fold': fold,
            'model': best_model,
            'best_params': best_params,
            'train_predictions': train_metrics['predictions'],
            'train_actuals': train_metrics['actuals'],
            'val_predictions': val_metrics['predictions'],
            'val_actuals': val_metrics['actuals'],
            'train_metrics': train_metrics,
            'val_metrics': val_metrics,
            'grid_search_results': svr_grid.cv_results_,
            'training_time': fold_training_time
        }
        all_fold_results.append(fold_details)

        # Store results for this fold
        fold_result = {
            'fold': fold,
            'train_mape': train_metrics['mape'],
            'val_mape': val_metrics['mape'],
            'train_mae': train_metrics['mae'],
            'val_mae': val_metrics['mae'],
            'train_r2': train_metrics['r2'],
            'val_r2': val_metrics['r2'],
            'train_rmse': train_metrics['rmse'],
            'val_rmse': val_metrics['rmse'],
            'best_params': best_params,
            'best_cv_score': best_score,
            'training_time': fold_training_time
        }

        fold_metrics.append(fold_result)
        trained_models.append(best_model)

        # Print fold results
        print(f"📊 Fold {fold} Results:")
        print_metrics(train_metrics, f"  Training")
        print_metrics(val_metrics, f"  Validation")

    # Calculate total training time
    total_time = time.time() - start_time

    return fold_metrics, trained_models, total_time, all_fold_results


def analyze_and_save_results(fold_metrics, trained_models, total_time, all_fold_results):
    """Analyze results and save to files"""
    print(f"\n📊 ANALYZING AND SAVING RESULTS")
    print("=" * 80)

    # Calculate average performance across all folds
    avg_train_mape = np.mean([f['train_mape'] for f in fold_metrics])
    avg_val_mape = np.mean([f['val_mape'] for f in fold_metrics])
    std_val_mape = np.std([f['val_mape'] for f in fold_metrics])

    avg_train_mae = np.mean([f['train_mae'] for f in fold_metrics])
    avg_val_mae = np.mean([f['val_mae'] for f in fold_metrics])
    std_val_mae = np.std([f['val_mae'] for f in fold_metrics])

    avg_train_r2 = np.mean([f['train_r2'] for f in fold_metrics])
    avg_val_r2 = np.mean([f['val_r2'] for f in fold_metrics])
    std_val_r2 = np.std([f['val_r2'] for f in fold_metrics])

    avg_training_time = np.mean([f['training_time'] for f in fold_metrics])

    # Print summary
    print(f"CROSS VALIDATION SUMMARY:")
    print(f"Average Train MAPE: {avg_train_mape:.2f}%")
    print(f"Average Val MAPE:   {avg_val_mape:.2f}% ± {std_val_mape:.2f}%")
    print(f"Average Train MAE:  {avg_train_mae:.4f}")
    print(f"Average Val MAE:    {avg_val_mae:.4f} ± {std_val_mae:.4f}")
    print(f"Average Train R²:   {avg_train_r2:.4f}")
    print(f"Average Val R²:     {avg_val_r2:.4f} ± {std_val_r2:.4f}")
    print(f"Average Training Time: {avg_training_time:.1f} seconds per fold")
    print(f"Total Training Time: {total_time:.1f} seconds")

    # Detailed results table
    print(f"\nDetailed Results by Fold:")
    print(f"{'Fold':<4} {'Train MAPE':<11} {'Val MAPE':<9} {'Train R²':<8} {'Val R²':<7} {'Time(s)':<8}")
    print("-" * 60)
    for f in fold_metrics:
        print(f"{f['fold']:<4} {f['train_mape']:<11.2f} {f['val_mape']:<9.2f} "
              f"{f['train_r2']:<8.4f} {f['val_r2']:<7.4f} {f['training_time']:<8.1f}")

    # Best performing fold
    best_fold = min(fold_metrics, key=lambda x: x['val_mape'])
    print(
        f"\n🏆 Best performing fold: Fold {best_fold['fold']} (Val MAPE: {best_fold['val_mape']:.2f}%)")

    # Most common hyperparameters
    print(f"\nMost common best parameters:")
    all_params = [f['best_params'] for f in fold_metrics]
    for param in ['kernel', 'C', 'gamma', 'epsilon']:
        values = [p.get(param) for p in all_params if param in p]
        if values:
            most_common = max(set(values), key=values.count)
            print(
                f"  {param}: {most_common} (appeared in {values.count(most_common)}/{len(values)} folds)")

    # Save results to CSV
    results_df = pd.DataFrame(fold_metrics)
    results_csv = 'svr_training_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\n💾 Results saved to: {results_csv}")

    # Save best model
    best_model_idx = np.argmin([f['val_mape'] for f in fold_metrics])
    best_model = trained_models[best_model_idx]
    best_model_file = 'best_svr_model.pkl'
    joblib.dump(best_model, best_model_file)
    print(f"💾 Best model saved to: {best_model_file}")

    # Create summary dictionary
    summary = {
        'avg_train_mape': avg_train_mape,
        'avg_val_mape': avg_val_mape,
        'std_val_mape': std_val_mape,
        'avg_train_mae': avg_train_mae,
        'avg_val_mae': avg_val_mae,
        'std_val_mae': std_val_mae,
        'avg_train_r2': avg_train_r2,
        'avg_val_r2': avg_val_r2,
        'std_val_r2': std_val_r2,
        'avg_training_time': avg_training_time,
        'total_time': total_time,
        'best_fold': best_fold['fold'],
        'best_val_mape': best_fold['val_mape']
    }

    return summary, all_fold_results


def create_comprehensive_plots(fold_metrics, all_fold_results, summary):
    """Create comprehensive evaluation plots for SVR model"""
    print(f"\n📊 CREATING COMPREHENSIVE EVALUATION PLOTS")
    print("=" * 80)

    # Find best fold for detailed analysis
    best_fold_idx = np.argmin([f['val_mape'] for f in fold_metrics])
    best_fold_details = all_fold_results[best_fold_idx]

    # Collect all predictions and actuals for overall analysis
    all_train_predictions = np.concatenate(
        [fold['train_predictions'] for fold in all_fold_results])
    all_train_actuals = np.concatenate(
        [fold['train_actuals'] for fold in all_fold_results])
    all_val_predictions = np.concatenate(
        [fold['val_predictions'] for fold in all_fold_results])
    all_val_actuals = np.concatenate(
        [fold['val_actuals'] for fold in all_fold_results])

    # Calculate residuals
    train_residuals = all_train_actuals - all_train_predictions
    val_residuals = all_val_actuals - all_val_predictions

    # ==========================================
    # PLOT 1: MAIN SVR EVALUATION DASHBOARD
    # ==========================================
    plt.figure(figsize=(18, 12))

    # 1.1: Prediction vs Actual (Training)
    plt.subplot(2, 4, 1)
    plt.scatter(all_train_actuals, all_train_predictions,
                alpha=0.6, s=8, color='blue')
    min_val = min(all_train_actuals.min(), all_train_predictions.min())
    max_val = max(all_train_actuals.max(), all_train_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual Execution Time')
    plt.ylabel('Predicted Execution Time')
    plt.title('Training: Predictions vs Actual')
    plt.grid(True, alpha=0.3)

    # Add R² score
    train_r2 = r2_score(all_train_actuals, all_train_predictions)
    plt.text(0.05, 0.95, f'R² = {train_r2:.4f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))

    # 1.2: Prediction vs Actual (Validation)
    plt.subplot(2, 4, 2)
    plt.scatter(all_val_actuals, all_val_predictions,
                alpha=0.6, s=8, color='red')
    min_val = min(all_val_actuals.min(), all_val_predictions.min())
    max_val = max(all_val_actuals.max(), all_val_predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('Actual Execution Time')
    plt.ylabel('Predicted Execution Time')
    plt.title('Validation: Predictions vs Actual')
    plt.grid(True, alpha=0.3)

    # Add R² score
    val_r2 = r2_score(all_val_actuals, all_val_predictions)
    plt.text(0.05, 0.95, f'R² = {val_r2:.4f}', transform=plt.gca().transAxes,
             bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.8))

    # 1.3: Residual Plot (Validation)
    plt.subplot(2, 4, 3)
    plt.scatter(all_val_predictions, val_residuals,
                alpha=0.6, s=8, color='green')
    plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
    plt.xlabel('Predicted Values')
    plt.ylabel('Residuals (Actual - Predicted)')
    plt.title('Residual Plot (Validation)')
    plt.grid(True, alpha=0.3)

    # 1.4: Cross-Validation MAPE Results
    plt.subplot(2, 4, 4)
    folds = [f['fold'] for f in fold_metrics]
    val_mapes = [f['val_mape'] for f in fold_metrics]
    colors = ['darkgreen' if i ==
              best_fold_idx else 'lightcoral' for i in range(len(folds))]
    bars = plt.bar(folds, val_mapes, color=colors)
    plt.xlabel('Fold')
    plt.ylabel('Validation MAPE (%)')
    plt.title('Cross-Validation Results')
    plt.grid(True, alpha=0.3, axis='y')

    # Highlight best fold
    best_fold_num = fold_metrics[best_fold_idx]['fold']
    best_mape = fold_metrics[best_fold_idx]['val_mape']
    plt.text(best_fold_num, best_mape + 0.1, f'Best\n{best_mape:.2f}%',
             ha='center', va='bottom', fontweight='bold', color='darkgreen')

    # 1.5: Error Distribution
    plt.subplot(2, 4, 5)
    percentage_errors = np.abs(val_residuals / all_val_actuals) * 100
    plt.hist(percentage_errors, bins=25, alpha=0.7,
             color='skyblue', edgecolor='black')
    plt.xlabel('Absolute Percentage Error (%)')
    plt.ylabel('Frequency')
    plt.title('Validation Error Distribution')
    plt.grid(True, alpha=0.3)

    # Add statistics
    mean_ape = np.mean(percentage_errors)
    median_ape = np.median(percentage_errors)
    plt.axvline(mean_ape, color='red', linestyle='--',
                linewidth=2, label=f'Mean: {mean_ape:.2f}%')
    plt.axvline(median_ape, color='orange', linestyle='--',
                linewidth=2, label=f'Median: {median_ape:.2f}%')
    plt.legend()

    # 1.6: Performance Metrics Comparison
    plt.subplot(2, 4, 6)
    metrics = ['MAPE', 'MAE', 'RMSE', 'R²']
    train_values = [
        np.mean([f['train_mape'] for f in fold_metrics]),
        np.mean([f['train_mae'] for f in fold_metrics]),
        np.mean([f['train_rmse'] for f in fold_metrics]),
        np.mean([f['train_r2'] for f in fold_metrics])
    ]
    val_values = [
        np.mean([f['val_mape'] for f in fold_metrics]),
        np.mean([f['val_mae'] for f in fold_metrics]),
        np.mean([f['val_rmse'] for f in fold_metrics]),
        np.mean([f['val_r2'] for f in fold_metrics])
    ]

    x = np.arange(len(metrics))
    width = 0.35
    plt.bar(x - width/2, train_values, width, label='Training', alpha=0.8)
    plt.bar(x + width/2, val_values, width, label='Validation', alpha=0.8)
    plt.xlabel('Metrics')
    plt.ylabel('Average Values')
    plt.title('Average Performance Metrics')
    plt.xticks(x, metrics)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # 1.7: Training Times
    plt.subplot(2, 4, 7)
    training_times = [f['training_time'] for f in fold_metrics]
    plt.bar(folds, training_times, color='lightblue', alpha=0.8)
    plt.xlabel('Fold')
    plt.ylabel('Training Time (seconds)')
    plt.title('Training Time by Fold')
    plt.grid(True, alpha=0.3, axis='y')

    # 1.8: Summary Statistics
    plt.subplot(2, 4, 8)
    plt.text(0.1, 0.9, "Cross-Validation Summary",
             fontsize=12, fontweight='bold')
    plt.text(
        0.1, 0.8, f"Avg Val MAPE: {summary['avg_val_mape']:.2f}% ± {summary['std_val_mape']:.2f}%", fontsize=10)
    plt.text(
        0.1, 0.7, f"Avg Val R²: {summary['avg_val_r2']:.4f} ± {summary['std_val_r2']:.4f}", fontsize=10)
    plt.text(
        0.1, 0.6, f"Best Fold: {summary['best_fold']} ({summary['best_val_mape']:.2f}%)", fontsize=10)
    plt.text(
        0.1, 0.5, f"Total Time: {summary['total_time']/60:.1f} minutes", fontsize=10)
    plt.text(
        0.1, 0.4, f"Avg Time/Fold: {summary['avg_training_time']:.1f}s", fontsize=10)
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('svr_main_evaluation.png', dpi=300, bbox_inches='tight')
    print("✅ Main evaluation dashboard saved to: svr_main_evaluation.png")

    # ==========================================
    # PLOT 2: HYPERPARAMETER ANALYSIS
    # ==========================================
    plt.figure(figsize=(15, 10))

    # Collect hyperparameter information
    all_params = [f['best_params'] for f in fold_metrics]

    # 2.1: Kernel distribution
    plt.subplot(2, 3, 1)
    kernels = [p.get('kernel', 'unknown') for p in all_params]
    kernel_counts = {k: kernels.count(k) for k in set(kernels)}
    plt.pie(kernel_counts.values(),
            labels=kernel_counts.keys(), autopct='%1.1f%%')
    plt.title('Best Kernel Distribution Across Folds')

    # 2.2: C parameter distribution
    plt.subplot(2, 3, 2)
    c_values = [p.get('C', 0) for p in all_params]
    plt.hist(c_values, bins=10, alpha=0.7,
             color='lightgreen', edgecolor='black')
    plt.xlabel('C Parameter')
    plt.ylabel('Frequency')
    plt.title('Distribution of Best C Values')
    plt.grid(True, alpha=0.3)

    # 2.3: Gamma parameter distribution
    plt.subplot(2, 3, 3)
    gamma_values = []
    gamma_labels = []
    for p in all_params:
        gamma = p.get('gamma', 'unknown')
        if isinstance(gamma, str):
            gamma_labels.append(gamma)
        else:
            gamma_values.append(gamma)

    if gamma_values:
        plt.hist(gamma_values, bins=10, alpha=0.7,
                 color='orange', edgecolor='black')
        plt.xlabel('Gamma Parameter')
        plt.ylabel('Frequency')
        plt.title('Distribution of Best Gamma Values')
        plt.grid(True, alpha=0.3)
    else:
        gamma_counts = {g: gamma_labels.count(g) for g in set(gamma_labels)}
        plt.pie(gamma_counts.values(),
                labels=gamma_counts.keys(), autopct='%1.1f%%')
        plt.title('Best Gamma Distribution')

    # 2.4: Epsilon parameter distribution
    plt.subplot(2, 3, 4)
    epsilon_values = [p.get('epsilon', 0) for p in all_params]
    plt.hist(epsilon_values, bins=10, alpha=0.7,
             color='pink', edgecolor='black')
    plt.xlabel('Epsilon Parameter')
    plt.ylabel('Frequency')
    plt.title('Distribution of Best Epsilon Values')
    plt.grid(True, alpha=0.3)

    # 2.5: Parameter vs Performance correlation
    plt.subplot(2, 3, 5)
    c_vs_mape = [(f['best_params'].get('C', 0), f['val_mape'])
                 for f in fold_metrics]
    c_vals, mape_vals = zip(*c_vs_mape)
    plt.scatter(c_vals, mape_vals, alpha=0.7, s=50)
    plt.xlabel('C Parameter')
    plt.ylabel('Validation MAPE (%)')
    plt.title('C Parameter vs MAPE')
    plt.grid(True, alpha=0.3)

    # 2.6: Best parameters summary table
    plt.subplot(2, 3, 6)
    param_text = "Best Parameters by Fold:\n\n"
    for i, (fold_metric, param) in enumerate(zip(fold_metrics, all_params)):
        fold_num = fold_metric['fold']
        mape = fold_metric['val_mape']
        param_text += f"Fold {fold_num} (MAPE: {mape:.2f}%):\n"
        param_text += f"  Kernel: {param.get('kernel', 'N/A')}\n"
        param_text += f"  C: {param.get('C', 'N/A')}\n"
        param_text += f"  Gamma: {param.get('gamma', 'N/A')}\n"
        param_text += f"  Epsilon: {param.get('epsilon', 'N/A')}\n\n"

    plt.text(0.05, 0.95, param_text, transform=plt.gca().transAxes, fontsize=8,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.8))
    plt.axis('off')

    plt.tight_layout()
    plt.savefig('svr_hyperparameter_analysis.png',
                dpi=300, bbox_inches='tight')
    print("✅ Hyperparameter analysis saved to: svr_hyperparameter_analysis.png")

    # ==========================================
    # PLOT 3: DETAILED METRICS ANALYSIS
    # ==========================================
    plt.figure(figsize=(15, 10))

    # 3.1: MAPE comparison across folds
    plt.subplot(2, 3, 1)
    train_mapes = [f['train_mape'] for f in fold_metrics]
    val_mapes = [f['val_mape'] for f in fold_metrics]
    x = np.arange(len(folds))
    width = 0.35
    plt.bar(x - width/2, train_mapes, width,
            label='Train MAPE', alpha=0.8, color='blue')
    plt.bar(x + width/2, val_mapes, width,
            label='Val MAPE', alpha=0.8, color='red')
    plt.xlabel('Fold')
    plt.ylabel('MAPE (%)')
    plt.title('MAPE Comparison Across Folds')
    plt.xticks(x, folds)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # 3.2: R² comparison across folds
    plt.subplot(2, 3, 2)
    train_r2s = [f['train_r2'] for f in fold_metrics]
    val_r2s = [f['val_r2'] for f in fold_metrics]
    plt.bar(x - width/2, train_r2s, width,
            label='Train R²', alpha=0.8, color='blue')
    plt.bar(x + width/2, val_r2s, width,
            label='Val R²', alpha=0.8, color='red')
    plt.xlabel('Fold')
    plt.ylabel('R² Score')
    plt.title('R² Comparison Across Folds')
    plt.xticks(x, folds)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # 3.3: MAE comparison across folds
    plt.subplot(2, 3, 3)
    train_maes = [f['train_mae'] for f in fold_metrics]
    val_maes = [f['val_mae'] for f in fold_metrics]
    plt.bar(x - width/2, train_maes, width,
            label='Train MAE', alpha=0.8, color='blue')
    plt.bar(x + width/2, val_maes, width,
            label='Val MAE', alpha=0.8, color='red')
    plt.xlabel('Fold')
    plt.ylabel('MAE')
    plt.title('MAE Comparison Across Folds')
    plt.xticks(x, folds)
    plt.legend()
    plt.grid(True, alpha=0.3, axis='y')

    # 3.4: Residuals distribution comparison
    plt.subplot(2, 3, 4)
    plt.hist(train_residuals, bins=30, alpha=0.6,
             label='Training', color='blue', density=True)
    plt.hist(val_residuals, bins=30, alpha=0.6,
             label='Validation', color='red', density=True)
    plt.xlabel('Residuals')
    plt.ylabel('Density')
    plt.title('Residuals Distribution Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # 3.5: Performance vs Training Time
    plt.subplot(2, 3, 5)
    training_times = [f['training_time'] for f in fold_metrics]
    plt.scatter(training_times, val_mapes, alpha=0.7, s=50, color='purple')
    plt.xlabel('Training Time (seconds)')
    plt.ylabel('Validation MAPE (%)')
    plt.title('Performance vs Training Time')
    plt.grid(True, alpha=0.3)

    # Add fold labels
    for i, (time, mape, fold_num) in enumerate(zip(training_times, val_mapes, folds)):
        plt.annotate(f'F{fold_num}', (time, mape), xytext=(5, 5),
                     textcoords='offset points', fontsize=8)

    # 3.6: Best fold detailed analysis
    plt.subplot(2, 3, 6)
    best_fold_num = fold_metrics[best_fold_idx]['fold']
    best_train_pred = best_fold_details['train_predictions']
    best_train_actual = best_fold_details['train_actuals']
    best_val_pred = best_fold_details['val_predictions']
    best_val_actual = best_fold_details['val_actuals']

    plt.scatter(best_val_actual, best_val_pred, alpha=0.6,
                s=15, color='darkgreen', label='Validation')
    plt.scatter(best_train_actual, best_train_pred, alpha=0.4,
                s=8, color='lightgreen', label='Training')

    min_val = min(np.min(best_train_actual), np.min(best_val_actual))
    max_val = max(np.max(best_train_actual), np.max(best_val_actual))
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)

    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title(f'Best Fold {best_fold_num} - Detailed View')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('svr_detailed_metrics.png', dpi=300, bbox_inches='tight')
    print("✅ Detailed metrics analysis saved to: svr_detailed_metrics.png")

    # Print summary of generated plots
    print(f"\n📊 VISUALIZATION SUMMARY")
    print("=" * 50)
    print("Generated SVR evaluation plots:")
    print("  1. svr_main_evaluation.png - Main evaluation dashboard (8 plots)")
    print("     • Training/Validation predictions vs actual")
    print("     • Residual analysis")
    print("     • Cross-validation results")
    print("     • Error distribution")
    print("     • Performance metrics comparison")
    print("     • Training times and summary statistics")
    print("  2. svr_hyperparameter_analysis.png - Hyperparameter analysis (6 plots)")
    print("     • Parameter distributions across folds")
    print("     • Parameter vs performance correlations")
    print("     • Best parameters summary")
    print("  3. svr_detailed_metrics.png - Detailed metrics analysis (6 plots)")
    print("     • Metric comparisons across all folds")
    print("     • Residual distributions")
    print("     • Performance vs training time")
    print("     • Best fold detailed analysis")

    return best_fold_idx, best_fold_details


def main():
    """Main function to run the SVR training pipeline"""
    # Setup logging
    setup_logging()

    try:
        # Load and preprocess data
        X, y, scaler, feature_cols, target_col = load_and_preprocess_data()

        # Setup K-Fold cross validation
        fold_data = setup_kfold_splits(X, y, k=5)

        # Setup hyperparameter grid
        param_grid = setup_hyperparameter_grid()

        # Train SVR with cross validation
        fold_metrics, trained_models, total_time, all_fold_results = train_svr_cross_validation(
            fold_data, param_grid)

        # Analyze and save results
        summary, all_fold_results = analyze_and_save_results(
            fold_metrics, trained_models, total_time, all_fold_results)

        # Create comprehensive plots
        best_fold_idx, best_fold_details = create_comprehensive_plots(
            fold_metrics, all_fold_results, summary)

        print(f"\n🎉 SVR TRAINING COMPLETED SUCCESSFULLY!")
        print(
            f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(
            f"📊 Final validation MAPE: {summary['avg_val_mape']:.2f}% ± {summary['std_val_mape']:.2f}%")
        print(f"🏆 Best fold achieved: {summary['best_val_mape']:.2f}% MAPE")

        print(f"\n📁 Output files generated:")
        print(f"  - svr_training_results.csv: Detailed results for each fold")
        print(f"  - best_svr_model.pkl: Best trained model")
        print(f"  - svr_main_evaluation.png: Main evaluation dashboard (8 plots)")
        print(f"  - svr_hyperparameter_analysis.png: Hyperparameter analysis (6 plots)")
        print(f"  - svr_detailed_metrics.png: Detailed metrics analysis (6 plots)")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
