import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Check for GPU availability
try:
    import cuml
    from cuml.ensemble import RandomForestRegressor as cuRF
    from cuml.model_selection import GridSearchCV as cuGridSearchCV
    from cuml.metrics import mean_squared_error as cu_mse, r2_score as cu_r2
    import cupy as cp
    GPU_AVAILABLE = True
    print("✅ GPU libraries (cuML/CuPy) detected - will use GPU acceleration")
except ImportError as e:
    print(f"⚠️  GPU libraries not available: {e}")
    print("📌 Falling back to CPU-based sklearn")
    GPU_AVAILABLE = False

# Sklearn imports (fallback)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

# PyTorch imports
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

# For saving models
import joblib

# GPU Detection
if GPU_AVAILABLE:
    try:
        # Check if GPU is actually available
        cp.cuda.Device(0).use()
        print(
            f"🚀 Using GPU: {cp.cuda.runtime.getDeviceProperties(0)['name'].decode()}")
        print(
            f"🔧 GPU Memory: {cp.cuda.runtime.memGetInfo()[1] / 1024**3:.1f} GB total")
    except Exception as e:
        print(f"⚠️  GPU detected but not accessible: {e}")
        GPU_AVAILABLE = False
        print("📌 Falling back to CPU")

print("Starting Random Forest HPC Training")
print("=" * 60)
print(f"Script started at: {datetime.now()}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Available files: {os.listdir('.')}")
print(f"GPU Acceleration: {'Enabled' if GPU_AVAILABLE else 'Disabled'}")

# Load data
print("\n📊 Loading data...")
if not os.path.exists('combined.csv'):
    print("Error: combined.csv not found!")
    sys.exit(1)

df = pd.read_csv('combined.csv')
print(f"Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# One-hot encode the Algorithm column
df_encoded = pd.get_dummies(
    df, columns=['Algorithm'], prefix='Algorithm', dtype=int)
df = df_encoded

# Define feature columns
feature_cols = [col for col in df.columns if col != 'Execution_Time_ms']
target_col = 'Execution_Time_ms'

print(f"\n📈 Data preprocessing:")
print(f"Features: {len(feature_cols)} columns")
print(f"Target: {target_col}")
print(f"Feature columns: {feature_cols}")

# Create features and target arrays
X = df[feature_cols].values
y = df[target_col].values

# Convert to GPU arrays if using cuML
if GPU_AVAILABLE:
    print("🔄 Converting data to GPU format...")
    X_gpu = cp.asarray(X, dtype=cp.float32)
    y_gpu = cp.asarray(y, dtype=cp.float32)
    print(
        f"GPU Memory usage after data loading: {cp.cuda.runtime.memGetInfo()[0] / 1024**3:.1f} GB used")
else:
    X_gpu = X
    y_gpu = y

print(f"X shape: {X.shape}")
print(f"y shape: {y.shape}")
print(
    f"y statistics: min={y.min():.2f}, max={y.max():.2f}, mean={y.mean():.2f}")

# K-Fold Cross Validation Setup
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

print(f"\n🔀 K-Fold Cross Validation Setup:")
print(f"Number of folds: {k}")

# Store results for each fold
fold_data = []

# Prepare data splits for each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"\n=== FOLD {fold} ===")

    if GPU_AVAILABLE:
        # GPU array slicing
        X_train = X_gpu[train_idx]
        X_val = X_gpu[val_idx]
        y_train = y_gpu[train_idx]
        y_val = y_gpu[val_idx]
    else:
        # Simple array slicing
        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    fold_data.append({
        'fold': fold,
        'X_train': X_train,
        'X_val': X_val,
        'y_train': y_train,
        'y_val': y_val,
        'train_size': len(X_train),
        'val_size': len(X_val)
    })

# Random Forest Model and Hyperparameters
print(f"\n🌲 Random Forest Configuration:")
print(f"Backend: {'cuML (GPU)' if GPU_AVAILABLE else 'sklearn (CPU)'}")

# Hyperparameter grid - optimized for GPU
if GPU_AVAILABLE:
    # cuML parameters
    param_grid_hpc = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, -1],  # -1 means no limit in cuML
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': [0.33, 0.5, 1.0]  # cuML uses float values
    }
else:
    # sklearn parameters
    param_grid_hpc = {
        'n_estimators': [100, 200, 300],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', 'log2']
    }

print(
    f"Hyperparameter grid combinations: {np.prod([len(v) for v in param_grid_hpc.values()])}")


def create_rf_with_gridsearch(cv_folds=3):
    if GPU_AVAILABLE:
        # cuML Random Forest
        rf = cuRF(random_state=42, split_criterion=2)  # 2 = MSE for regression

        # cuML GridSearchCV
        grid_search = cuGridSearchCV(
            estimator=rf,
            param_grid=param_grid_hpc,
            cv=cv_folds,
            scoring='neg_mean_squared_error'
        )
    else:
        # sklearn Random Forest
        rf = RandomForestRegressor(random_state=42, n_jobs=-1)

        grid_search = GridSearchCV(
            estimator=rf,
            param_grid=param_grid_hpc,
            cv=cv_folds,
            scoring='neg_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )

    return grid_search

# Evaluation Functions


def calculate_mape(y_true, y_pred):
    if GPU_AVAILABLE:
        y_true_np = cp.asnumpy(y_true) if hasattr(y_true, 'device') else y_true
        y_pred_np = cp.asnumpy(y_pred) if hasattr(y_pred, 'device') else y_pred
    else:
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)

    mask = y_true_np != 0
    return np.mean(np.abs((y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask])) * 100


def evaluate_rf(model, X, y):
    """Evaluate Random Forest model"""
    predictions = model.predict(X)

    if GPU_AVAILABLE:
        # Convert GPU arrays to numpy for some calculations
        y_np = cp.asnumpy(y) if hasattr(y, 'device') else y
        pred_np = cp.asnumpy(predictions) if hasattr(
            predictions, 'device') else predictions

        # Use cuML metrics when possible
        try:
            mse = float(cu_mse(y, predictions))
            r2 = float(cu_r2(y, predictions))
        except:
            mse = mean_squared_error(y_np, pred_np)
            r2 = r2_score(y_np, pred_np)

        mae = mean_absolute_error(y_np, pred_np)
        mape = calculate_mape(y_np, pred_np)
    else:
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
        'predictions': predictions
    }


def print_metrics(metrics, title="Results"):
    print(f"\n{title}:")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")


# Main Training Loop
print("\n🚀 Starting Random Forest K-Fold Cross Validation Training")
print("=" * 60)

# Store results from all folds
fold_metrics = []
trained_models = []

# Start timing
start_time = time.time()

# Loop through each fold
for fold_info in fold_data:
    fold = fold_info['fold']
    X_train = fold_info['X_train']
    X_val = fold_info['X_val']
    y_train = fold_info['y_train']
    y_val = fold_info['y_val']

    print(f"\n🌲 FOLD {fold}")
    print("-" * 20)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    if GPU_AVAILABLE:
        print(
            f"GPU Memory before training: {cp.cuda.runtime.memGetInfo()[0] / 1024**3:.1f} GB used")

    # Create Random Forest with GridSearchCV
    print("Running hyperparameter search...")
    rf_grid = create_rf_with_gridsearch(cv_folds=3)

    # Fit the model
    fold_start_time = time.time()
    rf_grid.fit(X_train, y_train)
    fold_training_time = time.time() - fold_start_time

    # Get the best model
    best_model = rf_grid.best_estimator_
    best_params = rf_grid.best_params_
    best_score = rf_grid.best_score_

    print(f"✅ Training complete in {fold_training_time:.1f} seconds")
    print(f"Best parameters: {best_params}")
    print(f"Best CV score (neg MSE): {best_score:.4f}")

    if GPU_AVAILABLE:
        print(
            f"GPU Memory after training: {cp.cuda.runtime.memGetInfo()[0] / 1024**3:.1f} GB used")

    # Evaluate on training and validation sets
    train_metrics = evaluate_rf(best_model, X_train, y_train)
    val_metrics = evaluate_rf(best_model, X_val, y_val)

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
    print(f"🌳 Fold {fold} Results:")
    print_metrics(train_metrics, f"  Training")
    print_metrics(val_metrics, f"  Validation")

    # Save model for this fold
    model_filename = f'rf_model_fold_{fold}.joblib'

    # For cuML models, we might need to convert to CPU for saving
    if GPU_AVAILABLE:
        try:
            # Try to save the cuML model directly
            joblib.dump(best_model, model_filename)
        except Exception as e:
            print(f"Warning: Could not save cuML model directly: {e}")
            print("Converting to sklearn format for saving...")
            # Convert to sklearn format for compatibility
            from sklearn.ensemble import RandomForestRegressor as skRF
            sk_model = skRF(**best_params, random_state=42)
            # Train sklearn model on CPU data for saving
            X_train_cpu = cp.asnumpy(X_train) if GPU_AVAILABLE else X_train
            y_train_cpu = cp.asnumpy(y_train) if GPU_AVAILABLE else y_train
            sk_model.fit(X_train_cpu, y_train_cpu)
            joblib.dump(sk_model, model_filename)
    else:
        joblib.dump(best_model, model_filename)

    print(f"Model saved: {model_filename}")

# Calculate total training time
total_time = time.time() - start_time

# Calculate average performance across all folds
print("\n" + "=" * 60)
print("🌲 RANDOM FOREST CROSS VALIDATION SUMMARY")
print("=" * 60)

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

print(f"Backend Used: {'cuML (GPU)' if GPU_AVAILABLE else 'sklearn (CPU)'}")
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

# Save the best model
best_model_idx = best_fold['fold'] - 1
best_model = trained_models[best_model_idx]

try:
    joblib.dump(best_model, 'best_rf_model.joblib')
    print(f"Best model saved as: best_rf_model.joblib")
except Exception as e:
    print(f"Warning: Could not save best model: {e}")
    if GPU_AVAILABLE:
        print("Converting best model to sklearn format for saving...")
        from sklearn.ensemble import RandomForestRegressor as skRF
        best_params = best_fold['best_params']
        sk_model = skRF(**best_params, random_state=42)
        # Get the training data for the best fold
        best_fold_data = fold_data[best_model_idx]
        X_train_cpu = cp.asnumpy(
            best_fold_data['X_train']) if GPU_AVAILABLE else best_fold_data['X_train']
        y_train_cpu = cp.asnumpy(
            best_fold_data['y_train']) if GPU_AVAILABLE else best_fold_data['y_train']
        sk_model.fit(X_train_cpu, y_train_cpu)
        joblib.dump(sk_model, 'best_rf_model.joblib')
        print(f"Best model saved as: best_rf_model.joblib (sklearn format)")

# Most common hyperparameters
print(f"\nMost common best parameters:")
all_params = [f['best_params'] for f in fold_metrics]
param_summary = {}
param_names = ['n_estimators', 'max_depth',
               'min_samples_split', 'min_samples_leaf', 'max_features']
for param in param_names:
    values = [p.get(param) for p in all_params if param in p]
    if values:
        most_common = max(set(values), key=values.count)
        count = values.count(most_common)
        param_summary[param] = most_common
        print(f"  {param}: {most_common} (appeared in {count}/{len(values)} folds)")

# Save results to CSV
results_df = pd.DataFrame(fold_metrics)
results_df.to_csv('rdf_training_results.csv', index=False)
print(f"\nResults saved to: rdf_training_results.csv")

# Create summary metrics
summary_metrics = {
    'backend': 'cuML (GPU)' if GPU_AVAILABLE else 'sklearn (CPU)',
    'total_training_time': total_time,
    'avg_train_mape': avg_train_mape,
    'avg_val_mape': avg_val_mape,
    'std_val_mape': std_val_mape,
    'avg_train_mae': avg_train_mae,
    'avg_val_mae': avg_val_mae,
    'std_val_mae': std_val_mae,
    'avg_train_r2': avg_train_r2,
    'avg_val_r2': avg_val_r2,
    'std_val_r2': std_val_r2,
    'best_fold': best_fold['fold'],
    'best_val_mape': best_fold['val_mape'],
    'num_folds': k,
    'timestamp': datetime.now().isoformat()
}

# Save summary metrics
summary_df = pd.DataFrame([summary_metrics])
summary_df.to_csv('rdf_summary_metrics.csv', index=False)
print(f"Summary metrics saved to: rdf_summary_metrics.csv")

# ===============================
# COMPREHENSIVE VISUALIZATION
# ===============================


def create_evaluation_dashboard(fold_data, fold_metrics, trained_models, feature_cols, target_col):
    """Create comprehensive Random Forest evaluation dashboard"""
    print("\n📊 Creating Random Forest evaluation dashboard...")

    # Collect all predictions and actuals for overall analysis
    all_predictions = []
    all_actuals = []
    all_fold_labels = []

    # Get predictions from best model on all data
    best_fold_idx = min(range(len(fold_metrics)),
                        key=lambda i: fold_metrics[i]['val_mape'])
    best_model = trained_models[best_fold_idx]

    for i, fold_info in enumerate(fold_data):
        # Get validation predictions for this fold
        val_pred = trained_models[i].predict(fold_info['X_val'])
        val_actual = fold_info['y_val']

        # Convert GPU arrays to numpy if needed
        if GPU_AVAILABLE:
            val_pred = cp.asnumpy(val_pred) if hasattr(
                val_pred, 'device') else val_pred
            val_actual = cp.asnumpy(val_actual) if hasattr(
                val_actual, 'device') else val_actual

        all_predictions.extend(val_pred)
        all_actuals.extend(val_actual)
        all_fold_labels.extend([f'Fold {i+1}'] * len(val_pred))

    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    # Create main dashboard
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('Random Forest Model Evaluation Dashboard',
                 fontsize=16, fontweight='bold')

    # 1. Predictions vs Actual (Overall)
    ax = axes[0, 0]
    ax.scatter(all_actuals, all_predictions, alpha=0.6, s=20)
    min_val, max_val = min(all_actuals.min(), all_predictions.min()), max(
        all_actuals.max(), all_predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Execution Time (ms)')
    ax.set_ylabel('Predicted Execution Time (ms)')
    ax.set_title('Predictions vs Actual (All Folds)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add R² annotation
    r2_overall = r2_score(all_actuals, all_predictions)
    ax.text(0.05, 0.95, f'R² = {r2_overall:.4f}', transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    # 2. Residual Plot
    ax = axes[0, 1]
    residuals = all_predictions - all_actuals
    ax.scatter(all_predictions, residuals, alpha=0.6, s=20)
    ax.axhline(y=0, color='r', linestyle='--', lw=2)
    ax.set_xlabel('Predicted Execution Time (ms)')
    ax.set_ylabel('Residuals (Predicted - Actual)')
    ax.set_title('Residual Analysis')
    ax.grid(True, alpha=0.3)

    # 3. Cross-Validation Results
    ax = axes[0, 2]
    folds = [f['fold'] for f in fold_metrics]
    val_mapes = [f['val_mape'] for f in fold_metrics]
    val_r2s = [f['val_r2'] for f in fold_metrics]

    ax2 = ax.twinx()
    bars1 = ax.bar([f - 0.2 for f in folds], val_mapes, 0.4,
                   label='Validation MAPE (%)', alpha=0.7, color='orange')
    bars2 = ax2.bar([f + 0.2 for f in folds], val_r2s, 0.4,
                    label='Validation R²', alpha=0.7, color='green')

    ax.set_xlabel('Fold')
    ax.set_ylabel('MAPE (%)', color='orange')
    ax2.set_ylabel('R²', color='green')
    ax.set_title('Cross-Validation Performance')
    ax.grid(True, alpha=0.3)

    # Highlight best fold
    best_fold_num = min(fold_metrics, key=lambda x: x['val_mape'])['fold']
    ax.axvline(x=best_fold_num, color='red', linestyle='--',
               alpha=0.7, label=f'Best Fold ({best_fold_num})')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 4. Error Distribution
    ax = axes[1, 0]
    percentage_errors = np.abs(
        (all_predictions - all_actuals) / all_actuals) * 100
    ax.hist(percentage_errors, bins=30, alpha=0.7, edgecolor='black')
    ax.axvline(x=np.mean(percentage_errors), color='red', linestyle='--', lw=2,
               label=f'Mean: {np.mean(percentage_errors):.2f}%')
    ax.axvline(x=np.median(percentage_errors), color='green', linestyle='--', lw=2,
               label=f'Median: {np.median(percentage_errors):.2f}%')
    ax.set_xlabel('Percentage Error (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 5. Performance Metrics Comparison
    ax = axes[1, 1]
    metrics_names = ['MAPE (%)', 'MAE', 'RMSE', 'R²']
    train_metrics = [np.mean([f['train_mape'] for f in fold_metrics]),
                     np.mean([f['train_mae'] for f in fold_metrics]),
                     np.mean([f['train_rmse'] for f in fold_metrics]),
                     np.mean([f['train_r2'] for f in fold_metrics])]
    val_metrics = [np.mean([f['val_mape'] for f in fold_metrics]),
                   np.mean([f['val_mae'] for f in fold_metrics]),
                   np.mean([f['val_rmse'] for f in fold_metrics]),
                   np.mean([f['val_r2'] for f in fold_metrics])]

    x = np.arange(len(metrics_names))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_metrics, width,
                   label='Training', alpha=0.7)
    bars2 = ax.bar(x + width/2, val_metrics, width,
                   label='Validation', alpha=0.7)

    ax.set_xlabel('Metrics')
    ax.set_ylabel('Values')
    ax.set_title('Average Performance Metrics Comparison')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # 6. Training Time Analysis
    ax = axes[1, 2]
    training_times = [f['training_time'] for f in fold_metrics]
    bars = ax.bar(folds, training_times, alpha=0.7, color='purple')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('Training Time by Fold')
    ax.grid(True, alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom')

    # Add average line
    avg_time = np.mean(training_times)
    ax.axhline(y=avg_time, color='red', linestyle='--', alpha=0.7,
               label=f'Average: {avg_time:.1f}s')
    ax.legend()

    # 7. Fold Performance Comparison
    ax = axes[2, 0]
    fold_nums = [f['fold'] for f in fold_metrics]
    mape_vals = [f['val_mape'] for f in fold_metrics]
    r2_vals = [f['val_r2'] for f in fold_metrics]

    # Create scatter plot
    scatter = ax.scatter(mape_vals, r2_vals, s=100,
                         alpha=0.7, c=fold_nums, cmap='viridis')

    # Add fold labels
    for i, (mape, r2, fold) in enumerate(zip(mape_vals, r2_vals, fold_nums)):
        ax.annotate(f'F{fold}', (mape, r2), xytext=(
            5, 5), textcoords='offset points')

    ax.set_xlabel('Validation MAPE (%)')
    ax.set_ylabel('Validation R²')
    ax.set_title('Fold Performance Comparison')
    ax.grid(True, alpha=0.3)

    # Highlight best fold
    best_idx = min(range(len(fold_metrics)),
                   key=lambda i: fold_metrics[i]['val_mape'])
    ax.scatter(mape_vals[best_idx], r2_vals[best_idx], s=200, color='red',
               marker='*', label=f'Best Fold ({fold_nums[best_idx]})')
    ax.legend()

    # 8. Model Summary Statistics
    ax = axes[2, 1]
    ax.axis('off')

    summary_text = f"""
Model Summary Statistics:
─────────────────────────
Backend: {'cuML (GPU)' if GPU_AVAILABLE else 'sklearn (CPU)'}
Cross-Validation Folds: {len(fold_metrics)}
Best Fold: {best_fold_num}

Performance Metrics:
• Avg Val MAPE: {np.mean(val_mapes):.2f}% ± {np.std(val_mapes):.2f}%
• Avg Val R²: {np.mean(val_r2s):.4f} ± {np.std(val_r2s):.4f}
• Best Val MAPE: {min(val_mapes):.2f}%
• Best Val R²: {max(val_r2s):.4f}

Training Efficiency:
• Avg Training Time: {np.mean(training_times):.1f}s
• Total Training Time: {sum(training_times):.1f}s
• Samples: {len(all_actuals)} total

Data Statistics:
• Execution Time Range: {all_actuals.min():.2f} - {all_actuals.max():.2f} ms
• Mean Execution Time: {all_actuals.mean():.2f} ms
• Features: {len(feature_cols)}
"""

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    # 9. Feature Importance (if available)
    ax = axes[2, 2]
    try:
        # Get feature importance from best model
        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_
            if GPU_AVAILABLE and hasattr(feature_importance, 'device'):
                feature_importance = cp.asnumpy(feature_importance)

            # Get top 10 features
            top_indices = np.argsort(feature_importance)[-10:]
            top_features = [feature_cols[i] for i in top_indices]
            top_importance = feature_importance[top_indices]

            bars = ax.barh(range(len(top_features)), top_importance, alpha=0.7)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features)
            ax.set_xlabel('Feature Importance')
            ax.set_title('Top 10 Feature Importance')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Feature importance\nnot available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance (N/A)')
    except Exception as e:
        ax.text(0.5, 0.5, f'Feature importance\nerror: {str(e)[:30]}...',
                ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Feature Importance (Error)')

    plt.tight_layout()
    plt.savefig('rdf_evaluation_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Evaluation dashboard saved: rdf_evaluation_dashboard.png")


def create_hyperparameter_analysis(fold_metrics, param_grid_hpc):
    """Create hyperparameter analysis plots"""
    print("\n📊 Creating hyperparameter analysis...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Random Forest Hyperparameter Analysis',
                 fontsize=16, fontweight='bold')

    # Extract hyperparameter data
    all_params = [f['best_params'] for f in fold_metrics]
    val_mapes = [f['val_mape'] for f in fold_metrics]
    val_r2s = [f['val_r2'] for f in fold_metrics]
    training_times = [f['training_time'] for f in fold_metrics]

    # 1. n_estimators analysis
    ax = axes[0, 0]
    n_estimators_vals = [p.get('n_estimators', 100) for p in all_params]
    unique_n_est = sorted(set(n_estimators_vals))
    n_est_mapes = [np.mean([m for i, m in enumerate(val_mapes) if n_estimators_vals[i] == n])
                   for n in unique_n_est]

    bars = ax.bar(range(len(unique_n_est)), n_est_mapes, alpha=0.7)
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('Average Validation MAPE (%)')
    ax.set_title('MAPE vs n_estimators')
    ax.set_xticks(range(len(unique_n_est)))
    ax.set_xticklabels(unique_n_est)
    ax.grid(True, alpha=0.3)

    # Highlight best
    best_idx = np.argmin(n_est_mapes)
    bars[best_idx].set_color('red')
    bars[best_idx].set_alpha(1.0)

    # 2. max_depth analysis
    ax = axes[0, 1]
    max_depth_vals = [p.get('max_depth', 10) for p in all_params]
    # Handle None/-1 values for unlimited depth
    max_depth_vals = [str(d) if d is not None and d != -
                      1 else 'None' for d in max_depth_vals]
    unique_depths = sorted(set(max_depth_vals), key=lambda x: float(
        'inf') if x == 'None' else float(x))
    depth_mapes = [np.mean([m for i, m in enumerate(val_mapes) if max_depth_vals[i] == d])
                   for d in unique_depths]

    bars = ax.bar(range(len(unique_depths)), depth_mapes, alpha=0.7)
    ax.set_xlabel('max_depth')
    ax.set_ylabel('Average Validation MAPE (%)')
    ax.set_title('MAPE vs max_depth')
    ax.set_xticks(range(len(unique_depths)))
    ax.set_xticklabels(unique_depths)
    ax.grid(True, alpha=0.3)

    # Highlight best
    best_idx = np.argmin(depth_mapes)
    bars[best_idx].set_color('red')
    bars[best_idx].set_alpha(1.0)

    # 3. min_samples_split analysis
    ax = axes[0, 2]
    min_split_vals = [p.get('min_samples_split', 2) for p in all_params]
    unique_splits = sorted(set(min_split_vals))
    split_mapes = [np.mean([m for i, m in enumerate(val_mapes) if min_split_vals[i] == s])
                   for s in unique_splits]

    bars = ax.bar(range(len(unique_splits)), split_mapes, alpha=0.7)
    ax.set_xlabel('min_samples_split')
    ax.set_ylabel('Average Validation MAPE (%)')
    ax.set_title('MAPE vs min_samples_split')
    ax.set_xticks(range(len(unique_splits)))
    ax.set_xticklabels(unique_splits)
    ax.grid(True, alpha=0.3)

    # Highlight best
    best_idx = np.argmin(split_mapes)
    bars[best_idx].set_color('red')
    bars[best_idx].set_alpha(1.0)

    # 4. Parameters vs Performance Scatter
    ax = axes[1, 0]
    # Use n_estimators vs MAPE colored by training time
    n_est_numeric = [p.get('n_estimators', 100) for p in all_params]
    scatter = ax.scatter(n_est_numeric, val_mapes,
                         c=training_times, s=100, alpha=0.7, cmap='viridis')
    ax.set_xlabel('n_estimators')
    ax.set_ylabel('Validation MAPE (%)')
    ax.set_title('Performance vs Parameters')
    ax.grid(True, alpha=0.3)

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Training Time (s)')

    # 5. Parameter Distribution
    ax = axes[1, 1]
    param_counts = {}
    param_names = ['n_estimators', 'max_depth',
                   'min_samples_split', 'min_samples_leaf']

    for param in param_names:
        values = [str(p.get(param, 'N/A')) for p in all_params]
        param_counts[param] = len(set(values))

    bars = ax.bar(param_names, [param_counts[p]
                  for p in param_names], alpha=0.7)
    ax.set_xlabel('Parameters')
    ax.set_ylabel('Number of Unique Values Used')
    ax.set_title('Parameter Space Exploration')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right')

    # 6. Best Parameters Summary
    ax = axes[1, 2]
    ax.axis('off')

    # Find most common best parameters
    best_params_summary = {}
    for param in param_names:
        values = [p.get(param) for p in all_params if param in p]
        if values:
            most_common = max(set(values), key=values.count)
            count = values.count(most_common)
            best_params_summary[param] = (most_common, count)

    summary_text = "Best Hyperparameters Summary:\n"
    summary_text += "─" * 30 + "\n\n"

    for param, (value, count) in best_params_summary.items():
        percentage = (count / len(fold_metrics)) * 100
        summary_text += f"{param}:\n"
        summary_text += f"  Most common: {value}\n"
        summary_text += f"  Used in {count}/{len(fold_metrics)} folds ({percentage:.0f}%)\n\n"

    # Add performance info
    best_fold = min(fold_metrics, key=lambda x: x['val_mape'])
    summary_text += f"Best Overall Performance:\n"
    summary_text += f"  Fold {best_fold['fold']}: {best_fold['val_mape']:.2f}% MAPE\n"
    summary_text += f"  Parameters: {best_fold['best_params']}\n"

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightblue", alpha=0.8))

    plt.tight_layout()
    plt.savefig('rdf_hyperparameter_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Hyperparameter analysis saved: rdf_hyperparameter_analysis.png")


def create_detailed_metrics_analysis(fold_metrics, fold_data):
    """Create detailed metrics analysis plots"""
    print("\n📊 Creating detailed metrics analysis...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Random Forest Detailed Metrics Analysis',
                 fontsize=16, fontweight='bold')

    folds = [f['fold'] for f in fold_metrics]

    # 1. MAPE Comparison (Train vs Val)
    ax = axes[0, 0]
    train_mapes = [f['train_mape'] for f in fold_metrics]
    val_mapes = [f['val_mape'] for f in fold_metrics]

    x = np.arange(len(folds))
    width = 0.35

    bars1 = ax.bar(x - width/2, train_mapes, width,
                   label='Training', alpha=0.7, color='blue')
    bars2 = ax.bar(x + width/2, val_mapes, width,
                   label='Validation', alpha=0.7, color='orange')

    ax.set_xlabel('Fold')
    ax.set_ylabel('MAPE (%)')
    ax.set_title('MAPE Comparison Across Folds')
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}', ha='center', va='bottom', fontsize=8)

    # 2. R² Comparison (Train vs Val)
    ax = axes[0, 1]
    train_r2s = [f['train_r2'] for f in fold_metrics]
    val_r2s = [f['val_r2'] for f in fold_metrics]

    bars1 = ax.bar(x - width/2, train_r2s, width,
                   label='Training', alpha=0.7, color='green')
    bars2 = ax.bar(x + width/2, val_r2s, width,
                   label='Validation', alpha=0.7, color='red')

    ax.set_xlabel('Fold')
    ax.set_ylabel('R²')
    ax.set_title('R² Comparison Across Folds')
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)

    # 3. MAE Comparison (Train vs Val)
    ax = axes[0, 2]
    train_maes = [f['train_mae'] for f in fold_metrics]
    val_maes = [f['val_mae'] for f in fold_metrics]

    bars1 = ax.bar(x - width/2, train_maes, width,
                   label='Training', alpha=0.7, color='purple')
    bars2 = ax.bar(x + width/2, val_maes, width,
                   label='Validation', alpha=0.7, color='brown')

    ax.set_xlabel('Fold')
    ax.set_ylabel('MAE')
    ax.set_title('MAE Comparison Across Folds')
    ax.set_xticks(x)
    ax.set_xticklabels(folds)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Metrics Distribution
    ax = axes[1, 0]
    metrics_data = [val_mapes, val_r2s, val_maes]
    metrics_labels = ['Val MAPE (%)', 'Val R²', 'Val MAE']

    # Normalize metrics for comparison (0-1 scale)
    normalized_metrics = []
    for i, data in enumerate(metrics_data):
        if i == 0:  # MAPE - lower is better, normalize by dividing by max
            norm_data = np.array(data) / max(data)
        elif i == 1:  # R² - higher is better, already 0-1 scale
            norm_data = np.array(data)
        else:  # MAE - lower is better, normalize by dividing by max
            norm_data = np.array(data) / max(data)
        normalized_metrics.append(norm_data)

    bp = ax.boxplot(normalized_metrics,
                    labels=metrics_labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral']
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)

    ax.set_ylabel('Normalized Values')
    ax.set_title('Metrics Distribution Across Folds')
    ax.grid(True, alpha=0.3)

    # 5. Performance vs Training Time
    ax = axes[1, 1]
    training_times = [f['training_time'] for f in fold_metrics]

    # Create scatter plot with fold labels
    scatter = ax.scatter(training_times, val_mapes, s=100,
                         alpha=0.7, c=folds, cmap='viridis')

    # Add fold labels
    for i, (time, mape, fold) in enumerate(zip(training_times, val_mapes, folds)):
        ax.annotate(f'F{fold}', (time, mape), xytext=(
            5, 5), textcoords='offset points')

    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('Validation MAPE (%)')
    ax.set_title('Performance vs Training Time')
    ax.grid(True, alpha=0.3)

    # Add trend line
    z = np.polyfit(training_times, val_mapes, 1)
    p = np.poly1d(z)
    ax.plot(training_times, p(training_times), "r--",
            alpha=0.8, label=f'Trend (slope: {z[0]:.3f})')
    ax.legend()

    # 6. Best Fold Detailed Analysis
    ax = axes[1, 2]
    ax.axis('off')

    # Find best and worst performing folds
    best_fold = min(fold_metrics, key=lambda x: x['val_mape'])
    worst_fold = max(fold_metrics, key=lambda x: x['val_mape'])

    analysis_text = f"""
Detailed Fold Analysis:
─────────────────────────

🏆 BEST FOLD (Fold {best_fold['fold']}):
  • Val MAPE: {best_fold['val_mape']:.2f}%
  • Val R²: {best_fold['val_r2']:.4f}
  • Val MAE: {best_fold['val_mae']:.4f}
  • Training Time: {best_fold['training_time']:.1f}s
  • Best Params: {str(best_fold['best_params'])[:50]}...

🔻 WORST FOLD (Fold {worst_fold['fold']}):
  • Val MAPE: {worst_fold['val_mape']:.2f}%
  • Val R²: {worst_fold['val_r2']:.4f}
  • Val MAE: {worst_fold['val_mae']:.4f}
  • Training Time: {worst_fold['training_time']:.1f}s

📊 OVERALL STATISTICS:
  • MAPE Range: {min(val_mapes):.2f}% - {max(val_mapes):.2f}%
  • R² Range: {min(val_r2s):.4f} - {max(val_r2s):.4f}
  • Avg Training Time: {np.mean(training_times):.1f}s
  • Total Samples: {sum(f['train_size'] + f['val_size'] for f in fold_data)}
  
🎯 CONSISTENCY:
  • MAPE Std Dev: {np.std(val_mapes):.2f}%
  • R² Std Dev: {np.std(val_r2s):.4f}
  • Time Std Dev: {np.std(training_times):.1f}s
"""

    ax.text(0.05, 0.95, analysis_text, transform=ax.transAxes, fontsize=9,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig('rdf_detailed_metrics_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Detailed metrics analysis saved: rdf_detailed_metrics_analysis.png")


# Generate all visualizations
print("\n🎨 Generating comprehensive Random Forest visualizations...")
print("=" * 60)

try:
    create_evaluation_dashboard(
        fold_data, fold_metrics, trained_models, feature_cols, target_col)
    create_hyperparameter_analysis(fold_metrics, param_grid_hpc)
    create_detailed_metrics_analysis(fold_metrics, fold_data)

    print("\n✅ All visualizations created successfully!")
    print("📊 Generated files:")
    print("  • rdf_evaluation_dashboard.png - Main evaluation dashboard")
    print("  • rdf_hyperparameter_analysis.png - Hyperparameter analysis")
    print("  • rdf_detailed_metrics_analysis.png - Detailed metrics analysis")

except Exception as e:
    print(f"❌ Error creating visualizations: {e}")
    print("Creating fallback basic visualization...")

# Create fallback basic visualization if comprehensive fails
# Create visualization
plt.figure(figsize=(12, 8))

# Plot 1: MAPE by fold
plt.subplot(2, 2, 1)
folds = [f['fold'] for f in fold_metrics]
train_mapes = [f['train_mape'] for f in fold_metrics]
val_mapes = [f['val_mape'] for f in fold_metrics]
plt.plot(folds, train_mapes, 'o-', label='Training MAPE')
plt.plot(folds, val_mapes, 's-', label='Validation MAPE')
plt.xlabel('Fold')
plt.ylabel('MAPE (%)')
plt.title('MAPE by Fold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: R² by fold
plt.subplot(2, 2, 2)
train_r2s = [f['train_r2'] for f in fold_metrics]
val_r2s = [f['val_r2'] for f in fold_metrics]
plt.plot(folds, train_r2s, 'o-', label='Training R²')
plt.plot(folds, val_r2s, 's-', label='Validation R²')
plt.xlabel('Fold')
plt.ylabel('R²')
plt.title('R² by Fold')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Training time by fold
plt.subplot(2, 2, 3)
training_times = [f['training_time'] for f in fold_metrics]
plt.bar(folds, training_times, alpha=0.7)
plt.xlabel('Fold')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time by Fold')
plt.grid(True, alpha=0.3)

# Plot 4: Performance summary
plt.subplot(2, 2, 4)
metrics_names = ['Train MAPE', 'Val MAPE', 'Train R²', 'Val R²']
metrics_values = [avg_train_mape, avg_val_mape, avg_train_r2, avg_val_r2]
# Normalize for visualization
normalized_values = []
for i, val in enumerate(metrics_values):
    if 'MAPE' in metrics_names[i]:
        normalized_values.append(val / 100)  # Convert percentage to 0-1 scale
    else:
        normalized_values.append(val)

plt.bar(metrics_names, normalized_values, alpha=0.7)
plt.title('Average Performance Metrics')
plt.ylabel('Normalized Values')
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig('rdf_training_results.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Basic visualization saved to: rdf_training_results.png")

print(f"\n🎉 Random Forest Training Complete!")
print(
    f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print(f"📊 Best model performance: {best_fold['val_mape']:.2f}% MAPE")
print(f"🚀 Backend used: {'cuML (GPU)' if GPU_AVAILABLE else 'sklearn (CPU)'}")
print(f"Script completed at: {datetime.now()}")

if GPU_AVAILABLE:
    print(f"\n💡 GPU Performance Notes:")
    print(f"   - Used GPU-accelerated Random Forest from cuML")
    print(f"   - Data was processed on GPU memory")
    print(f"   - Training should be significantly faster than CPU-only")
else:
    print(f"\n💡 To enable GPU acceleration:")
    print(f"   - Install cuML: conda install -c rapidsai -c nvidia cuml")
    print(f"   - Install CuPy: pip install cupy-cuda11x (replace 11x with your CUDA version)")
    print(f"   - Ensure NVIDIA GPU with CUDA support is available")
