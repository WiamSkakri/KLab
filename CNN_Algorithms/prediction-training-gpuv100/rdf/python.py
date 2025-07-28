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

print("Starting Random Forest V100 GPU Training")
print("=" * 60)
print(f"Script started at: {datetime.now()}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Available files: {os.listdir('.')}")
print(f"GPU Acceleration: {'Enabled' if GPU_AVAILABLE else 'Disabled'}")

# Load data
print("\n📊 Loading V100 GPU experiment data...")
if not os.path.exists('combined_v100.csv'):
    print("Error: combined_v100.csv not found!")
    print("Make sure the V100 experiment data file is in the current directory.")
    sys.exit(1)

df = pd.read_csv('combined_v100.csv')
print(
    f"V100 data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# One-hot encode the Algorithm column if it exists
if 'Algorithm' in df.columns:
    df_encoded = pd.get_dummies(
        df, columns=['Algorithm'], prefix='Algorithm', dtype=int)
    df = df_encoded
    print("✅ Algorithm column one-hot encoded")

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
        X_train = X_gpu[train_idx]
        X_val = X_gpu[val_idx]
        y_train = y_gpu[train_idx]
        y_val = y_gpu[val_idx]
    else:
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

# Model Configuration
print(f"\n🌲 Random Forest Configuration for V100 GPU:")
print(f"Backend: {'cuML (GPU)' if GPU_AVAILABLE else 'sklearn (CPU)'}")

# Hyperparameter grid optimized for V100 (32GB VRAM)
if GPU_AVAILABLE:
    # V100-optimized parameters (reduced from L40s due to smaller VRAM)
    param_grid = {
        'n_estimators': [200, 400, 600],        # Reduced from L40s version
        'max_depth': [10, 15, 20, 25],          # Slightly reduced
        'max_features': [0.8, 0.9, 1.0],       # Same as L40s
        'min_samples_split': [2, 5, 10],       # Same as L40s
        'min_samples_leaf': [1, 2, 4],         # Same as L40s
        'max_samples': [0.8, 0.9, 1.0]         # Same as L40s
    }

    base_params = {
        'random_state': 42,
        'verbose': 1,
        'n_streams': 4  # V100 optimization
    }
else:
    # CPU parameters (fallback)
    param_grid = {
        'n_estimators': [200, 400, 600],
        'max_depth': [10, 15, 20],
        'max_features': ['sqrt', 'log2', None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_samples': [0.8, 0.9, 1.0]
    }

    base_params = {
        'random_state': 42,
        'verbose': 1,
        'n_jobs': -1
    }

total_combinations = np.prod([len(v) for v in param_grid.values()])
print(f"Parameter combinations: {total_combinations}")
print(f"Using GridSearchCV with 3-fold cross-validation")

# Evaluation Functions


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    if GPU_AVAILABLE:
        y_true_np = cp.asnumpy(y_true) if hasattr(
            y_true, 'get') else np.array(y_true)
        y_pred_np = cp.asnumpy(y_pred) if hasattr(
            y_pred, 'get') else np.array(y_pred)
    else:
        y_true_np = np.array(y_true)
        y_pred_np = np.array(y_pred)

    mask = y_true_np != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask])) * 100


def evaluate_rf(model, X, y):
    """Evaluate Random Forest model"""
    predictions = model.predict(X)

    if GPU_AVAILABLE:
        y_cpu = cp.asnumpy(y) if hasattr(y, 'get') else y
        pred_cpu = cp.asnumpy(predictions) if hasattr(
            predictions, 'get') else predictions
    else:
        y_cpu = y
        pred_cpu = predictions

    mape = calculate_mape(y_cpu, pred_cpu)
    mae = mean_absolute_error(y_cpu, pred_cpu)
    mse = mean_squared_error(y_cpu, pred_cpu)
    r2 = r2_score(y_cpu, pred_cpu)

    return {
        'mape': mape,
        'mae': mae,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2,
        'predictions': pred_cpu
    }


def print_metrics(metrics, title="Results"):
    print(f"\n{title}:")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")


# Main Training Loop
print("\n🚀 Starting V100 GPU Random Forest K-Fold Cross Validation Training")
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

    # Create Random Forest model
    if GPU_AVAILABLE:
        rf_model = cuRF(**base_params)
        # Use cuML's GridSearchCV for GPU acceleration
        grid_search = cuGridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=1  # cuML handles parallelization internally
        )
    else:
        rf_model = RandomForestRegressor(**base_params)
        grid_search = GridSearchCV(
            estimator=rf_model,
            param_grid=param_grid,
            cv=3,
            scoring='neg_mean_squared_error',
            verbose=1,
            n_jobs=-1
        )

    print(f"Running V100-optimized hyperparameter search...")

    # Fit the model
    fold_start_time = time.time()
    grid_search.fit(X_train, y_train)
    fold_training_time = time.time() - fold_start_time

    # Get the best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

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
    model_filename = f'rf_v100_model_fold_{fold}.joblib'
    joblib.dump(best_model, model_filename)
    print(f"Model saved: {model_filename}")

# Calculate total training time
total_time = time.time() - start_time

# Calculate average performance across all folds
print("\n" + "=" * 60)
print("🌲 V100 RANDOM FOREST CROSS VALIDATION SUMMARY")
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

print(
    f"Backend used: {'cuML (V100 GPU)' if GPU_AVAILABLE else 'sklearn (CPU)'}")
print(f"Average Train MAPE: {avg_train_mape:.2f}%")
print(f"Average Val MAPE:   {avg_val_mape:.2f}% ± {std_val_mape:.2f}%")
print(f"Average Train MAE:  {avg_train_mae:.4f}")
print(f"Average Val MAE:    {avg_val_mae:.4f} ± {std_val_mae:.4f}")
print(f"Average Train R²:   {avg_train_r2:.4f}")
print(f"Average Val R²:     {avg_val_r2:.4f} ± {std_val_r2:.4f}")
print(f"Average Training Time: {avg_training_time:.1f} seconds per fold")
print(
    f"Total Training Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

# Detailed results table
print(f"\nDetailed V100 Results by Fold:")
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

joblib.dump(best_model, 'best_rf_v100_model.joblib')
print(f"Best model saved as: best_rf_v100_model.joblib")

# Save results to CSV
results_df = pd.DataFrame(fold_metrics)
results_df.to_csv('rdf_v100_training_results.csv', index=False)
print(f"\nResults saved to: rdf_v100_training_results.csv")

# Create summary metrics
summary_metrics = {
    'backend': 'cuML (V100 GPU)' if GPU_AVAILABLE else 'sklearn (CPU)',
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
    'gpu_memory_gb': cp.cuda.runtime.memGetInfo()[1] / 1024**3 if GPU_AVAILABLE else 0,
    'timestamp': datetime.now().isoformat()
}

# Save summary metrics
summary_df = pd.DataFrame([summary_metrics])
summary_df.to_csv('rdf_v100_summary_metrics.csv', index=False)
print(f"Summary metrics saved to: rdf_v100_summary_metrics.csv")

# Create visualization
print("\n📊 Creating V100 training visualization...")

plt.figure(figsize=(15, 10))

# Plot 1: MAPE by fold
plt.subplot(2, 3, 1)
folds = [f['fold'] for f in fold_metrics]
train_mapes = [f['train_mape'] for f in fold_metrics]
val_mapes = [f['val_mape'] for f in fold_metrics]
plt.plot(folds, train_mapes, 'o-', label='Training MAPE', linewidth=2)
plt.plot(folds, val_mapes, 's-', label='Validation MAPE', linewidth=2)
plt.xlabel('Fold')
plt.ylabel('MAPE (%)')
plt.title('MAPE by Fold (V100 GPU)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: R² by fold
plt.subplot(2, 3, 2)
train_r2s = [f['train_r2'] for f in fold_metrics]
val_r2s = [f['val_r2'] for f in fold_metrics]
plt.plot(folds, train_r2s, 'o-', label='Training R²', linewidth=2)
plt.plot(folds, val_r2s, 's-', label='Validation R²', linewidth=2)
plt.xlabel('Fold')
plt.ylabel('R²')
plt.title('R² by Fold (V100 GPU)')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 3: Training time by fold
plt.subplot(2, 3, 3)
training_times = [f['training_time'] for f in fold_metrics]
bars = plt.bar(folds, training_times, alpha=0.7, color='green')
plt.xlabel('Fold')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time by Fold (V100)')
plt.grid(True, alpha=0.3)

# Add value labels
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{height:.1f}s', ha='center', va='bottom')

# Plot 4: Performance comparison
plt.subplot(2, 3, 4)
metrics_names = ['Train MAPE', 'Val MAPE', 'Train R²', 'Val R²']
metrics_values = [avg_train_mape, avg_val_mape, avg_train_r2, avg_val_r2]
colors = ['lightblue', 'lightcoral', 'lightgreen', 'lightyellow']
bars = plt.bar(metrics_names, metrics_values, color=colors, alpha=0.7)
plt.title('Average Performance Metrics (V100)')
plt.ylabel('Values')
plt.xticks(rotation=45)

# Add value labels
for bar, value in zip(bars, metrics_values):
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2., height,
             f'{value:.3f}', ha='center', va='bottom')

# Plot 5: Model complexity analysis
plt.subplot(2, 3, 5)
n_estimators_values = [f['best_params'].get(
    'n_estimators', 0) for f in fold_metrics]
max_depth_values = [f['best_params'].get('max_depth', 0) for f in fold_metrics]

plt.scatter(n_estimators_values, max_depth_values,
            c=val_mapes, cmap='viridis', s=100, alpha=0.7)
plt.colorbar(label='Validation MAPE (%)')
plt.xlabel('N Estimators')
plt.ylabel('Max Depth')
plt.title('Model Complexity vs Performance (V100)')
plt.grid(True, alpha=0.3)

# Plot 6: Summary text
plt.subplot(2, 3, 6)
plt.axis('off')
summary_text = f"""
V100 GPU TRAINING SUMMARY
{'='*25}

🚀 Hardware: Tesla V100 (32GB)
📊 Backend: {'cuML (GPU)' if GPU_AVAILABLE else 'sklearn (CPU)'}

⏱️ Performance:
• Total Time: {total_time:.1f}s ({total_time/60:.1f} min)
• Avg per fold: {avg_training_time:.1f}s
• Best Val MAPE: {min(val_mapes):.2f}%
• Avg Val R²: {avg_val_r2:.4f}

🎯 Best Model (Fold {best_fold['fold']}):
• MAPE: {best_fold['val_mape']:.2f}%
• R²: {best_fold['val_r2']:.4f}
• Time: {best_fold['training_time']:.1f}s

📈 Grid Search:
• {total_combinations} combinations tested
• 3-fold CV per combination
• V100-optimized parameters
"""

plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

plt.tight_layout()
plt.savefig('rdf_v100_training_results.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"V100 visualization saved to: rdf_v100_training_results.png")

print(f"\n🎉 V100 Random Forest Training Complete!")
print(
    f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print(f"📊 Best model performance: {best_fold['val_mape']:.2f}% MAPE")
print(
    f"🚀 Backend used: {'cuML (V100 GPU)' if GPU_AVAILABLE else 'sklearn (CPU)'}")
print(f"Script completed at: {datetime.now()}")

if GPU_AVAILABLE:
    print(
        f"💾 Final GPU memory usage: {cp.cuda.runtime.memGetInfo()[0] / 1024**3:.1f} GB")
