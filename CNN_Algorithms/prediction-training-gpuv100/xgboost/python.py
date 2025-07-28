import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Check for XGBoost GPU availability
try:
    import xgboost as xgb
    print("✅ XGBoost detected")

    # Check for V100 GPU support
    try:
        # Test GPU availability by creating a small DMatrix and training
        test_data = xgb.DMatrix(np.random.rand(
            10, 5), label=np.random.rand(10))
        # Modern XGBoost GPU syntax
        test_params = {'tree_method': 'hist', 'device': 'cuda'}
        xgb.train(test_params, test_data,
                  num_boost_round=1, verbose_eval=False)
        GPU_AVAILABLE = True
        print("🚀 XGBoost V100 GPU support detected - will use GPU acceleration")
    except Exception as e:
        print(f"⚠️  XGBoost GPU not available: {e}")
        print("📌 Falling back to CPU-based XGBoost")
        GPU_AVAILABLE = False

except ImportError as e:
    print(f"❌ XGBoost not installed: {e}")
    print("Please install XGBoost: pip install xgboost")
    sys.exit(1)

# Standard imports
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# For saving models
import joblib

print("🔧 XGBoost build info:")
print(f"   Version: {xgb.__version__}")
if GPU_AVAILABLE:
    # Try to get GPU name
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"   GPU detected: {gpu_name}")
    except:
        print("   GPU detected: Available")

print("Starting V100 GPU XGBoost HPC Training")
print("=" * 60)
print(f"Script started at: {datetime.now()}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Available files: {os.listdir('.')}")
print(f"XGBoost version: {xgb.__version__}")
print(f"GPU Acceleration: {'Enabled' if GPU_AVAILABLE else 'Disabled'}")

# Load data
print("\n📊 Loading V100 GPU experiment data...")
if not os.path.exists('combined_v100.csv'):
    print("Error: combined_v100.csv not found!")
    print("Available files in current directory:")
    for f in os.listdir('.'):
        print(f"  - {f}")
    sys.exit(1)

df = pd.read_csv('combined_v100.csv')
print(
    f"V100 data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
print(f"Columns: {list(df.columns)}")

# One-hot encode the Algorithm column
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

# Enhanced parameter space for V100 GPU
if GPU_AVAILABLE:
    # GPU parameters - optimized for V100 with 32GB VRAM (reduced from L40s)
    param_space = {
        'n_estimators': [200, 400, 600, 800],       # Reduced from L40s version
        'max_depth': [8, 12, 16, 20],              # Slightly reduced
        'learning_rate': [0.01, 0.05, 0.1, 0.2],  # Same as L40s
        'subsample': [0.8, 0.9, 1.0],             # Same as L40s
        'colsample_bytree': [0.8, 0.9, 1.0],      # Same as L40s
        'colsample_bylevel': [0.8, 0.9, 1.0],     # Same as L40s
        'reg_alpha': [0, 0.1, 0.5],               # Same as L40s
        'reg_lambda': [1, 2, 5],                  # Same as L40s
        'min_child_weight': [1, 3, 5],            # Same as L40s
        'gamma': [0, 0.1, 0.5]                    # Same as L40s
    }

    base_params = {
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42,
        'verbosity': 1,
        'max_bin': 256  # Reduced from L40s for V100 compatibility
    }
else:
    # CPU parameters
    param_space = {
        'n_estimators': [200, 400, 600],
        'max_depth': [8, 12, 16],
        'learning_rate': [0.01, 0.05, 0.1],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 2, 5],
        'min_child_weight': [1, 3, 5]
    }

    base_params = {
        'tree_method': 'hist',
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42,
        'verbosity': 1,
        'n_jobs': -1
    }

print(f"\n🚀 XGBoost Configuration for V100 GPU:")
print(f"Backend: {'GPU (CUDA)' if GPU_AVAILABLE else 'CPU'}")
print(
    f"Parameter space combinations: {np.prod([len(v) for v in param_space.values()])}")
print(f"Using RandomizedSearchCV with 100 iterations")

# Evaluation Functions


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    mask = y_true_np != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true_np[mask] - y_pred_np[mask]) / y_true_np[mask])) * 100


def evaluate_xgb(model, X, y):
    """Evaluate XGBoost model"""
    predictions = model.predict(X)

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
print("\n🚀 Starting XGBoost V100 GPU K-Fold Cross Validation Training")
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

    print(f"\n🚀 FOLD {fold}")
    print("-" * 20)
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")

    # Create DMatrix for XGBoost
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    print("Running V100 GPU hyperparameter search...")

    # Create XGBoost regressor
    xgb_model = xgb.XGBRegressor(**base_params)

    # Randomized search for efficiency on V100
    search = RandomizedSearchCV(
        estimator=xgb_model,
        param_distributions=param_space,
        n_iter=min(100, np.prod([len(v) for v in param_space.values()])),
        cv=3,
        scoring='neg_mean_squared_error',
        random_state=42,
        n_jobs=1,  # XGBoost handles parallelization internally
        verbose=1
    )

    # Fit the model
    fold_start_time = time.time()
    search.fit(X_train, y_train)
    fold_training_time = time.time() - fold_start_time

    # Get the best model
    best_model = search.best_estimator_
    best_params = search.best_params_
    best_score = search.best_score_

    print(f"✅ Training complete in {fold_training_time:.1f} seconds")
    print(f"Best parameters: {best_params}")
    print(f"Best CV score (neg MSE): {best_score:.4f}")

    # Evaluate on training and validation sets
    train_metrics = evaluate_xgb(best_model, X_train, y_train)
    val_metrics = evaluate_xgb(best_model, X_val, y_val)

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
    model_filename = f'xgb_v100_model_fold_{fold}.joblib'
    joblib.dump(best_model, model_filename)
    print(f"Model saved: {model_filename}")

# Calculate total training time
total_time = time.time() - start_time

# Calculate average performance across all folds
print("\n" + "=" * 60)
print("🌲 V100 XGBOOST CROSS VALIDATION SUMMARY")
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
    f"Backend Used: {'XGBoost (V100 GPU)' if GPU_AVAILABLE else 'XGBoost (CPU)'}")
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

joblib.dump(best_model, 'best_xgb_v100_model.joblib')
print(f"Best model saved as: best_xgb_v100_model.joblib")

# Most common hyperparameters
print(f"\nMost common best parameters:")
all_params = [f['best_params'] for f in fold_metrics]
param_summary = {}
param_names = ['n_estimators', 'max_depth',
               'learning_rate', 'subsample', 'colsample_bytree']
for param in param_names:
    values = [p.get(param) for p in all_params if param in p]
    if values:
        most_common = max(set(values), key=values.count)
        count = values.count(most_common)
        param_summary[param] = most_common
        print(f"  {param}: {most_common} (appeared in {count}/{len(values)} folds)")

# Save results to CSV
results_df = pd.DataFrame(fold_metrics)
results_df.to_csv('xgb_v100_training_results.csv', index=False)
print(f"\nResults saved to: xgb_v100_training_results.csv")

# Create summary metrics
summary_metrics = {
    'backend': 'XGBoost (V100 GPU)' if GPU_AVAILABLE else 'XGBoost (CPU)',
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
summary_df.to_csv('xgb_v100_summary_metrics.csv', index=False)
print(f"Summary metrics saved to: xgb_v100_summary_metrics.csv")

# Create visualization
print("\n📊 Creating V100 XGBoost training visualization...")

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
param_count = np.prod([len(v) for v in param_space.values()])
summary_text = f"""
V100 GPU XGBOOST SUMMARY
{'='*25}

🚀 Hardware: Tesla V100 (32GB)
📊 Backend: {'XGBoost (GPU)' if GPU_AVAILABLE else 'XGBoost (CPU)'}

⏱️ Performance:
• Total Time: {total_time:.1f}s ({total_time/60:.1f} min)
• Avg per fold: {avg_training_time:.1f}s
• Best Val MAPE: {min(val_mapes):.2f}%
• Avg Val R²: {avg_val_r2:.4f}

🎯 Best Model (Fold {best_fold['fold']}):
• MAPE: {best_fold['val_mape']:.2f}%
• R²: {best_fold['val_r2']:.4f}
• Time: {best_fold['training_time']:.1f}s

📈 Hyperparameter Search:
• {param_count} total combinations
• 100 RandomizedSearchCV iterations
• V100-optimized parameters
• 3-fold CV per iteration
"""

plt.text(0.05, 0.95, summary_text, transform=plt.gca().transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

plt.tight_layout()
plt.savefig('xgb_v100_training_results.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"V100 visualization saved to: xgb_v100_training_results.png")

print(f"\n🎉 V100 XGBoost Training Complete!")
print(
    f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print(f"📊 Best model performance: {best_fold['val_mape']:.2f}% MAPE")
print(
    f"🚀 Backend used: {'XGBoost (V100 GPU)' if GPU_AVAILABLE else 'XGBoost (CPU)'}")
print(f"Script completed at: {datetime.now()}")
