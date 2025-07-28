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

    # Check for L40s GPU support
    try:
        # Test GPU availability by creating a small DMatrix and training
        test_data = xgb.DMatrix(np.random.rand(
            10, 5), label=np.random.rand(10))
        # Modern XGBoost GPU syntax
        test_params = {'tree_method': 'hist', 'device': 'cuda'}
        xgb.train(test_params, test_data,
                  num_boost_round=1, verbose_eval=False)
        GPU_AVAILABLE = True
        print("🚀 XGBoost L40s GPU support detected - will use GPU acceleration")
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

# Check CUDA availability
if GPU_AVAILABLE:
    try:
        print(f"🔧 XGBoost build info:")
        print(f"   Version: {xgb.__version__}")
        # Check if we can access CUDA device info
        import subprocess
        try:
            result = subprocess.run(['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                                    capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                gpu_name = result.stdout.strip()
                print(f"   GPU detected: {gpu_name}")
        except:
            print("   GPU info not available via nvidia-smi")
    except Exception as e:
        print(f"   Error getting XGBoost info: {e}")

print("Starting L40s GPU XGBoost HPC Training")
print("=" * 60)
print(f"Script started at: {datetime.now()}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Available files: {os.listdir('.')}")
print(f"XGBoost version: {xgb.__version__}")
print(f"GPU Acceleration: {'Enabled' if GPU_AVAILABLE else 'Disabled'}")

# Load L40s data
print("\n📊 Loading L40s GPU experiment data...")
if not os.path.exists('combined_l40s.csv'):
    print("Error: combined_l40s.csv not found!")
    print("Available files in current directory:")
    for f in os.listdir('.'):
        print(f"  - {f}")
    print("Make sure the L40s experiment data file is in the current directory.")
    sys.exit(1)

df = pd.read_csv('combined_l40s.csv')
print(
    f"L40s data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")
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

# XGBoost Model and Hyperparameters - Optimized for L40s GPU
print(f"\n🚀 XGBoost Configuration for L40s GPU:")
print(f"Backend: {'GPU (CUDA)' if GPU_AVAILABLE else 'CPU'}")

# Enhanced parameter space for L40s GPU
if GPU_AVAILABLE:
    # GPU parameters - optimized for L40s with 48GB VRAM
    param_space = {
        'n_estimators': [300, 500, 800, 1000],
        'max_depth': [8, 12, 16, 20],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'subsample': [0.8, 0.9, 1.0],
        'colsample_bytree': [0.8, 0.9, 1.0],
        'colsample_bylevel': [0.8, 0.9, 1.0],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [1, 2, 5],
        'min_child_weight': [1, 3, 5],
        'gamma': [0, 0.1, 0.5]
    }

    base_params = {
        'tree_method': 'hist',
        'device': 'cuda',
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42,
        'verbosity': 1,
        'max_bin': 512  # Optimal for GPU
    }
else:
    # CPU parameters
    param_space = {
        'n_estimators': [300, 500, 800],
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

print(
    f"Parameter space combinations: {np.prod([len(v) for v in param_space.values()])}")
print(
    f"Using RandomizedSearchCV with {min(100, np.prod([len(v) for v in param_space.values()]))} iterations")

# Evaluation Functions


def calculate_mape(y_true, y_pred):
    y_true_np = np.array(y_true)
    y_pred_np = np.array(y_pred)
    mask = y_true_np != 0
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
print("\n🚀 Starting XGBoost L40s GPU K-Fold Cross Validation Training")
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

    print("Running L40s GPU hyperparameter search...")

    # Create XGBoost regressor
    xgb_model = xgb.XGBRegressor(**base_params)

    # Randomized search for efficiency on L40s
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
    print(f"🚀 Fold {fold} Results:")
    print_metrics(train_metrics, f"  Training")
    print_metrics(val_metrics, f"  Validation")

    # Save model for this fold
    model_filename = f'xgb_l40s_model_fold_{fold}.joblib'
    joblib.dump(best_model, model_filename)
    print(f"Model saved: {model_filename}")

# Calculate total training time
total_time = time.time() - start_time

# Calculate average performance across all folds
print("\n" + "=" * 60)
print("🚀 XGBOOST L40s GPU CROSS VALIDATION SUMMARY")
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

print(f"Hardware: L40s GPU")
print(
    f"Backend Used: {'XGBoost GPU (CUDA)' if GPU_AVAILABLE else 'XGBoost CPU'}")
print(f"Dataset: combined_l40s.csv")
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

joblib.dump(best_model, 'best_xgb_l40s_model.joblib')
print(f"Best L40s XGBoost model saved as: best_xgb_l40s_model.joblib")

# Most common hyperparameters
print(f"\nMost common best parameters for L40s XGBoost:")
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
results_df.to_csv('xgb_l40s_training_results.csv', index=False)
print(f"\nResults saved to: xgb_l40s_training_results.csv")

# Create summary metrics
summary_metrics = {
    'hardware': 'L40s GPU',
    'backend': 'XGBoost GPU (CUDA)' if GPU_AVAILABLE else 'XGBoost CPU',
    'dataset': 'combined_l40s.csv',
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
summary_df.to_csv('xgb_l40s_summary_metrics.csv', index=False)
print(f"Summary metrics saved to: xgb_l40s_summary_metrics.csv")

# ===============================
# L40s GPU XGBOOST VISUALIZATION
# ===============================


def create_l40s_xgb_dashboard(fold_data, fold_metrics, trained_models, feature_cols, target_col):
    """Create comprehensive XGBoost evaluation dashboard for L40s GPU"""
    print("\n📊 Creating L40s GPU XGBoost evaluation dashboard...")

    # Collect all predictions and actuals for overall analysis
    all_predictions = []
    all_actuals = []

    for i, fold_info in enumerate(fold_data):
        val_pred = trained_models[i].predict(fold_info['X_val'])
        val_actual = fold_info['y_val']

        all_predictions.extend(val_pred)
        all_actuals.extend(val_actual)

    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    # Create main dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('XGBoost L40s GPU Model Evaluation Dashboard',
                 fontsize=16, fontweight='bold')

    # 1. Predictions vs Actual
    ax = axes[0, 0]
    ax.scatter(all_actuals, all_predictions, alpha=0.6, s=20)
    min_val, max_val = min(all_actuals.min(), all_predictions.min()), max(
        all_actuals.max(), all_predictions.max())
    ax.plot([min_val, max_val], [min_val, max_val],
            'r--', lw=2, label='Perfect Prediction')
    ax.set_xlabel('Actual Execution Time (ms)')
    ax.set_ylabel('Predicted Execution Time (ms)')
    ax.set_title('L40s XGBoost: Predictions vs Actual')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add R² annotation
    r2_overall = r2_score(all_actuals, all_predictions)
    ax.text(0.05, 0.95, f'R² = {r2_overall:.4f}', transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    # 2. Cross-Validation Results
    ax = axes[0, 1]
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
    ax.set_title('L40s XGBoost: Cross-Validation Performance')
    ax.grid(True, alpha=0.3)

    # Highlight best fold
    best_fold_num = min(fold_metrics, key=lambda x: x['val_mape'])['fold']
    ax.axvline(x=best_fold_num, color='red', linestyle='--',
               alpha=0.7, label=f'Best Fold ({best_fold_num})')
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 3. Performance Metrics Comparison
    ax = axes[0, 2]
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
    ax.set_title('L40s XGBoost: Average Performance Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(metrics_names)
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Training Time Analysis
    ax = axes[1, 0]
    training_times = [f['training_time'] for f in fold_metrics]
    bars = ax.bar(folds, training_times, alpha=0.7, color='purple')
    ax.set_xlabel('Fold')
    ax.set_ylabel('Training Time (seconds)')
    ax.set_title('L40s XGBoost: Training Time by Fold')
    ax.grid(True, alpha=0.3)

    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}s', ha='center', va='bottom')

    # Add average line
    avg_time = np.mean(training_times)
    ax.axhline(y=avg_time, color='red', linestyle='--',
               alpha=0.7, label=f'Average: {avg_time:.1f}s')
    ax.legend()

    # 5. Model Summary Statistics
    ax = axes[1, 1]
    ax.axis('off')

    summary_text = f"""
L40s XGBoost Model Summary:
──────────────────────────
Hardware: L40s GPU
Backend: {'XGBoost GPU (CUDA)' if GPU_AVAILABLE else 'XGBoost CPU'}
Dataset: combined_l40s.csv
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

    # 6. Feature Importance
    ax = axes[1, 2]
    try:
        best_model_idx = min(range(len(fold_metrics)),
                             key=lambda i: fold_metrics[i]['val_mape'])
        best_model = trained_models[best_model_idx]

        if hasattr(best_model, 'feature_importances_'):
            feature_importance = best_model.feature_importances_

            # Get top 10 features
            top_indices = np.argsort(feature_importance)[-10:]
            top_features = [feature_cols[i] for i in top_indices]
            top_importance = feature_importance[top_indices]

            bars = ax.barh(range(len(top_features)), top_importance, alpha=0.7)
            ax.set_yticks(range(len(top_features)))
            ax.set_yticklabels(top_features)
            ax.set_xlabel('Feature Importance')
            ax.set_title('L40s XGBoost: Top 10 Feature Importance')
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'Feature importance\nnot available',
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title('Feature Importance (N/A)')
    except Exception as e:
        ax.text(
            0.5, 0.5, f'Feature importance\nerror: {str(e)[:30]}...', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('Feature Importance (Error)')

    plt.tight_layout()
    plt.savefig('xgb_l40s_evaluation_dashboard.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ L40s XGBoost evaluation dashboard saved: xgb_l40s_evaluation_dashboard.png")


# Generate L40s XGBoost visualizations
print("\n🎨 Generating L40s GPU XGBoost visualizations...")
print("=" * 60)

try:
    create_l40s_xgb_dashboard(fold_data, fold_metrics,
                              trained_models, feature_cols, target_col)
    print("\n✅ L40s XGBoost visualizations created successfully!")
    print("📊 Generated files:")
    print("  • xgb_l40s_evaluation_dashboard.png - L40s XGBoost evaluation dashboard")

except Exception as e:
    print(f"❌ Error creating L40s XGBoost visualizations: {e}")
    print("Creating fallback basic visualization...")

    # Create fallback basic visualization
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
    plt.title('L40s XGBoost: MAPE by Fold')
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
    plt.title('L40s XGBoost: R² by Fold')
    plt.legend()
    plt.grid(True, alpha=0.3)

    # Plot 3: Training time by fold
    plt.subplot(2, 2, 3)
    training_times = [f['training_time'] for f in fold_metrics]
    plt.bar(folds, training_times, alpha=0.7)
    plt.xlabel('Fold')
    plt.ylabel('Training Time (seconds)')
    plt.title('L40s XGBoost: Training Time by Fold')
    plt.grid(True, alpha=0.3)

    # Plot 4: Performance summary
    plt.subplot(2, 2, 4)
    metrics_names = ['Train MAPE', 'Val MAPE', 'Train R²', 'Val R²']
    metrics_values = [avg_train_mape, avg_val_mape, avg_train_r2, avg_val_r2]
    # Normalize for visualization
    normalized_values = []
    for i, val in enumerate(metrics_values):
        if 'MAPE' in metrics_names[i]:
            # Convert percentage to 0-1 scale
            normalized_values.append(val / 100)
        else:
            normalized_values.append(val)

    plt.bar(metrics_names, normalized_values, alpha=0.7)
    plt.title('L40s XGBoost: Average Performance Metrics')
    plt.ylabel('Normalized Values')
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig('xgb_l40s_training_results.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"L40s XGBoost basic visualization saved to: xgb_l40s_training_results.png")

print(f"\n🎉 L40s GPU XGBoost Training Complete!")
print(
    f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print(f"📊 Best model performance: {best_fold['val_mape']:.2f}% MAPE")
print(f"🚀 Hardware: L40s GPU")
print(
    f"🚀 Backend used: {'XGBoost GPU (CUDA)' if GPU_AVAILABLE else 'XGBoost CPU'}")
print(f"📁 Dataset: combined_l40s.csv")
print(f"Script completed at: {datetime.now()}")

if GPU_AVAILABLE:
    print(f"\n💡 L40s GPU XGBoost Performance Notes:")
    print(f"   - Used GPU-accelerated XGBoost with CUDA")
    print(f"   - Tree method: hist (optimized for L40s)")
    print(f"   - Device: cuda:0 (L40s GPU)")
    print(f"   - Enhanced parameter space for GPU optimization")
else:
    print(f"\n💡 To enable L40s GPU acceleration:")
    print(f"   - Install XGBoost with GPU support: pip install xgboost[gpu]")
    print(f"   - Ensure L40s GPU with CUDA support is available")
    print(f"   - Load appropriate CUDA modules on HPC")
