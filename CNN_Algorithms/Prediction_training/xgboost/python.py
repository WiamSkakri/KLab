import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Check for GPU availability
try:
    import xgboost as xgb
    print("✅ XGBoost detected")

    # Check for GPU support
    try:
        # Test GPU availability by creating a small DMatrix and training
        test_data = xgb.DMatrix(np.random.rand(
            10, 5), label=np.random.rand(10))
        test_params = {'tree_method': 'gpu_hist', 'gpu_id': 0}
        xgb.train(test_params, test_data,
                  num_boost_round=1, verbose_eval=False)
        GPU_AVAILABLE = True
        print("🚀 XGBoost GPU support detected - will use GPU acceleration")
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
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# For saving models
import joblib

print("Starting XGBoost HPC Training")
print("=" * 60)
print(f"Script started at: {datetime.now()}")
print(f"Python version: {sys.version}")
print(f"Current working directory: {os.getcwd()}")
print(f"Available files: {os.listdir('.')}")
print(f"XGBoost version: {xgb.__version__}")
print(f"GPU Acceleration: {'Enabled' if GPU_AVAILABLE else 'Disabled'}")

# Load data
print("\n📊 Loading data...")
if not os.path.exists('combined.csv'):
    print("Error: combined.csv not found!")
    print("Available files in current directory:")
    for f in os.listdir('.'):
        print(f"  - {f}")
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

# XGBoost Model and Hyperparameters
print(f"\n🌲 XGBoost Configuration:")
print(f"Backend: {'GPU-accelerated' if GPU_AVAILABLE else 'CPU-only'}")

# Base parameters
base_params = {
    'objective': 'reg:squarederror',
    'eval_metric': 'mae',
    'random_state': 42,
    'verbosity': 0
}

# Add GPU-specific parameters if available
if GPU_AVAILABLE:
    base_params.update({
        'tree_method': 'gpu_hist',
        'gpu_id': 0
    })
else:
    base_params.update({
        'tree_method': 'hist',
        'n_jobs': -1
    })

# Hyperparameter grid - optimized for both GPU and CPU
param_grid_hpc = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 6, 10, 15],
    'learning_rate': [0.01, 0.1, 0.2, 0.3],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0],
    'reg_alpha': [0, 0.1, 1],
    'reg_lambda': [1, 1.5, 2]
}

print(
    f"Hyperparameter grid combinations: {np.prod([len(v) for v in param_grid_hpc.values()])}")


def create_xgb_with_gridsearch(cv_folds=3):
    """Create XGBoost model with GridSearchCV"""

    # Create XGBoost regressor
    xgb_model = xgb.XGBRegressor(**base_params)

    # GridSearchCV
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid_hpc,
        cv=cv_folds,
        scoring='neg_mean_absolute_error',
        n_jobs=1 if GPU_AVAILABLE else -1,  # Use single job for GPU to avoid conflicts
        verbose=1
    )

    return grid_search

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
print("\n🚀 Starting XGBoost K-Fold Cross Validation Training")
print("=" * 60)

# Store results from all folds
fold_metrics = []
trained_models = []
training_histories = []  # Store training progress for each fold

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

    # Create XGBoost with GridSearchCV
    print("Running hyperparameter search...")
    xgb_grid = create_xgb_with_gridsearch(cv_folds=3)

    # Fit the model
    fold_start_time = time.time()
    xgb_grid.fit(X_train, y_train)
    fold_training_time = time.time() - fold_start_time

    # Get the best model
    best_model = xgb_grid.best_estimator_
    best_params = xgb_grid.best_params_
    best_score = xgb_grid.best_score_

    print(f"✅ Training complete in {fold_training_time:.1f} seconds")
    print(f"Best parameters: {best_params}")
    print(f"Best CV score (neg MAE): {best_score:.4f}")

    # Train final model with progress tracking for learning curves
    print("Training final model with progress tracking...")
    final_model = xgb.XGBRegressor(**{**base_params, **best_params})

    # Set up evaluation sets for progress tracking
    eval_set = [(X_train, y_train), (X_val, y_val)]
    eval_names = ['train', 'validation']

    # Train with evaluation tracking
    final_model.fit(
        X_train, y_train,
        eval_set=eval_set,
        verbose=False
    )

    # Get training history
    training_history = {
        'fold': fold,
        'evals_result': final_model.evals_result(),
        'best_params': best_params
    }
    training_histories.append(training_history)

    # Use the final model for evaluation
    best_model = final_model

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
    model_filename = f'xgb_model_fold_{fold}.joblib'
    joblib.dump(best_model, model_filename)
    print(f"Model saved: {model_filename}")

# Calculate total training time
total_time = time.time() - start_time

# Calculate average performance across all folds
print("\n" + "=" * 60)
print("🌲 XGBOOST CROSS VALIDATION SUMMARY")
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

print(f"Backend Used: {'XGBoost (GPU)' if GPU_AVAILABLE else 'XGBoost (CPU)'}")
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

joblib.dump(best_model, 'best_xgb_model.joblib')
print(f"Best model saved as: best_xgb_model.joblib")

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
results_df.to_csv('xgb_training_results.csv', index=False)
print(f"\nResults saved to: xgb_training_results.csv")

# Create summary metrics
summary_metrics = {
    'backend': 'XGBoost (GPU)' if GPU_AVAILABLE else 'XGBoost (CPU)',
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
summary_df.to_csv('xgb_summary_metrics.csv', index=False)
print(f"Summary metrics saved to: xgb_summary_metrics.csv")

# ===============================
# COMPREHENSIVE VISUALIZATION
# ===============================


def create_evaluation_dashboard(fold_data, fold_metrics, trained_models, feature_cols, target_col, training_histories):
    """Create comprehensive XGBoost evaluation dashboard"""
    print("\n📊 Creating XGBoost evaluation dashboard...")

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

        all_predictions.extend(val_pred)
        all_actuals.extend(val_actual)
        all_fold_labels.extend([f'Fold {i+1}'] * len(val_pred))

    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    # Create main dashboard
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle('XGBoost Model Evaluation Dashboard',
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

    # 7. Training Progress Curves (Boosting Rounds)
    ax = axes[2, 0]

    # Plot training progress for each fold
    colors = plt.cm.tab10(np.linspace(0, 1, len(training_histories)))

    for i, history in enumerate(training_histories):
        fold_num = history['fold']
        evals_result = history['evals_result']

        if 'validation_0' in evals_result and 'validation_1' in evals_result:
            # Get training and validation errors
            train_errors = evals_result['validation_0']['mae']
            val_errors = evals_result['validation_1']['mae']
            rounds = range(len(train_errors))

            # Plot training curve (lighter color)
            ax.plot(rounds, train_errors, '--', color=colors[i], alpha=0.6,
                    label=f'Fold {fold_num} Train' if i == 0 else '')

            # Plot validation curve (solid line)
            ax.plot(rounds, val_errors, '-', color=colors[i], alpha=0.8,
                    label=f'Fold {fold_num} Val' if i == 0 else '')

    ax.set_xlabel('Boosting Rounds')
    ax.set_ylabel('MAE')
    ax.set_title('Training Progress Curves')
    ax.grid(True, alpha=0.3)

    # Add legend for first fold only to avoid clutter
    ax.legend(['Training', 'Validation'], loc='upper right')

    # Highlight convergence
    ax.text(0.05, 0.95, f'{len(training_histories)} folds shown',
            transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

    # 8. Model Summary Statistics
    ax = axes[2, 1]
    ax.axis('off')

    summary_text = f"""
Model Summary Statistics:
─────────────────────────
Backend: {'XGBoost (GPU)' if GPU_AVAILABLE else 'XGBoost (CPU)'}
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
    plt.savefig('xgb_evaluation_dashboard.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Evaluation dashboard saved: xgb_evaluation_dashboard.png")


def create_hyperparameter_analysis(fold_metrics, param_grid_hpc):
    """Create hyperparameter analysis plots"""
    print("\n📊 Creating hyperparameter analysis...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('XGBoost Hyperparameter Analysis',
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
    max_depth_vals = [p.get('max_depth', 6) for p in all_params]
    unique_depths = sorted(set(max_depth_vals))
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

    # 3. learning_rate analysis
    ax = axes[0, 2]
    lr_vals = [p.get('learning_rate', 0.1) for p in all_params]
    unique_lrs = sorted(set(lr_vals))
    lr_mapes = [np.mean([m for i, m in enumerate(val_mapes) if lr_vals[i] == lr])
                for lr in unique_lrs]

    bars = ax.bar(range(len(unique_lrs)), lr_mapes, alpha=0.7)
    ax.set_xlabel('learning_rate')
    ax.set_ylabel('Average Validation MAPE (%)')
    ax.set_title('MAPE vs learning_rate')
    ax.set_xticks(range(len(unique_lrs)))
    ax.set_xticklabels(unique_lrs)
    ax.grid(True, alpha=0.3)

    # Highlight best
    best_idx = np.argmin(lr_mapes)
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
                   'learning_rate', 'subsample', 'colsample_bytree']

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
    plt.savefig('xgb_hyperparameter_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Hyperparameter analysis saved: xgb_hyperparameter_analysis.png")


def create_detailed_metrics_analysis(fold_metrics, fold_data):
    """Create detailed metrics analysis plots"""
    print("\n📊 Creating detailed metrics analysis...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('XGBoost Detailed Metrics Analysis',
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
    plt.savefig('xgb_detailed_metrics_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Detailed metrics analysis saved: xgb_detailed_metrics_analysis.png")


# Generate all visualizations
print("\n🎨 Generating comprehensive XGBoost visualizations...")
print("=" * 60)

try:
    create_evaluation_dashboard(
        fold_data, fold_metrics, trained_models, feature_cols, target_col, training_histories)
    create_hyperparameter_analysis(fold_metrics, param_grid_hpc)
    create_detailed_metrics_analysis(fold_metrics, fold_data)

    print("\n✅ All visualizations created successfully!")
    print("📊 Generated files:")
    print("  • xgb_evaluation_dashboard.png - Main evaluation dashboard")
    print("  • xgb_hyperparameter_analysis.png - Hyperparameter analysis")
    print("  • xgb_detailed_metrics_analysis.png - Detailed metrics analysis")

except Exception as e:
    print(f"❌ Error creating comprehensive visualizations: {e}")
    print("Creating fallback basic visualization...")

# Create fallback basic visualization if comprehensive fails
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
plt.savefig('xgb_training_results.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"Basic visualization saved to: xgb_training_results.png")

print(f"\n🎉 XGBoost Training Complete!")
print(
    f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print(f"📊 Best model performance: {best_fold['val_mape']:.2f}% MAPE")
print(
    f"🚀 Backend used: {'XGBoost (GPU)' if GPU_AVAILABLE else 'XGBoost (CPU)'}")
print(f"Script completed at: {datetime.now()}")

if GPU_AVAILABLE:
    print(f"\n💡 GPU Performance Notes:")
    print(f"   - Used GPU-accelerated XGBoost with gpu_hist tree method")
    print(f"   - Training should be significantly faster than CPU-only")
    print(f"   - Hyperparameter search benefits from GPU acceleration")
else:
    print(f"\n💡 To enable GPU acceleration:")
    print(f"   - Install XGBoost with GPU support: pip install xgboost[gpu]")
    print(f"   - Ensure NVIDIA GPU with CUDA support is available")
    print(f"   - Check XGBoost GPU setup: python -c 'import xgboost as xgb; print(xgb.device.Device().device_type)'")
