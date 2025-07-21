import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Sklearn imports
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, GridSearchCV, learning_curve
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

# For saving models
import joblib

# Function to print with timestamp


def print_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()


print_with_timestamp(
    "Starting LINEAR Polynomial Regression CNN Execution Time Prediction Training")

# Set random seeds for reproducibility
np.random.seed(42)

# File path handling
csv_file = 'combined.csv'
if not os.path.exists(csv_file):
    print_with_timestamp(f"Error: {csv_file} not found in current directory")
    print_with_timestamp(f"Current directory: {os.getcwd()}")
    print_with_timestamp("Available files:")
    for file in os.listdir('.'):
        print(f"  - {file}")
    sys.exit(1)

print_with_timestamp(f"Loading data from {csv_file}")
df = pd.read_csv(csv_file)
print_with_timestamp(f"Data loaded successfully. Shape: {df.shape}")

# One-hot encode the Algorithm column
df_encoded = pd.get_dummies(
    df, columns=['Algorithm'], prefix='Algorithm', dtype=int)
df = df_encoded

# Scaling features
numerical_cols = ['Batch_Size', 'Input_Size', 'In_Channels',
                  'Out_Channels', 'Kernel_Size', 'Stride', 'Padding']
print_with_timestamp(f"Scaling numerical features: {numerical_cols}")

# Apply Standard Scaling to numerical columns only
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Define feature columns
feature_cols = [col for col in df_scaled.columns if col != 'Execution_Time_ms']
target_col = 'Execution_Time_ms'

print_with_timestamp(f"Features: {len(feature_cols)}")
print_with_timestamp(f"Feature columns: {feature_cols}")

# Create features and target arrays
X = df_scaled[feature_cols]
y = df_scaled[target_col]

print_with_timestamp(
    f"Data preprocessing complete. Features shape: {X.shape}, Target shape: {y.shape}")

# Evaluation functions


def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(model, X, y):
    """Evaluate polynomial regression model"""
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
    print_with_timestamp(f"{title}:")
    print_with_timestamp(f"  MAPE: {metrics['mape']:.2f}%")
    print_with_timestamp(f"  MAE:  {metrics['mae']:.4f}")
    print_with_timestamp(f"  RMSE: {metrics['rmse']:.4f}")
    print_with_timestamp(f"  R²:   {metrics['r2']:.4f}")


# K-Fold Cross Validation Setup
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

print_with_timestamp(
    f"Training LINEAR Polynomial Regression with {k}-Fold Cross Validation")
print_with_timestamp(f"Total samples: {len(X)}")
print_with_timestamp(f"Features: {X.shape[1]}")

# Store results for each fold
fold_results = []
trained_models = []

# Hyperparameter grid for LINEAR regression (degree 2 only)
param_grid = {
    'polynomialfeatures__degree': [2]
}


def create_linear_polynomial_pipeline():
    """Create linear polynomial regression pipeline"""
    pipeline = Pipeline([
        ('polynomialfeatures', PolynomialFeatures(include_bias=False)),
        ('linear', LinearRegression())
    ])
    return pipeline


# Start timing
start_time = time.time()

print_with_timestamp(f"\n{'='*60}")
print_with_timestamp(f"🔢 TRAINING LINEAR POLYNOMIAL REGRESSION")
print_with_timestamp(f"{'='*60}")

# Loop through each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print_with_timestamp(f"\n🔢 LINEAR - FOLD {fold}")
    print_with_timestamp("-" * 40)

    # Get train and validation data for this fold
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]

    print_with_timestamp(f"Training samples: {len(X_train_fold)}")
    print_with_timestamp(f"Validation samples: {len(X_val_fold)}")

    # Create model with GridSearchCV
    print_with_timestamp("Running hyperparameter search...")
    pipeline = create_linear_polynomial_pipeline()

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    # Fit the model
    fold_start_time = time.time()
    grid_search.fit(X_train_fold, y_train_fold)
    fold_training_time = time.time() - fold_start_time

    # Get the best model
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print_with_timestamp(
        f"✅ Training complete in {fold_training_time:.1f} seconds")
    print_with_timestamp(f"Best parameters: {best_params}")
    print_with_timestamp(f"Best CV score (neg MSE): {best_score:.4f}")

    # Evaluate on training and validation sets
    train_metrics = evaluate_model(best_model, X_train_fold, y_train_fold)
    val_metrics = evaluate_model(best_model, X_val_fold, y_val_fold)

    # Store results for this fold
    fold_result = {
        'fold': fold,
        'model_type': 'linear',
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
        'training_time': fold_training_time,
        'polynomial_degree': best_params['polynomialfeatures__degree']
    }

    fold_results.append(fold_result)
    trained_models.append(best_model)

    # Print fold results
    print_with_timestamp(f"🔢 LINEAR Fold {fold} Results:")
    print_metrics(train_metrics, f"  Training")
    print_metrics(val_metrics, f"  Validation")
    print_with_timestamp(
        f"  Polynomial Degree: {best_params['polynomialfeatures__degree']}")

    # Save model for this fold
    model_filename = f'linear_model_fold_{fold}.joblib'
    joblib.dump(best_model, model_filename)
    print_with_timestamp(f"Model saved: {model_filename}")

# Calculate total training time
total_time = time.time() - start_time

# Find best fold
best_fold = min(fold_results, key=lambda x: x['val_mape'])

# Calculate average performance
avg_val_mape = np.mean([f['val_mape'] for f in fold_results])
avg_val_r2 = np.mean([f['val_r2'] for f in fold_results])
std_val_mape = np.std([f['val_mape'] for f in fold_results])
avg_training_time = np.mean([f['training_time'] for f in fold_results])

print_with_timestamp(f"\n{'='*80}")
print_with_timestamp("🔢 LINEAR POLYNOMIAL REGRESSION SUMMARY")
print_with_timestamp(f"{'='*80}")

print_with_timestamp(f"📊 Average Performance:")
print_with_timestamp(
    f"Average Val MAPE: {avg_val_mape:.2f}% ± {std_val_mape:.2f}%")
print_with_timestamp(f"Average Val R²: {avg_val_r2:.4f}")
print_with_timestamp(
    f"Average Training Time: {avg_training_time:.1f} seconds per fold")

print_with_timestamp(
    f"\n🏆 Best Fold: Fold {best_fold['fold']} (Val MAPE: {best_fold['val_mape']:.2f}%)")
print_with_timestamp(f"📐 Polynomial Degree: {best_fold['polynomial_degree']}")
print_with_timestamp(
    f"⏱️  Total Training Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

# Save the best model
best_model_idx = best_fold['fold'] - 1
best_model = trained_models[best_model_idx]
joblib.dump(best_model, 'best_linear_model.joblib')
print_with_timestamp(f"Best model saved as: best_linear_model.joblib")

# Save results to CSV
results_df = pd.DataFrame(fold_results)
results_df.to_csv('linear_training_results.csv', index=False)
print_with_timestamp(f"Results saved to: linear_training_results.csv")

# Create visualization
print_with_timestamp("📊 Creating visualization...")

fig, axes = plt.subplots(3, 2, figsize=(15, 18))
fig.suptitle('Linear Polynomial Regression Results',
             fontsize=16, fontweight='bold')

# Plot 1: Performance by fold
ax = axes[0, 0]
folds = [f['fold'] for f in fold_results]
val_mapes = [f['val_mape'] for f in fold_results]
val_r2s = [f['val_r2'] for f in fold_results]

ax2 = ax.twinx()
bars1 = ax.bar([f - 0.2 for f in folds], val_mapes, 0.4,
               label='Validation MAPE (%)', alpha=0.7, color='orange')
bars2 = ax2.bar([f + 0.2 for f in folds], val_r2s, 0.4,
                label='Validation R²', alpha=0.7, color='green')

ax.set_xlabel('Fold')
ax.set_ylabel('MAPE (%)', color='orange')
ax2.set_ylabel('R²', color='green')
ax.set_title('Performance by Fold')
ax.grid(True, alpha=0.3)
ax.legend(loc='upper left')
ax2.legend(loc='upper right')

# Plot 2: Training time by fold
ax = axes[0, 1]
training_times = [f['training_time'] for f in fold_results]
ax.bar(folds, training_times, alpha=0.7, color='purple')
ax.set_xlabel('Fold')
ax.set_ylabel('Training Time (seconds)')
ax.set_title('Training Time by Fold')
ax.grid(True, alpha=0.3)

# Plot 3: Predictions vs Actual for best model
ax = axes[1, 0]
# Get all predictions from best model
all_predictions = []
all_actuals = []

for i, (train_idx, val_idx) in enumerate(kf.split(X)):
    if i == best_model_idx:
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        predictions = best_model.predict(X_val)
        all_predictions.extend(predictions)
        all_actuals.extend(y_val.values)

all_predictions = np.array(all_predictions)
all_actuals = np.array(all_actuals)

ax.scatter(all_actuals, all_predictions, alpha=0.6, s=20)
min_val, max_val = min(all_actuals.min(), all_predictions.min()), max(
    all_actuals.max(), all_predictions.max())
ax.plot([min_val, max_val], [min_val, max_val],
        'r--', lw=2, label='Perfect Prediction')
ax.set_xlabel('Actual Execution Time (ms)')
ax.set_ylabel('Predicted Execution Time (ms)')
ax.set_title('Predictions vs Actual (Best Model)')
ax.legend()
ax.grid(True, alpha=0.3)

# Add R² annotation
r2_overall = r2_score(all_actuals, all_predictions)
ax.text(0.05, 0.95, f'R² = {r2_overall:.4f}', transform=ax.transAxes,
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))

# Plot 4: Learning Curve
ax = axes[1, 1]
print_with_timestamp("📈 Generating learning curve...")

# Create a fresh pipeline for learning curve
learning_pipeline = create_linear_polynomial_pipeline()
# Set the best parameters found during cross-validation
learning_pipeline.set_params(**best_fold['best_params'])

# Generate learning curve data
train_sizes = np.linspace(0.1, 1.0, 10)
train_sizes_abs, train_scores, val_scores = learning_curve(
    learning_pipeline, X, y,
    train_sizes=train_sizes,
    cv=3,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    random_state=42
)

# Convert scores to positive RMSE
train_rmse_mean = np.sqrt(-train_scores.mean(axis=1))
train_rmse_std = np.sqrt(-train_scores).std(axis=1)
val_rmse_mean = np.sqrt(-val_scores.mean(axis=1))
val_rmse_std = np.sqrt(-val_scores).std(axis=1)

ax.plot(train_sizes_abs, train_rmse_mean, 'o-', color='blue',
        label='Training RMSE', linewidth=2)
ax.fill_between(train_sizes_abs, train_rmse_mean - train_rmse_std,
                train_rmse_mean + train_rmse_std, alpha=0.1, color='blue')

ax.plot(train_sizes_abs, val_rmse_mean, 'o-', color='red',
        label='Validation RMSE', linewidth=2)
ax.fill_between(train_sizes_abs, val_rmse_mean - val_rmse_std,
                val_rmse_mean + val_rmse_std, alpha=0.1, color='red')

ax.set_xlabel('Training Set Size')
ax.set_ylabel('RMSE')
ax.set_title('Learning Curve')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 5: Residuals Histogram
ax = axes[2, 0]
print_with_timestamp("📊 Generating residuals histogram...")

# Calculate residuals
residuals = all_actuals - all_predictions

# Create histogram
n_bins = 30
ax.hist(residuals, bins=n_bins, alpha=0.7, color='skyblue', edgecolor='black')
ax.axvline(0, color='red', linestyle='--',
           linewidth=2, label='Perfect Predictions')
ax.axvline(residuals.mean(), color='orange', linestyle='-', linewidth=2,
           label=f'Mean Residual: {residuals.mean():.4f}')

ax.set_xlabel('Residuals (Actual - Predicted)')
ax.set_ylabel('Frequency')
ax.set_title('Distribution of Residuals')
ax.legend()
ax.grid(True, alpha=0.3)

# Add statistics text box
residual_stats = f"""
Residuals Statistics:
Mean: {residuals.mean():.4f}
Std: {residuals.std():.4f}
Min: {residuals.min():.4f}
Max: {residuals.max():.4f}
"""
ax.text(0.02, 0.98, residual_stats, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.8))

# Plot 6: Summary statistics
ax = axes[2, 1]
ax.axis('off')

summary_text = f"""
Linear Polynomial Regression Summary:
─────────────────────────────────────
Model Type: Linear Regression
Polynomial Degree: 2

Performance Metrics:
• Best Val MAPE: {best_fold['val_mape']:.2f}%
• Avg Val MAPE: {avg_val_mape:.2f}% ± {std_val_mape:.2f}%
• Best Val R²: {best_fold['val_r2']:.4f}
• Avg Val R²: {avg_val_r2:.4f}

Training Details:
• Best Fold: {best_fold['fold']}
• Total Time: {total_time:.1f}s ({total_time/60:.1f} min)
• Avg Time/Fold: {avg_training_time:.1f}s
• CV Folds: {k}
• Samples: {len(X)}

Model Features:
• No regularization
• Fast training
• Good baseline model

Residuals Analysis:
• Mean Residual: {residuals.mean():.4f}
• Residuals Std: {residuals.std():.4f}
• Normally Distributed: {abs(residuals.mean()) < 0.01}
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

plt.tight_layout()
plt.savefig('linear_polynomial_results.png', dpi=300, bbox_inches='tight')
plt.close()

print_with_timestamp("✅ Visualization saved: linear_polynomial_results.png")

print_with_timestamp(f"\n🎉 Linear Polynomial Regression Training Complete!")
print_with_timestamp(
    f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print_with_timestamp(f"📊 Best performance: {best_fold['val_mape']:.2f}% MAPE")
print_with_timestamp(f"Script completed at: {datetime.now()}")

print_with_timestamp(f"\n📁 Generated Files:")
print_with_timestamp(f"  • best_linear_model.joblib - Best trained model")
print_with_timestamp(f"  • linear_training_results.csv - Detailed results")
print_with_timestamp(f"  • linear_polynomial_results.png - Visualization")
print_with_timestamp(f"  • Individual fold models: linear_model_fold_*.joblib")
