import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Sklearn imports
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, GridSearchCV, validation_curve, learning_curve
from sklearn.linear_model import Ridge
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
    "Starting FIXED RIDGE Polynomial Regression CNN Execution Time Prediction Training")

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

# Check target variable distribution
print_with_timestamp(f"Original target variable statistics:")
print_with_timestamp(f"  Min: {df['Execution_Time_ms'].min():.4f}")
print_with_timestamp(f"  Max: {df['Execution_Time_ms'].max():.4f}")
print_with_timestamp(f"  Mean: {df['Execution_Time_ms'].mean():.4f}")
print_with_timestamp(f"  Std: {df['Execution_Time_ms'].std():.4f}")

# One-hot encode the Algorithm column
df_encoded = pd.get_dummies(
    df, columns=['Algorithm'], prefix='Algorithm', dtype=int)
df = df_encoded

# CRITICAL FIX 1: Apply log transformation to target variable
print_with_timestamp("🔧 APPLYING LOG TRANSFORMATION to target variable")
# Add small constant to avoid log(0)
df['Execution_Time_ms_log'] = np.log1p(
    df['Execution_Time_ms'])  # log1p = log(1+x)

print_with_timestamp(f"Log-transformed target statistics:")
print_with_timestamp(f"  Min: {df['Execution_Time_ms_log'].min():.4f}")
print_with_timestamp(f"  Max: {df['Execution_Time_ms_log'].max():.4f}")
print_with_timestamp(f"  Mean: {df['Execution_Time_ms_log'].mean():.4f}")
print_with_timestamp(f"  Std: {df['Execution_Time_ms_log'].std():.4f}")

# Scaling features
numerical_cols = ['Batch_Size', 'Input_Size', 'In_Channels',
                  'Out_Channels', 'Kernel_Size', 'Stride', 'Padding']
print_with_timestamp(f"Scaling numerical features: {numerical_cols}")

# CRITICAL FIX 2: Create separate scalers for X and y
scaler_X = StandardScaler()
scaler_y = StandardScaler()

df_scaled = df.copy()
df_scaled[numerical_cols] = scaler_X.fit_transform(df[numerical_cols])

# CRITICAL FIX 3: Scale the log-transformed target
print_with_timestamp("🔧 SCALING LOG-TRANSFORMED TARGET variable")
df_scaled['Execution_Time_ms_log_scaled'] = scaler_y.fit_transform(
    df[['Execution_Time_ms_log']])

# Define feature columns (exclude all target variants)
feature_cols = [col for col in df_scaled.columns if col not in [
    'Execution_Time_ms', 'Execution_Time_ms_log', 'Execution_Time_ms_log_scaled']]
target_col = 'Execution_Time_ms_log_scaled'

print_with_timestamp(f"Features: {len(feature_cols)}")
print_with_timestamp(f"Feature columns: {feature_cols}")

# Create features and target arrays
X = df_scaled[feature_cols]
y = df_scaled[target_col]

print_with_timestamp(
    f"Data preprocessing complete. Features shape: {X.shape}, Target shape: {y.shape}")

# CRITICAL FIX 4: Updated evaluation functions with proper inverse transform


def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def inverse_transform_predictions(y_pred_scaled, scaler_y):
    """Convert scaled log predictions back to original scale"""
    # Step 1: Inverse scale
    y_pred_log = scaler_y.inverse_transform(
        y_pred_scaled.reshape(-1, 1)).flatten()
    # Step 2: Clip to prevent overflow in expm1 (exp(x) - 1)
    # Clipping to reasonable range prevents inf/nan values
    y_pred_log = np.clip(y_pred_log, -50, 50)
    # Step 3: Inverse log transform
    # expm1 = exp(x) - 1, inverse of log1p
    y_pred_original = np.expm1(y_pred_log)
    return y_pred_original


def evaluate_model(model, X, y, scaler_y, y_original):
    """Evaluate polynomial regression model with proper inverse transform"""
    predictions_scaled = model.predict(X)

    # Convert predictions back to original scale
    predictions_original = inverse_transform_predictions(
        predictions_scaled, scaler_y)

    # Calculate metrics on original scale
    mape = calculate_mape(y_original, predictions_original)
    mae = mean_absolute_error(y_original, predictions_original)
    mse = mean_squared_error(y_original, predictions_original)
    r2 = r2_score(y_original, predictions_original)

    return {
        'mape': mape,
        'mae': mae,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2,
        'predictions': predictions_original
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
    f"Training FIXED RIDGE Polynomial Regression with {k}-Fold Cross Validation")
print_with_timestamp(f"Total samples: {len(X)}")
print_with_timestamp(f"Features: {X.shape[1]}")

# Store results for each fold
fold_results = []
trained_models = []

# CRITICAL FIX 5: Expanded alpha range for better regularization tuning
print_with_timestamp(
    "🔧 EXPANDING ALPHA RANGE for better regularization tuning")
param_grid = {
    'polynomialfeatures__degree': [3],
    # Broader range for better tuning
    'ridge__alpha': [1e-4, 1e-3, 1e-2, 1e-1, 1.0]
}


def create_ridge_polynomial_pipeline():
    """Create ridge polynomial regression pipeline"""
    pipeline = Pipeline([
        ('polynomialfeatures', PolynomialFeatures(include_bias=False)),
        ('ridge', Ridge(random_state=42))
    ])
    return pipeline


# Start timing
start_time = time.time()

print_with_timestamp(f"\n{'='*60}")
print_with_timestamp(f"🔢 TRAINING FIXED RIDGE POLYNOMIAL REGRESSION")
print_with_timestamp(f"{'='*60}")

# Store learning curve data
learning_curve_data = []
validation_curve_data = []

# Loop through each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print_with_timestamp(f"\n🔢 RIDGE - FOLD {fold}")
    print_with_timestamp("-" * 40)

    # Get train and validation data for this fold
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]

    # Get original scale targets for evaluation
    y_train_original = df.iloc[train_idx]['Execution_Time_ms']
    y_val_original = df.iloc[val_idx]['Execution_Time_ms']

    print_with_timestamp(f"Training samples: {len(X_train_fold)}")
    print_with_timestamp(f"Validation samples: {len(X_val_fold)}")

    # Create model with GridSearchCV
    print_with_timestamp("Running hyperparameter search...")
    pipeline = create_ridge_polynomial_pipeline()

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

    # Generate validation curve for this fold (MAE vs Alpha)
    print_with_timestamp("📊 Generating validation curve (MAE vs Alpha)...")
    alpha_range = param_grid['ridge__alpha']

    # Custom scoring function that returns MAE on original scale
    def mae_scorer_original_scale(estimator, X_val, y_val):
        y_pred_scaled = estimator.predict(X_val)
        y_pred_original = inverse_transform_predictions(
            y_pred_scaled, scaler_y)
        # Get corresponding original targets
        y_original = df.iloc[X_val.index]['Execution_Time_ms']
        # Negative because sklearn maximizes
        return -mean_absolute_error(y_original, y_pred_original)

    train_scores, val_scores = validation_curve(
        pipeline, X_train_fold, y_train_fold,
        param_name='ridge__alpha', param_range=alpha_range,
        cv=3, scoring=mae_scorer_original_scale, n_jobs=-1
    )

    validation_curve_data.append({
        'fold': fold,
        'alpha_range': alpha_range,
        'train_mae_scores': -train_scores,  # Convert back to positive MAE
        'val_mae_scores': -val_scores
    })

    # Generate learning curve for this fold (MAE vs Training Size)
    print_with_timestamp(
        "📊 Generating learning curve (MAE vs Training Size)...")
    train_sizes = np.linspace(0.1, 1.0, 10)

    train_sizes_abs, train_scores_lc, val_scores_lc = learning_curve(
        best_model, X_train_fold, y_train_fold,
        train_sizes=train_sizes, cv=3,
        scoring=mae_scorer_original_scale, n_jobs=-1
    )

    learning_curve_data.append({
        'fold': fold,
        'train_sizes': train_sizes_abs,
        'train_mae_scores': -train_scores_lc,
        'val_mae_scores': -val_scores_lc
    })

    # Evaluate on training and validation sets with proper inverse transform
    train_metrics = evaluate_model(
        best_model, X_train_fold, y_train_fold, scaler_y, y_train_original)
    val_metrics = evaluate_model(
        best_model, X_val_fold, y_val_fold, scaler_y, y_val_original)

    # Store results for this fold
    fold_result = {
        'fold': fold,
        'model_type': 'ridge_fixed',
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
        'polynomial_degree': best_params['polynomialfeatures__degree'],
        'alpha': best_params['ridge__alpha']
    }

    fold_results.append(fold_result)
    trained_models.append(best_model)

    # Print fold results
    print_with_timestamp(f"🔢 FIXED RIDGE Fold {fold} Results:")
    print_metrics(train_metrics, f"  Training")
    print_metrics(val_metrics, f"  Validation")
    print_with_timestamp(
        f"  Polynomial Degree: {best_params['polynomialfeatures__degree']}")
    print_with_timestamp(
        f"  Alpha (L2 regularization): {best_params['ridge__alpha']}")

    # Save model for this fold
    model_filename = f'ridge_model_fold_{fold}.joblib'
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
print_with_timestamp("🔢 FIXED RIDGE POLYNOMIAL REGRESSION SUMMARY")
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
print_with_timestamp(f"🎯 Best Alpha: {best_fold['alpha']}")
print_with_timestamp(
    f"⏱️  Total Training Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

# Save the best model and scalers
best_model_idx = best_fold['fold'] - 1
best_model = trained_models[best_model_idx]
joblib.dump(best_model, 'best_ridge_model.joblib')
joblib.dump(scaler_X, 'scaler_X.joblib')
joblib.dump(scaler_y, 'scaler_y.joblib')
print_with_timestamp(f"Best model saved as: best_ridge_model.joblib")
print_with_timestamp(f"Scalers saved as: scaler_X.joblib, scaler_y.joblib")

# Save results to CSV
results_df = pd.DataFrame(fold_results)
results_df.to_csv('ridge_training_results.csv', index=False)
print_with_timestamp(f"Results saved to: ridge_training_results.csv")

# Create enhanced visualization with residual plots and learning curves
print_with_timestamp(
    "📊 Creating enhanced visualization with residual plots and learning curves...")

fig, axes = plt.subplots(3, 2, figsize=(16, 18))
fig.suptitle('Fixed Ridge Polynomial Regression Results',
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

# Plot 2: Alpha values used
ax = axes[0, 1]
alphas = [f['alpha'] for f in fold_results]
ax.bar(folds, alphas, alpha=0.7, color='blue')
ax.set_xlabel('Fold')
ax.set_ylabel('Best Alpha Value')
ax.set_title('Best Alpha (L2 Regularization) by Fold')
ax.set_yscale('log')
ax.grid(True, alpha=0.3)

# Plot 3: Predictions vs Actual for best model
ax = axes[1, 0]
# Get all predictions from best model
all_predictions = []
all_actuals = []

for i, (train_idx, val_idx) in enumerate(kf.split(X)):
    if i == best_model_idx:
        X_val = X.iloc[val_idx]
        y_val_original = df.iloc[val_idx]['Execution_Time_ms']
        predictions = inverse_transform_predictions(
            best_model.predict(X_val), scaler_y)
        all_predictions.extend(predictions)
        all_actuals.extend(y_val_original.values)

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

# Plot 4: Residual Plot
ax = axes[1, 1]
residuals = all_predictions - all_actuals
ax.scatter(all_predictions, residuals, alpha=0.6, s=20)
ax.axhline(y=0, color='red', linestyle='--', lw=2, label='Perfect Fit')
ax.set_xlabel('Predicted Execution Time (ms)')
ax.set_ylabel('Residuals (Predicted - Actual)')
ax.set_title('Residual Plot (Best Model)')
ax.legend()
ax.grid(True, alpha=0.3)

# Add residual statistics
residual_mean = np.mean(residuals)
residual_std = np.std(residuals)
ax.text(0.05, 0.95, f'Mean: {residual_mean:.2f}\nStd: {residual_std:.2f}',
        transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcyan"))

# Plot 5: Validation Curve (MAE vs Alpha)
ax = axes[2, 0]
# Average validation curves across all folds
alphas = validation_curve_data[0]['alpha_range']
avg_train_mae = np.mean([fold['train_mae_scores']
                        for fold in validation_curve_data], axis=0)
avg_val_mae = np.mean([fold['val_mae_scores']
                      for fold in validation_curve_data], axis=0)
std_train_mae = np.std([fold['train_mae_scores']
                       for fold in validation_curve_data], axis=0)
std_val_mae = np.std([fold['val_mae_scores']
                     for fold in validation_curve_data], axis=0)

ax.semilogx(alphas, avg_train_mae.mean(axis=1), 'o-', color='blue',
            label='Training MAE', alpha=0.8, linewidth=2)
ax.fill_between(alphas,
                avg_train_mae.mean(axis=1) - std_train_mae.mean(axis=1),
                avg_train_mae.mean(axis=1) + std_train_mae.mean(axis=1),
                alpha=0.2, color='blue')

ax.semilogx(alphas, avg_val_mae.mean(axis=1), 'o-', color='red',
            label='Validation MAE', alpha=0.8, linewidth=2)
ax.fill_between(alphas,
                avg_val_mae.mean(axis=1) - std_val_mae.mean(axis=1),
                avg_val_mae.mean(axis=1) + std_val_mae.mean(axis=1),
                alpha=0.2, color='red')

ax.set_xlabel('Alpha (Regularization Parameter)')
ax.set_ylabel('MAE (Original Scale)')
ax.set_title('Validation Curve: MAE vs Alpha')
ax.legend()
ax.grid(True, alpha=0.3)

# Highlight best alpha
best_alpha = best_fold['alpha']
ax.axvline(x=best_alpha, color='green', linestyle='--', linewidth=2,
           label=f'Best Alpha: {best_alpha}')
ax.legend()

# Plot 6: Learning Curve (MAE vs Training Size)
ax = axes[2, 1]
# Average learning curves across all folds
if learning_curve_data:
    # Get average training sizes (should be same across folds)
    train_sizes = learning_curve_data[0]['train_sizes']
    avg_train_mae_lc = np.mean([fold['train_mae_scores']
                               for fold in learning_curve_data], axis=0)
    avg_val_mae_lc = np.mean([fold['val_mae_scores']
                             for fold in learning_curve_data], axis=0)
    std_train_mae_lc = np.std([fold['train_mae_scores']
                              for fold in learning_curve_data], axis=0)
    std_val_mae_lc = np.std([fold['val_mae_scores']
                            for fold in learning_curve_data], axis=0)

    ax.plot(train_sizes, avg_train_mae_lc.mean(axis=1), 'o-', color='blue',
            label='Training MAE', alpha=0.8, linewidth=2)
    ax.fill_between(train_sizes,
                    avg_train_mae_lc.mean(axis=1) -
                    std_train_mae_lc.mean(axis=1),
                    avg_train_mae_lc.mean(axis=1) +
                    std_train_mae_lc.mean(axis=1),
                    alpha=0.2, color='blue')

    ax.plot(train_sizes, avg_val_mae_lc.mean(axis=1), 'o-', color='red',
            label='Validation MAE', alpha=0.8, linewidth=2)
    ax.fill_between(train_sizes,
                    avg_val_mae_lc.mean(axis=1) - std_val_mae_lc.mean(axis=1),
                    avg_val_mae_lc.mean(axis=1) + std_val_mae_lc.mean(axis=1),
                    alpha=0.2, color='red')

    ax.set_xlabel('Training Set Size')
    ax.set_ylabel('MAE (Original Scale)')
    ax.set_title('Learning Curve: MAE vs Training Size')
    ax.legend()
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Learning curve data not available',
            ha='center', va='center', transform=ax.transAxes)
    ax.set_title('Learning Curve: MAE vs Training Size')

plt.tight_layout()
plt.savefig('ridge_polynomial_results.png', dpi=300, bbox_inches='tight')
plt.close()

print_with_timestamp(
    "✅ Enhanced visualization with residuals and learning curves saved: ridge_polynomial_results.png")

print_with_timestamp(
    f"\n🎉 FIXED Ridge Polynomial Regression Training Complete!")
print_with_timestamp(
    f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print_with_timestamp(f"📊 Best performance: {best_fold['val_mape']:.2f}% MAPE")
print_with_timestamp(f"🎯 Best regularization: Alpha = {best_fold['alpha']}")
print_with_timestamp(f"Script completed at: {datetime.now()}")

print_with_timestamp(f"\n📁 Generated Files:")
print_with_timestamp(f"  • best_ridge_model.joblib - Best trained model")
print_with_timestamp(f"  • scaler_X.joblib - Feature scaler")
print_with_timestamp(f"  • scaler_y.joblib - Target scaler")
print_with_timestamp(f"  • ridge_training_results.csv - Detailed results")
print_with_timestamp(
    f"  • ridge_polynomial_results.png - Enhanced visualization with residuals & learning curves")
print_with_timestamp(f"  • Individual fold models: ridge_model_fold_*.joblib")

print_with_timestamp(f"\n🔧 KEY FIXES APPLIED:")
print_with_timestamp(
    f"  1. ✅ Log transformation of target variable (handles wide range)")
print_with_timestamp(f"  2. ✅ Proper scaling of log-transformed target")
print_with_timestamp(
    f"  3. ✅ Smaller alpha values: {param_grid['ridge__alpha']}")
print_with_timestamp(
    f"  4. ✅ Proper inverse transform for evaluation on original scale")
print_with_timestamp(
    f"  5. ✅ Residual plots for model diagnostics")
print_with_timestamp(
    f"  6. ✅ Learning curves tracking MAE vs training size")
print_with_timestamp(
    f"  7. ✅ Validation curves showing MAE vs alpha parameters")
