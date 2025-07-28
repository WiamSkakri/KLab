import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# Sklearn imports
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import ElasticNet
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
    "🚀 Starting ENHANCED ELASTICNET Polynomial Regression CNN Execution Time Prediction Training")

print_with_timestamp("🔧 CRITICAL FIXES APPLIED:")
print_with_timestamp("  ✅ Log transformation of target variable (CRITICAL)")
print_with_timestamp("  ✅ Proper inverse transformation for evaluation")
print_with_timestamp("  ✅ Separate scalers for features and targets")
print_with_timestamp("  ✅ Optimized alpha range (0.0001 to 0.1)")
print_with_timestamp("  ✅ Enhanced convergence (10,000 iterations)")
print_with_timestamp("  ✅ Random feature selection for better convergence")
print_with_timestamp("  ✅ Polynomial degree 2 optimization")

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

# Evaluation functions


def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(model, X, y_scaled, scaler_y, return_scaled=False):
    """CRITICAL FIX 4: Updated evaluation with proper inverse transform"""
    predictions_scaled = model.predict(X)

    if return_scaled:
        # Return scaled metrics for internal cross-validation
        mape = calculate_mape(y_scaled, predictions_scaled)
        mae = mean_absolute_error(y_scaled, predictions_scaled)
        mse = mean_squared_error(y_scaled, predictions_scaled)
        r2 = r2_score(y_scaled, predictions_scaled)

        return {
            'mape': mape,
            'mae': mae,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'predictions': predictions_scaled
        }
    else:
        # CRITICAL: Inverse transform properly: scaled -> log -> original
        # Step 1: Inverse scale to get log-transformed values
        y_log = scaler_y.inverse_transform(
            y_scaled.values.reshape(-1, 1)).flatten()
        predictions_log = scaler_y.inverse_transform(
            predictions_scaled.reshape(-1, 1)).flatten()

        # Step 2: Inverse log transform to get original scale
        y_original = np.expm1(y_log)  # expm1 is inverse of log1p
        predictions_original = np.expm1(predictions_log)

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
            'predictions': predictions_original,
            'predictions_scaled': predictions_scaled
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
    f"Training ELASTICNET Polynomial Regression with {k}-Fold Cross Validation")
print_with_timestamp(f"Total samples: {len(X)}")
print_with_timestamp(f"Features: {X.shape[1]}")

# Store results for each fold
fold_results = []
trained_models = []

# CRITICAL FIX 5: Updated alpha range based on successful models
print_with_timestamp(
    "🔧 USING ENHANCED PARAMETER GRID with improved alpha range")
param_grid = {
    'polynomialfeatures__degree': [2],  # Test degree 2
    # Alpha range optimized for log-transformed targets
    'elasticnet__alpha': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
    'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
}


def create_elasticnet_polynomial_pipeline():
    """Create IMPROVED elasticnet polynomial regression pipeline"""
    pipeline = Pipeline([
        ('polynomialfeatures', PolynomialFeatures(include_bias=False)),
        # IMPROVED: Better convergence parameters and selection method
        ('elasticnet', ElasticNet(
            random_state=42,
            max_iter=10000,  # Increased from 2000
            tol=1e-4,        # Better tolerance
            selection='random'  # Random selection for better convergence
        ))
    ])
    return pipeline


# Start timing
start_time = time.time()

print_with_timestamp(f"\n{'='*60}")
print_with_timestamp(f"🔢 TRAINING ENHANCED ELASTICNET POLYNOMIAL REGRESSION")
print_with_timestamp(f"{'='*60}")

# Loop through each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print_with_timestamp(f"\n🔢 ENHANCED ELASTICNET - FOLD {fold}")
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
    pipeline = create_elasticnet_polynomial_pipeline()

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=3,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    # Fit the model with improved error handling
    fold_start_time = time.time()

    # Suppress convergence warnings temporarily for cleaner output
    import warnings
    from sklearn.exceptions import ConvergenceWarning

    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=ConvergenceWarning)
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

    # Evaluate on training and validation sets (on original scale for meaningful metrics)
    train_metrics = evaluate_model(
        best_model, X_train_fold, y_train_fold, scaler_y, return_scaled=False)
    val_metrics = evaluate_model(
        best_model, X_val_fold, y_val_fold, scaler_y, return_scaled=False)

    # Get feature selection info (coefficients that are zero)
    elasticnet_model = best_model.named_steps['elasticnet']
    non_zero_coefs = np.sum(elasticnet_model.coef_ != 0)
    total_coefs = len(elasticnet_model.coef_)

    # Store results for this fold
    fold_result = {
        'fold': fold,
        'model_type': 'elasticnet',
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
        'alpha': best_params['elasticnet__alpha'],
        'l1_ratio': best_params['elasticnet__l1_ratio'],
        'non_zero_coefs': non_zero_coefs,
        'total_coefs': total_coefs,
        'sparsity': (total_coefs - non_zero_coefs) / total_coefs
    }

    fold_results.append(fold_result)
    trained_models.append(best_model)

    # Print fold results
    print_with_timestamp(f"🔢 ENHANCED ELASTICNET Fold {fold} Results:")
    print_metrics(train_metrics, f"  Training")
    print_metrics(val_metrics, f"  Validation")
    print_with_timestamp(
        f"  Polynomial Degree: {best_params['polynomialfeatures__degree']}")
    print_with_timestamp(
        f"  Alpha (regularization strength): {best_params['elasticnet__alpha']}")
    print_with_timestamp(
        f"  L1 Ratio (L1 vs L2 mix): {best_params['elasticnet__l1_ratio']}")
    print_with_timestamp(
        f"  Feature Selection: {non_zero_coefs}/{total_coefs} features selected ({100*(1-fold_result['sparsity']):.1f}%)")

    # Save model for this fold
    model_filename = f'elasticnet_model_fold_{fold}.joblib'
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
avg_sparsity = np.mean([f['sparsity'] for f in fold_results])

print_with_timestamp(f"\n{'='*80}")
print_with_timestamp("🔢 ENHANCED ELASTICNET POLYNOMIAL REGRESSION SUMMARY")
print_with_timestamp(f"{'='*80}")

print_with_timestamp(f"📊 Average Performance:")
print_with_timestamp(
    f"Average Val MAPE: {avg_val_mape:.2f}% ± {std_val_mape:.2f}%")
print_with_timestamp(f"Average Val R²: {avg_val_r2:.4f}")
print_with_timestamp(
    f"Average Training Time: {avg_training_time:.1f} seconds per fold")
print_with_timestamp(
    f"Average Sparsity: {avg_sparsity:.2f} ({100*avg_sparsity:.1f}% features removed)")

print_with_timestamp(
    f"\n🏆 Best Fold: Fold {best_fold['fold']} (Val MAPE: {best_fold['val_mape']:.2f}%)")
print_with_timestamp(f"📐 Polynomial Degree: {best_fold['polynomial_degree']}")
print_with_timestamp(f"🎯 Best Alpha: {best_fold['alpha']}")
print_with_timestamp(
    f"⚖️  Best L1 Ratio: {best_fold['l1_ratio']} ({best_fold['l1_ratio']*100:.0f}% L1, {(1-best_fold['l1_ratio'])*100:.0f}% L2)")
print_with_timestamp(
    f"🔍 Feature Selection: {best_fold['non_zero_coefs']}/{best_fold['total_coefs']} features ({100*(1-best_fold['sparsity']):.1f}% kept)")
print_with_timestamp(
    f"⏱️  Total Training Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

# Save the best model and scalers
best_model_idx = best_fold['fold'] - 1
best_model = trained_models[best_model_idx]
joblib.dump(best_model, 'best_enhanced_elasticnet_model.joblib')
joblib.dump(scaler_X, 'enhanced_elasticnet_scaler_X.joblib')
joblib.dump(scaler_y, 'enhanced_elasticnet_scaler_y.joblib')
print_with_timestamp(
    f"Best model saved as: best_enhanced_elasticnet_model.joblib")
print_with_timestamp(
    f"Scalers saved as: enhanced_elasticnet_scaler_X.joblib, enhanced_elasticnet_scaler_y.joblib")

# Save results to CSV
results_df = pd.DataFrame(fold_results)
results_df.to_csv('enhanced_elasticnet_training_results.csv', index=False)
print_with_timestamp(
    f"Results saved to: enhanced_elasticnet_training_results.csv")

# Create visualization
print_with_timestamp("📊 Creating visualization...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle('ElasticNet Polynomial Regression Results',
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

# Plot 2: L1 Ratio and Alpha values
ax = axes[0, 1]
l1_ratios = [f['l1_ratio'] for f in fold_results]
alphas = [f['alpha'] for f in fold_results]

# Create a scatter plot with L1 ratio vs Alpha
scatter = ax.scatter(l1_ratios, alphas, c=val_mapes,
                     cmap='viridis_r', s=100, alpha=0.7)
ax.set_xlabel('L1 Ratio (L1 vs L2 Mix)')
ax.set_ylabel('Alpha (Regularization Strength)')
ax.set_title('Hyperparameter Selection')
ax.grid(True, alpha=0.3)

# Add colorbar for MAPE values
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Validation MAPE (%)')

# Plot 3: Predictions vs Actual for best model
ax = axes[1, 0]
# Get all predictions from best model
all_predictions = []
all_actuals = []

for i, (train_idx, val_idx) in enumerate(kf.split(X)):
    if i == best_model_idx:
        X_val = X.iloc[val_idx]
        y_val = y.iloc[val_idx]
        # Get predictions and transform back to original scale for visualization
        predictions_scaled = best_model.predict(X_val)
        # Transform back: scaled -> log -> original
        predictions_log = scaler_y.inverse_transform(
            predictions_scaled.reshape(-1, 1)).flatten()
        predictions_original = np.expm1(predictions_log)

        y_val_log = scaler_y.inverse_transform(
            y_val.values.reshape(-1, 1)).flatten()
        y_val_original = np.expm1(y_val_log)

        all_predictions.extend(predictions_original)
        all_actuals.extend(y_val_original)

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

# Plot 4: Summary statistics
ax = axes[1, 1]
ax.axis('off')

summary_text = f"""
🚀 ENHANCED ElasticNet Polynomial Regression:
────────────────────────────────────────
Model Type: ElasticNet (L1 + L2)
Polynomial Degree: {best_fold['polynomial_degree']}
✅ Log Transformation: Applied
✅ Target Scaling: Applied  
✅ Convergence: Enhanced (10K iter)
✅ Alpha Range: Optimized (0.0001 to 0.1)

Performance Metrics:
• Best Val MAPE: {best_fold['val_mape']:.2f}%
• Avg Val MAPE: {avg_val_mape:.2f}% ± {std_val_mape:.2f}%
• Best Val R²: {best_fold['val_r2']:.4f}
• Avg Val R²: {avg_val_r2:.4f}

Regularization Mix:
• Best Alpha: {best_fold['alpha']}
• Best L1 Ratio: {best_fold['l1_ratio']}
• L1 Component: {best_fold['l1_ratio']*100:.0f}%
• L2 Component: {(1-best_fold['l1_ratio'])*100:.0f}%

Feature Selection:
• Features Selected: {best_fold['non_zero_coefs']}/{best_fold['total_coefs']}
• Sparsity: {100*best_fold['sparsity']:.1f}% features removed
• Avg Sparsity: {100*avg_sparsity:.1f}%

Training Details:
• Best Fold: {best_fold['fold']}
• Total Time: {total_time:.1f}s ({total_time/60:.1f} min)
• Avg Time/Fold: {avg_training_time:.1f}s
• CV Folds: {k}
• Samples: {len(X)}

Model Features:
• Combined L1 + L2 regularization
• Feature selection + coefficient shrinking
• Best of both Ridge and Lasso
• Robust to multicollinearity
"""

ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
        verticalalignment='top', fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

plt.tight_layout()
plt.savefig('enhanced_elasticnet_polynomial_results.png',
            dpi=300, bbox_inches='tight')
plt.close()

print_with_timestamp(
    "✅ Visualization saved: enhanced_elasticnet_polynomial_results.png")

print_with_timestamp(
    f"\n🎉 ENHANCED ElasticNet Polynomial Regression Training Complete!")
print_with_timestamp("🚀 CRITICAL FIXES SUCCESSFULLY APPLIED!")
print_with_timestamp("✅ Log transformation applied to target variable")
print_with_timestamp("✅ Proper inverse transformation implemented")
print_with_timestamp("✅ Optimized alpha range for log-scale targets")
print_with_timestamp(
    f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print_with_timestamp(f"📊 Best performance: {best_fold['val_mape']:.2f}% MAPE")
print_with_timestamp(
    f"🎯 Best regularization: Alpha = {best_fold['alpha']}, L1 Ratio = {best_fold['l1_ratio']}")
print_with_timestamp(
    f"🔍 Feature selection: {100*(1-best_fold['sparsity']):.1f}% features kept")
print_with_timestamp(
    f"📐 Best polynomial degree: {best_fold['polynomial_degree']}")
print_with_timestamp(f"Script completed at: {datetime.now()}")

print_with_timestamp(f"\n📁 Generated Files:")
print_with_timestamp(
    f"  • best_enhanced_elasticnet_model.joblib - Best trained model")
print_with_timestamp(
    f"  • enhanced_elasticnet_scaler_X.joblib - Feature scaler")
print_with_timestamp(
    f"  • enhanced_elasticnet_scaler_y.joblib - Target scaler")
print_with_timestamp(
    f"  • enhanced_elasticnet_training_results.csv - Detailed results")
print_with_timestamp(
    f"  • enhanced_elasticnet_polynomial_results.png - Visualization")
print_with_timestamp(
    f"  • Individual fold models: elasticnet_model_fold_*.joblib")
