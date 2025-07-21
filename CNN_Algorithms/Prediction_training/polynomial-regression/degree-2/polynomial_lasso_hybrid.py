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
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.compose import TransformedTargetRegressor

# For saving models
import joblib


def print_with_timestamp(message):
    """Print with timestamp for better logging"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model(model, X, y):
    """Comprehensive model evaluation"""
    predictions = model.predict(X)

    return {
        'mape': calculate_mape(y, predictions),
        'mae': mean_absolute_error(y, predictions),
        'mse': mean_squared_error(y, predictions),
        'rmse': np.sqrt(mean_squared_error(y, predictions)),
        'r2': r2_score(y, predictions),
        'predictions': predictions
    }


def build_lasso_pipeline(alpha=0.01):
    """Build Lasso polynomial pipeline with log transformation"""
    poly = PolynomialFeatures(
        degree=2, interaction_only=True, include_bias=False)
    lasso = Lasso(alpha=alpha, max_iter=5000, random_state=42)

    # Use log transformation for execution time data (Version 1 approach)
    return TransformedTargetRegressor(
        regressor=Pipeline([
            ('poly', poly),
            ('lasso', lasso)
        ]),
        func=np.log1p,
        inverse_func=np.expm1
    )


print_with_timestamp("🚀 Starting Hybrid Lasso Polynomial Regression")

# Data loading with error handling (Version 2 approach)
csv_file = 'combined.csv'
if not os.path.exists(csv_file):
    print_with_timestamp(f"Error: {csv_file} not found in current directory")
    print_with_timestamp(f"Current directory: {os.getcwd()}")
    sys.exit(1)

print_with_timestamp(f"Loading data from {csv_file}")
df = pd.read_csv(csv_file)
print_with_timestamp(f"Data loaded successfully. Shape: {df.shape}")

# Data preprocessing
df = pd.get_dummies(df, columns=['Algorithm'], prefix='Algorithm', dtype=int)

# Feature scaling
numerical_cols = ['Batch_Size', 'Input_Size', 'In_Channels',
                  'Out_Channels', 'Kernel_Size', 'Stride', 'Padding']
scaler = StandardScaler()

# Apply scaling to numerical features only
X_features = df.drop('Execution_Time_ms', axis=1)
X_scaled = X_features.copy()
X_scaled[numerical_cols] = scaler.fit_transform(X_features[numerical_cols])
y = df['Execution_Time_ms']

print_with_timestamp(
    f"Features shape: {X_scaled.shape}, Target shape: {y.shape}")

# Cross-validation setup
kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {
    'regressor__lasso__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]
}

fold_results = []
trained_models = []
start_time = time.time()

print_with_timestamp(
    "🔄 Starting 5-Fold Cross Validation with Log Transformation")

# Training loop combining both approaches
for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
    print_with_timestamp(f"\n📦 FOLD {fold}")
    print_with_timestamp("-" * 40)

    X_train, X_val = X_scaled.iloc[train_idx], X_scaled.iloc[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    print_with_timestamp(f"Training samples: {len(X_train)}")
    print_with_timestamp(f"Validation samples: {len(X_val)}")

    # Grid search for best hyperparameters
    grid = GridSearchCV(
        build_lasso_pipeline(),
        param_grid,
        scoring='neg_mean_squared_error',
        cv=3,
        n_jobs=-1
    )

    fold_start = time.time()
    grid.fit(X_train, y_train)
    fold_duration = time.time() - fold_start

    best_model = grid.best_estimator_

    # Evaluate model
    train_metrics = evaluate_model(best_model, X_train, y_train)
    val_metrics = evaluate_model(best_model, X_val, y_val)

    # Feature selection analysis (Version 2 approach)
    lasso_regressor = best_model.regressor.named_steps['lasso']
    non_zero_coefs = np.sum(lasso_regressor.coef_ != 0)
    total_coefs = len(lasso_regressor.coef_)
    sparsity = (total_coefs - non_zero_coefs) / total_coefs

    fold_result = {
        'fold': fold,
        'alpha': grid.best_params_['regressor__lasso__alpha'],
        'train_mape': train_metrics['mape'],
        'val_mape': val_metrics['mape'],
        'train_mae': train_metrics['mae'],
        'val_mae': val_metrics['mae'],
        'train_r2': train_metrics['r2'],
        'val_r2': val_metrics['r2'],
        'train_rmse': train_metrics['rmse'],
        'val_rmse': val_metrics['rmse'],
        'training_time': fold_duration,
        'non_zero_coefs': non_zero_coefs,
        'total_coefs': total_coefs,
        'sparsity': sparsity,
        'actual': y_val.values,
        'predicted': val_metrics['predictions'],
        'model': best_model
    }

    fold_results.append(fold_result)
    trained_models.append(best_model)

    print_with_timestamp(f"✅ Fold {fold} Complete:")
    print_with_timestamp(f"  Val MAPE: {val_metrics['mape']:.2f}%")
    print_with_timestamp(f"  Val R²: {val_metrics['r2']:.4f}")
    print_with_timestamp(
        f"  Alpha: {grid.best_params_['regressor__lasso__alpha']}")
    print_with_timestamp(
        f"  Features: {non_zero_coefs}/{total_coefs} ({100*(1-sparsity):.1f}% kept)")
    print_with_timestamp(f"  Time: {fold_duration:.2f}s")

# Results analysis
total_time = time.time() - start_time
best_fold = min(fold_results, key=lambda x: x['val_mape'])
avg_val_mape = np.mean([f['val_mape'] for f in fold_results])
avg_val_r2 = np.mean([f['val_r2'] for f in fold_results])
avg_sparsity = np.mean([f['sparsity'] for f in fold_results])

print_with_timestamp(f"\n{'='*80}")
print_with_timestamp("🎯 HYBRID LASSO REGRESSION SUMMARY")
print_with_timestamp(f"{'='*80}")
print_with_timestamp(
    f"🏆 Best Fold: {best_fold['fold']} (MAPE: {best_fold['val_mape']:.2f}%)")
print_with_timestamp(f"📊 Average MAPE: {avg_val_mape:.2f}%")
print_with_timestamp(f"📈 Average R²: {avg_val_r2:.4f}")
print_with_timestamp(
    f"🔍 Average Sparsity: {100*avg_sparsity:.1f}% features removed")
print_with_timestamp(f"⏱️ Total Time: {total_time:.1f}s")

# Save best model and results
best_model = best_fold['model']
joblib.dump(best_model, 'best_hybrid_lasso_model.joblib')

results_df = pd.DataFrame([{
    'Fold': f['fold'],
    'Alpha': f['alpha'],
    'Val_MAPE': f['val_mape'],
    'Val_R2': f['val_r2'],
    'Val_MAE': f['val_mae'],
    'Sparsity': f['sparsity'],
    'Training_Time': f['training_time']
} for f in fold_results])
results_df.to_csv('hybrid_lasso_results.csv', index=False)

# Enhanced visualization combining both approaches
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Hybrid Lasso Polynomial Regression (Log-Transformed Target)',
             fontsize=16, fontweight='bold')

# Plot 1: Performance metrics
ax1 = axes[0, 0]
folds = [f['fold'] for f in fold_results]
val_mapes = [f['val_mape'] for f in fold_results]
val_r2s = [f['val_r2'] for f in fold_results]

ax1_twin = ax1.twinx()
bars1 = ax1.bar([f - 0.2 for f in folds], val_mapes, 0.4,
                label='MAPE (%)', color='orange', alpha=0.7)
bars2 = ax1_twin.bar([f + 0.2 for f in folds], val_r2s,
                     0.4, label='R²', color='green', alpha=0.7)

ax1.set_xlabel('Fold')
ax1.set_ylabel('MAPE (%)', color='orange')
ax1_twin.set_ylabel('R²', color='green')
ax1.set_title('Performance by Fold')
ax1.grid(True, alpha=0.3)

# Plot 2: Feature selection analysis
ax2 = axes[0, 1]
sparsity_pct = [100 * (1 - f['sparsity']) for f in fold_results]
alphas = [f['alpha'] for f in fold_results]
scatter = ax2.scatter(folds, sparsity_pct, c=alphas,
                      cmap='viridis', s=100, alpha=0.8)
ax2.set_xlabel('Fold')
ax2.set_ylabel('Features Selected (%)')
ax2.set_title('Feature Selection by Fold')
ax2.grid(True, alpha=0.3)
plt.colorbar(scatter, ax=ax2, label='Alpha Value')

# Plot 3: Predictions vs Actual
ax3 = axes[1, 0]
all_actual = np.concatenate([f['actual'] for f in fold_results])
all_pred = np.concatenate([f['predicted'] for f in fold_results])
ax3.scatter(all_actual, all_pred, alpha=0.6, s=20)
min_val, max_val = min(all_actual.min(), all_pred.min()), max(
    all_actual.max(), all_pred.max())
ax3.plot([min_val, max_val], [min_val, max_val],
         'r--', lw=2, label='Perfect Prediction')
ax3.set_xlabel('Actual Execution Time (ms)')
ax3.set_ylabel('Predicted Execution Time (ms)')
ax3.set_title('Predictions vs Actual (All Folds)')
ax3.legend()
ax3.grid(True, alpha=0.3)

# Add overall R²
overall_r2 = r2_score(all_actual, all_pred)
ax3.text(0.05, 0.95, f'Overall R² = {overall_r2:.4f}', transform=ax3.transAxes,
         bbox=dict(boxstyle="round", facecolor="lightblue"))

# Plot 4: Learning curve (Version 1 approach)
ax4 = axes[1, 1]
base_pipeline = best_model.regressor
train_sizes, train_scores, val_scores = learning_curve(
    estimator=base_pipeline,
    X=X_scaled, y=np.log1p(y),  # Use log-transformed target
    cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_rmse = np.sqrt(-train_scores.mean(axis=1))
val_rmse = np.sqrt(-val_scores.mean(axis=1))

ax4.plot(train_sizes, train_rmse, 'o-', label='Training RMSE', color='blue')
ax4.plot(train_sizes, val_rmse, 's-', label='Validation RMSE', color='red')
ax4.set_xlabel('Training Set Size')
ax4.set_ylabel('RMSE (Log Scale)')
ax4.set_title('Learning Curve')
ax4.legend()
ax4.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('hybrid_lasso_polynomial_results.png',
            dpi=300, bbox_inches='tight')
plt.close()

print_with_timestamp(
    "📊 Visualization saved: hybrid_lasso_polynomial_results.png")
print_with_timestamp("💾 Model saved: best_hybrid_lasso_model.joblib")
print_with_timestamp("📄 Results saved: hybrid_lasso_results.csv")
print_with_timestamp("✅ Hybrid Lasso Polynomial Regression Complete!")
