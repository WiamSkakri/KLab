import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import time
import os
import joblib
from datetime import datetime

from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import Lasso
from sklearn.compose import TransformedTargetRegressor
from sklearn.pipeline import Pipeline
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import learning_curve

# ------------------- Logging Function -------------------


def log(msg):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


# ------------------- Load Data -------------------
csv_file = 'combined.csv'
if not os.path.exists(csv_file):
    raise FileNotFoundError(f"{csv_file} not found in working directory.")

df = pd.read_csv(csv_file)
df = pd.get_dummies(df, columns=['Algorithm'], prefix='Algorithm', dtype=int)

# ------------------- Feature Setup -------------------
target_col = 'Execution_Time_ms'
feature_cols = [col for col in df.columns if col != target_col]
X = df[feature_cols]
y = df[target_col]

# ------------------- Scale Features -------------------
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# ------------------- Model Pipeline -------------------


def build_pipeline(alpha):
    poly = PolynomialFeatures(
        degree=2, interaction_only=True, include_bias=False)
    lasso = Lasso(alpha=alpha, max_iter=5000, random_state=42)
    return TransformedTargetRegressor(
        regressor=Pipeline([
            ('poly', poly),
            ('lasso', lasso)
        ]),
        func=np.log1p,
        inverse_func=np.expm1
    )


# ------------------- Cross-Validation Setup -------------------
kf = KFold(n_splits=5, shuffle=True, random_state=42)
param_grid = {
    'regressor__lasso__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0]
}

fold_results = []
trained_models = []
start_time = time.time()
log("🔁 Starting Lasso Polynomial Regression with log-transformed target")

# ------------------- Training Loop -------------------
for fold, (train_idx, val_idx) in enumerate(kf.split(X_scaled), 1):
    log(f"📦 Fold {fold} started...")
    X_train, X_val = X_scaled[train_idx], X_scaled[val_idx]
    y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

    grid = GridSearchCV(build_pipeline(0.01), param_grid,
                        scoring='neg_mean_squared_error', cv=3, n_jobs=-1)
    fold_start = time.time()
    grid.fit(X_train, y_train)
    fold_duration = time.time() - fold_start

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_val)

    mae = mean_absolute_error(y_val, y_pred)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    r2 = r2_score(y_val, y_pred)

    fold_results.append({
        'fold': fold,
        'alpha': grid.best_params_['regressor__lasso__alpha'],
        'mae': mae,
        'rmse': rmse,
        'r2': r2,
        'train_time': fold_duration,
        'actual': y_val.values,
        'predicted': y_pred,
        'model': best_model
    })

    log(f"✅ Fold {fold} - R²: {r2:.4f}, MAE: {mae:.2f}, RMSE: {rmse:.2f}, Time: {fold_duration:.2f}s")

# ------------------- Best Model Selection -------------------
best_fold = max(fold_results, key=lambda x: x['r2'])
best_model = best_fold['model']
joblib.dump(best_model, 'best_lasso_poly_model.joblib')
log("💾 Saved best model to 'best_lasso_poly_model.joblib'")

# ------------------- Save Results -------------------
results_df = pd.DataFrame([{
    'Fold': f['fold'],
    'Alpha': f['alpha'],
    'MAE': f['mae'],
    'RMSE': f['rmse'],
    'R2': f['r2'],
    'TrainTime': f['train_time']
} for f in fold_results])
results_df.to_csv('lasso_poly_results.csv', index=False)
log("📄 Saved results to 'lasso_poly_results.csv'")

# ------------------- Visualization -------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
fig.suptitle(
    'Lasso Polynomial Regression Results (Log-Transformed Target)', fontsize=16)

# Performance by Fold
ax1 = axes[0, 0]
ax1.bar(results_df['Fold'] - 0.2, results_df['MAE'],
        width=0.4, label='MAE', color='orange')
ax1.bar(results_df['Fold'] + 0.2, results_df['R2'],
        width=0.4, label='R²', color='green')
ax1.set_title('Performance by Fold')
ax1.set_xlabel('Fold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# Training Time
ax2 = axes[0, 1]
ax2.bar(results_df['Fold'], results_df['TrainTime'], color='purple')
ax2.set_title('Training Time per Fold')
ax2.set_xlabel('Fold')
ax2.set_ylabel('Time (s)')
ax2.grid(True, alpha=0.3)

# Predictions vs Actual
ax3 = axes[1, 0]
ax3.scatter(best_fold['actual'], best_fold['predicted'], alpha=0.6)
ax3.plot([min(y), max(y)], [min(y), max(y)], 'r--', label='Perfect Prediction')
ax3.set_title(f'Predicted vs Actual (Best Fold: {best_fold["fold"]})')
ax3.set_xlabel('Actual Execution Time (ms)')
ax3.set_ylabel('Predicted Execution Time (ms)')
ax3.legend()
ax3.grid(True, alpha=0.3)
ax3.text(0.05, 0.95, f'R² = {best_fold["r2"]:.4f}', transform=ax3.transAxes,
         bbox=dict(boxstyle="round", facecolor="lightblue"))

# Summary Text
ax4 = axes[1, 1]
ax4.axis('off')
summary = f"""
Best Fold: {best_fold['fold']}
Best Alpha: {best_fold['alpha']}

MAE:  {best_fold['mae']:.2f}
RMSE: {best_fold['rmse']:.2f}
R²:   {best_fold['r2']:.4f}

Total Time: {time.time() - start_time:.1f} seconds
"""
ax4.text(0.05, 0.95, summary, transform=ax4.transAxes, fontsize=11,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle="round", facecolor="lightgray", alpha=0.5))

plt.tight_layout()
# Plot Learning Curve
# Extract the internal pipeline (poly + lasso)
base_pipeline = best_model.regressor
train_sizes, train_scores, val_scores = learning_curve(
    estimator=base_pipeline,
    X=X_scaled, y=np.log1p(y),
    cv=5, scoring='neg_mean_squared_error',
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_rmse = np.sqrt(-train_scores.mean(axis=1))
val_rmse = np.sqrt(-val_scores.mean(axis=1))

fig_lc, ax_lc = plt.subplots(figsize=(8, 6))
ax_lc.plot(train_sizes, train_rmse, label='Training RMSE', marker='o')
ax_lc.plot(train_sizes, val_rmse, label='Validation RMSE', marker='s')
ax_lc.set_xlabel('Training Set Size')
ax_lc.set_ylabel('RMSE')
ax_lc.set_title('Learning Curve (Lasso Polynomial Regression)')
ax_lc.legend()
ax_lc.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lasso_poly_learning_curve.png', dpi=300)
plt.close()

log("📉 Saved learning curve to 'lasso_poly_learning_curve.png'")

plt.savefig('lasso_poly_visualization.png', dpi=300)
plt.close()
log("📊 Saved visualization to 'lasso_poly_visualization.png'")
log("✅ Lasso Polynomial Regression Pipeline Complete.")
