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
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
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
    "Starting Polynomial Regression CNN Execution Time Prediction Training")

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

print_with_timestamp(f"Training with {k}-Fold Cross Validation")
print_with_timestamp(f"Total samples: {len(X)}")
print_with_timestamp(f"Features: {X.shape[1]}")

# Store results for each fold
fold_data = []

# Prepare data splits for each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print_with_timestamp(f"Preparing FOLD {fold}")

    # Get train and validation data for this fold
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]

    print_with_timestamp(f"Training samples: {len(X_train_fold)}")
    print_with_timestamp(f"Validation samples: {len(X_val_fold)}")

    fold_data.append({
        'fold': fold,
        'X_train': X_train_fold,
        'X_val': X_val_fold,
        'y_train': y_train_fold,
        'y_val': y_val_fold,
        'train_size': len(X_train_fold),
        'val_size': len(X_val_fold)
    })

# Polynomial Regression Configuration
print_with_timestamp("🔢 Polynomial Regression Configuration:")

# Hyperparameter grid for polynomial regression
param_grids = {
    'ridge': {
        'polynomialfeatures__degree': [1, 2, 3, 4],
        'ridge__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    },
    'lasso': {
        'polynomialfeatures__degree': [1, 2, 3, 4],
        'lasso__alpha': [0.01, 0.1, 1.0, 10.0, 100.0]
    },
    'elasticnet': {
        'polynomialfeatures__degree': [1, 2, 3, 4],
        'elasticnet__alpha': [0.01, 0.1, 1.0, 10.0],
        'elasticnet__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9]
    },
    'linear': {
        'polynomialfeatures__degree': [1, 2, 3, 4, 5]
    }
}


def create_polynomial_pipeline(model_type='ridge'):
    """Create polynomial regression pipeline with specified model type"""
    if model_type == 'ridge':
        model = Ridge(random_state=42)
    elif model_type == 'lasso':
        model = Lasso(random_state=42, max_iter=2000)
    elif model_type == 'elasticnet':
        model = ElasticNet(random_state=42, max_iter=2000)
    elif model_type == 'linear':
        model = LinearRegression()
    else:
        raise ValueError(f"Unknown model type: {model_type}")

    # Create pipeline with polynomial features and the model
    pipeline = Pipeline([
        ('polynomialfeatures', PolynomialFeatures(include_bias=False)),
        (model_type, model)
    ])

    return pipeline


def create_poly_with_gridsearch(model_type='ridge', cv_folds=3):
    """Create polynomial regression with grid search"""
    pipeline = create_polynomial_pipeline(model_type)
    param_grid = param_grids[model_type]

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv_folds,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    return grid_search


# Test different polynomial regression models
model_types = ['ridge', 'lasso', 'elasticnet', 'linear']
print_with_timestamp(
    f"Will test {len(model_types)} model types: {model_types}")

# Store results from all folds and models
all_results = {}
trained_models = {}

# Start timing
start_time = time.time()

# Loop through each model type
for model_type in model_types:
    print_with_timestamp(f"\n{'='*60}")
    print_with_timestamp(
        f"🔢 TRAINING {model_type.upper()} POLYNOMIAL REGRESSION")
    print_with_timestamp(f"{'='*60}")

    fold_metrics = []
    fold_models = []

    # Loop through each fold for this model type
    for fold_info in fold_data:
        fold = fold_info['fold']
        X_train = fold_info['X_train']
        X_val = fold_info['X_val']
        y_train = fold_info['y_train']
        y_val = fold_info['y_val']

        print_with_timestamp(f"\n🔢 {model_type.upper()} - FOLD {fold}")
        print_with_timestamp("-" * 40)

        # Create model with GridSearchCV
        print_with_timestamp("Running hyperparameter search...")
        poly_grid = create_poly_with_gridsearch(model_type, cv_folds=3)

        # Fit the model
        fold_start_time = time.time()
        poly_grid.fit(X_train, y_train)
        fold_training_time = time.time() - fold_start_time

        # Get the best model
        best_model = poly_grid.best_estimator_
        best_params = poly_grid.best_params_
        best_score = poly_grid.best_score_

        print_with_timestamp(
            f"✅ Training complete in {fold_training_time:.1f} seconds")
        print_with_timestamp(f"Best parameters: {best_params}")
        print_with_timestamp(f"Best CV score (neg MSE): {best_score:.4f}")

        # Evaluate on training and validation sets
        train_metrics = evaluate_model(best_model, X_train, y_train)
        val_metrics = evaluate_model(best_model, X_val, y_val)

        # Store results for this fold
        fold_result = {
            'fold': fold,
            'model_type': model_type,
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

        fold_metrics.append(fold_result)
        fold_models.append(best_model)

        # Print fold results
        print_with_timestamp(f"🔢 {model_type.upper()} Fold {fold} Results:")
        print_metrics(train_metrics, f"  Training")
        print_metrics(val_metrics, f"  Validation")
        print_with_timestamp(
            f"  Polynomial Degree: {best_params['polynomialfeatures__degree']}")

        # Save model for this fold
        model_filename = f'poly_{model_type}_model_fold_{fold}.joblib'
        joblib.dump(best_model, model_filename)
        print_with_timestamp(f"Model saved: {model_filename}")

    # Store results for this model type
    all_results[model_type] = fold_metrics
    trained_models[model_type] = fold_models

    # Calculate average performance for this model type
    avg_val_mape = np.mean([f['val_mape'] for f in fold_metrics])
    avg_val_r2 = np.mean([f['val_r2'] for f in fold_metrics])
    std_val_mape = np.std([f['val_mape'] for f in fold_metrics])
    avg_training_time = np.mean([f['training_time'] for f in fold_metrics])

    print_with_timestamp(f"\n📊 {model_type.upper()} SUMMARY:")
    print_with_timestamp(
        f"Average Val MAPE: {avg_val_mape:.2f}% ± {std_val_mape:.2f}%")
    print_with_timestamp(f"Average Val R²: {avg_val_r2:.4f}")
    print_with_timestamp(
        f"Average Training Time: {avg_training_time:.1f} seconds per fold")

# Calculate total training time
total_time = time.time() - start_time

# Find best overall model
print_with_timestamp(f"\n{'='*80}")
print_with_timestamp("🔢 POLYNOMIAL REGRESSION CROSS VALIDATION SUMMARY")
print_with_timestamp(f"{'='*80}")

# Compare all models
model_comparison = []
for model_type, fold_metrics in all_results.items():
    avg_val_mape = np.mean([f['val_mape'] for f in fold_metrics])
    avg_val_r2 = np.mean([f['val_r2'] for f in fold_metrics])
    std_val_mape = np.std([f['val_mape'] for f in fold_metrics])
    std_val_r2 = np.std([f['val_r2'] for f in fold_metrics])
    avg_training_time = np.mean([f['training_time'] for f in fold_metrics])

    model_comparison.append({
        'model_type': model_type,
        'avg_val_mape': avg_val_mape,
        'std_val_mape': std_val_mape,
        'avg_val_r2': avg_val_r2,
        'std_val_r2': std_val_r2,
        'avg_training_time': avg_training_time
    })

# Sort by validation MAPE (lower is better)
model_comparison.sort(key=lambda x: x['avg_val_mape'])

print_with_timestamp("📊 Model Performance Comparison (sorted by Val MAPE):")
print_with_timestamp(
    f"{'Model':<12} {'Val MAPE':<15} {'Val R²':<15} {'Avg Time(s)':<12}")
print_with_timestamp("-" * 70)
for comp in model_comparison:
    print_with_timestamp(f"{comp['model_type']:<12} "
                         f"{comp['avg_val_mape']:.2f}% ± {comp['std_val_mape']:.2f}%    "
                         f"{comp['avg_val_r2']:.4f} ± {comp['std_val_r2']:.4f}   "
                         f"{comp['avg_training_time']:<12.1f}")

# Best overall model
best_model_type = model_comparison[0]['model_type']
best_fold_metrics = all_results[best_model_type]
best_fold = min(best_fold_metrics, key=lambda x: x['val_mape'])

print_with_timestamp(f"\n🏆 Best Overall Model: {best_model_type.upper()}")
print_with_timestamp(
    f"🎯 Best Fold: Fold {best_fold['fold']} (Val MAPE: {best_fold['val_mape']:.2f}%)")
print_with_timestamp(
    f"📐 Best Polynomial Degree: {best_fold['polynomial_degree']}")
print_with_timestamp(
    f"⏱️  Total Training Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

# Save the best model
best_model_idx = best_fold['fold'] - 1
best_model = trained_models[best_model_type][best_model_idx]
joblib.dump(best_model, 'best_polynomial_model.joblib')
print_with_timestamp(f"Best model saved as: best_polynomial_model.joblib")

# Save all results to CSV
all_fold_results = []
for model_type, fold_metrics in all_results.items():
    all_fold_results.extend(fold_metrics)

results_df = pd.DataFrame(all_fold_results)
results_df.to_csv('polynomial_training_results.csv', index=False)
print_with_timestamp(f"Results saved to: polynomial_training_results.csv")

# Create summary metrics
summary_metrics = []
for comp in model_comparison:
    summary_metrics.append({
        'model_type': comp['model_type'],
        'avg_val_mape': comp['avg_val_mape'],
        'std_val_mape': comp['std_val_mape'],
        'avg_val_r2': comp['avg_val_r2'],
        'std_val_r2': comp['std_val_r2'],
        'avg_training_time': comp['avg_training_time'],
        'total_training_time': total_time,
        'num_folds': k,
        'best_model': comp['model_type'] == best_model_type,
        'timestamp': datetime.now().isoformat()
    })

summary_df = pd.DataFrame(summary_metrics)
summary_df.to_csv('polynomial_summary_metrics.csv', index=False)
print_with_timestamp(
    f"Summary metrics saved to: polynomial_summary_metrics.csv")

# ===============================
# COMPREHENSIVE VISUALIZATION
# ===============================


def create_polynomial_evaluation_dashboard():
    """Create comprehensive polynomial regression evaluation dashboard"""
    print_with_timestamp(
        "📊 Creating polynomial regression evaluation dashboard...")

    # Collect all predictions and actuals from best model
    best_models = trained_models[best_model_type]
    all_predictions = []
    all_actuals = []
    all_fold_labels = []

    for i, fold_info in enumerate(fold_data):
        # Get validation predictions for this fold
        val_pred = best_models[i].predict(fold_info['X_val'])
        val_actual = fold_info['y_val'].values

        all_predictions.extend(val_pred)
        all_actuals.extend(val_actual)
        all_fold_labels.extend([f'Fold {i+1}'] * len(val_pred))

    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    # Create main dashboard
    fig, axes = plt.subplots(3, 3, figsize=(18, 15))
    fig.suptitle(f'Polynomial Regression Model Evaluation Dashboard\nBest Model: {best_model_type.upper()}',
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
    ax.set_title(f'Predictions vs Actual ({best_model_type.upper()})')
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

    # 3. Model Comparison
    ax = axes[0, 2]
    model_names = [comp['model_type'].upper() for comp in model_comparison]
    val_mapes = [comp['avg_val_mape'] for comp in model_comparison]

    bars = ax.bar(model_names, val_mapes, alpha=0.7)
    # Highlight best model
    bars[0].set_color('red')
    bars[0].set_alpha(1.0)

    ax.set_xlabel('Model Type')
    ax.set_ylabel('Average Validation MAPE (%)')
    ax.set_title('Model Performance Comparison')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45)

    # 4. Polynomial Degree Analysis
    ax = axes[1, 0]
    degrees_data = {}
    for model_type, fold_metrics in all_results.items():
        for fold_metric in fold_metrics:
            degree = fold_metric['polynomial_degree']
            if degree not in degrees_data:
                degrees_data[degree] = []
            degrees_data[degree].append(fold_metric['val_mape'])

    degrees = sorted(degrees_data.keys())
    avg_mapes = [np.mean(degrees_data[d]) for d in degrees]

    ax.bar(degrees, avg_mapes, alpha=0.7)
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Average Validation MAPE (%)')
    ax.set_title('Performance vs Polynomial Degree')
    ax.grid(True, alpha=0.3)

    # 5. Cross-Validation Results for Best Model
    ax = axes[1, 1]
    best_fold_results = all_results[best_model_type]
    folds = [f['fold'] for f in best_fold_results]
    fold_mapes = [f['val_mape'] for f in best_fold_results]
    fold_r2s = [f['val_r2'] for f in best_fold_results]

    ax2 = ax.twinx()
    bars1 = ax.bar([f - 0.2 for f in folds], fold_mapes, 0.4,
                   label='Validation MAPE (%)', alpha=0.7, color='orange')
    bars2 = ax2.bar([f + 0.2 for f in folds], fold_r2s, 0.4,
                    label='Validation R²', alpha=0.7, color='green')

    ax.set_xlabel('Fold')
    ax.set_ylabel('MAPE (%)', color='orange')
    ax2.set_ylabel('R²', color='green')
    ax.set_title(f'CV Performance ({best_model_type.upper()})')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')

    # 6. Error Distribution
    ax = axes[1, 2]
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

    # 7. Training Time Comparison
    ax = axes[2, 0]
    model_names = [comp['model_type'].upper() for comp in model_comparison]
    training_times = [comp['avg_training_time'] for comp in model_comparison]

    bars = ax.bar(model_names, training_times, alpha=0.7, color='purple')
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Average Training Time (seconds)')
    ax.set_title('Training Time Comparison')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45)

    # 8. Performance vs Complexity
    ax = axes[2, 1]
    for model_type, fold_metrics in all_results.items():
        degrees = [f['polynomial_degree'] for f in fold_metrics]
        mapes = [f['val_mape'] for f in fold_metrics]
        ax.scatter(degrees, mapes, label=model_type.upper(), alpha=0.7, s=50)

    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Validation MAPE (%)')
    ax.set_title('Performance vs Model Complexity')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 9. Model Summary
    ax = axes[2, 2]
    ax.axis('off')

    summary_text = f"""
Polynomial Regression Summary:
─────────────────────────────
Best Model: {best_model_type.upper()}
Best Fold: {best_fold['fold']}
Best Polynomial Degree: {best_fold['polynomial_degree']}

Performance Metrics:
• Best Val MAPE: {best_fold['val_mape']:.2f}%
• Best Val R²: {best_fold['val_r2']:.4f}
• Best Val MAE: {best_fold['val_mae']:.4f}

Model Comparison:
"""

    for i, comp in enumerate(model_comparison[:3]):  # Top 3 models
        rank = i + 1
        summary_text += f"• #{rank}: {comp['model_type'].upper()} ({comp['avg_val_mape']:.2f}% MAPE)\n"

    summary_text += f"""
Training Efficiency:
• Total Time: {total_time:.1f}s ({total_time/60:.1f} min)
• Samples: {len(all_actuals)} total
• CV Folds: {k}
• Models Tested: {len(model_types)}
"""

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))

    plt.tight_layout()
    plt.savefig('polynomial_evaluation_dashboard.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print_with_timestamp(
        "✅ Evaluation dashboard saved: polynomial_evaluation_dashboard.png")


def create_detailed_polynomial_analysis():
    """Create detailed polynomial regression analysis plots"""
    print_with_timestamp("📊 Creating detailed polynomial analysis...")

    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Polynomial Regression Detailed Analysis',
                 fontsize=16, fontweight='bold')

    # 1. MAPE Comparison Across Models and Folds
    ax = axes[0, 0]

    for i, (model_type, fold_metrics) in enumerate(all_results.items()):
        folds = [f['fold'] for f in fold_metrics]
        mapes = [f['val_mape'] for f in fold_metrics]
        ax.plot(folds, mapes, 'o-', label=model_type.upper(),
                linewidth=2, markersize=6)

    ax.set_xlabel('Fold')
    ax.set_ylabel('Validation MAPE (%)')
    ax.set_title('MAPE Across Folds for All Models')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 2. R² Comparison Across Models and Folds
    ax = axes[0, 1]

    for i, (model_type, fold_metrics) in enumerate(all_results.items()):
        folds = [f['fold'] for f in fold_metrics]
        r2s = [f['val_r2'] for f in fold_metrics]
        ax.plot(folds, r2s, 's-', label=model_type.upper(),
                linewidth=2, markersize=6)

    ax.set_xlabel('Fold')
    ax.set_ylabel('Validation R²')
    ax.set_title('R² Across Folds for All Models')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 3. Polynomial Degree Distribution
    ax = axes[0, 2]
    all_degrees = []
    for model_type, fold_metrics in all_results.items():
        degrees = [f['polynomial_degree'] for f in fold_metrics]
        all_degrees.extend(degrees)

    degree_counts = {}
    for degree in all_degrees:
        degree_counts[degree] = degree_counts.get(degree, 0) + 1

    degrees = sorted(degree_counts.keys())
    counts = [degree_counts[d] for d in degrees]

    bars = ax.bar(degrees, counts, alpha=0.7)
    ax.set_xlabel('Polynomial Degree')
    ax.set_ylabel('Frequency (across all models and folds)')
    ax.set_title('Polynomial Degree Selection Frequency')
    ax.grid(True, alpha=0.3)

    # Add percentage labels
    total_selections = sum(counts)
    for bar, count in zip(bars, counts):
        height = bar.get_height()
        percentage = (count / total_selections) * 100
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{percentage:.1f}%', ha='center', va='bottom')

    # 4. Model Performance Box Plot
    ax = axes[1, 0]
    mape_data = []
    model_labels = []

    for model_type, fold_metrics in all_results.items():
        mapes = [f['val_mape'] for f in fold_metrics]
        mape_data.append(mapes)
        model_labels.append(model_type.upper())

    bp = ax.boxplot(mape_data, labels=model_labels, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'lightcoral', 'lightyellow']
    for patch, color in zip(bp['boxes'], colors[:len(bp['boxes'])]):
        patch.set_facecolor(color)

    ax.set_ylabel('Validation MAPE (%)')
    ax.set_title('Model Performance Distribution')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45)

    # 5. Training Time vs Performance Scatter
    ax = axes[1, 1]

    for model_type, fold_metrics in all_results.items():
        times = [f['training_time'] for f in fold_metrics]
        mapes = [f['val_mape'] for f in fold_metrics]
        ax.scatter(times, mapes, label=model_type.upper(), alpha=0.7, s=50)

    ax.set_xlabel('Training Time (seconds)')
    ax.set_ylabel('Validation MAPE (%)')
    ax.set_title('Training Time vs Performance')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 6. Best Parameters Summary
    ax = axes[1, 2]
    ax.axis('off')

    params_text = "Best Parameters by Model:\n"
    params_text += "─" * 30 + "\n\n"

    for model_type, fold_metrics in all_results.items():
        best_fold = min(fold_metrics, key=lambda x: x['val_mape'])
        params_text += f"{model_type.upper()}:\n"
        params_text += f"  • Fold: {best_fold['fold']}\n"
        params_text += f"  • MAPE: {best_fold['val_mape']:.2f}%\n"
        params_text += f"  • Degree: {best_fold['polynomial_degree']}\n"

        # Extract regularization parameter if available
        if 'alpha' in str(best_fold['best_params']):
            alpha_key = [k for k in best_fold['best_params'].keys()
                         if 'alpha' in k]
            if alpha_key:
                alpha_val = best_fold['best_params'][alpha_key[0]]
                params_text += f"  • Alpha: {alpha_val}\n"

        if 'l1_ratio' in str(best_fold['best_params']):
            l1_ratio = best_fold['best_params'].get(
                'elasticnet__l1_ratio', 'N/A')
            params_text += f"  • L1 Ratio: {l1_ratio}\n"

        params_text += "\n"

    params_text += f"Overall Best: {best_model_type.upper()}\n"
    params_text += f"Best MAPE: {best_fold['val_mape']:.2f}%"

    ax.text(0.05, 0.95, params_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightyellow", alpha=0.8))

    plt.tight_layout()
    plt.savefig('polynomial_detailed_analysis.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print_with_timestamp(
        "✅ Detailed analysis saved: polynomial_detailed_analysis.png")


# Generate all visualizations
print_with_timestamp(
    "\n🎨 Generating comprehensive polynomial regression visualizations...")
print_with_timestamp("=" * 60)

try:
    create_polynomial_evaluation_dashboard()
    create_detailed_polynomial_analysis()

    print_with_timestamp("\n✅ All visualizations created successfully!")
    print_with_timestamp("📊 Generated files:")
    print_with_timestamp(
        "  • polynomial_evaluation_dashboard.png - Main evaluation dashboard")
    print_with_timestamp(
        "  • polynomial_detailed_analysis.png - Detailed analysis")

except Exception as e:
    print_with_timestamp(f"❌ Error creating visualizations: {e}")
    print_with_timestamp("Creating fallback basic visualization...")

    # Fallback basic visualization
    plt.figure(figsize=(12, 8))

    # Plot 1: Model comparison
    plt.subplot(2, 2, 1)
    model_names = [comp['model_type'].upper() for comp in model_comparison]
    val_mapes = [comp['avg_val_mape'] for comp in model_comparison]
    plt.bar(model_names, val_mapes, alpha=0.7)
    plt.ylabel('Average Validation MAPE (%)')
    plt.title('Model Performance Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    # Plot 2: Best model performance by fold
    plt.subplot(2, 2, 2)
    best_fold_results = all_results[best_model_type]
    folds = [f['fold'] for f in best_fold_results]
    fold_mapes = [f['val_mape'] for f in best_fold_results]
    plt.plot(folds, fold_mapes, 'o-', linewidth=2, markersize=8)
    plt.xlabel('Fold')
    plt.ylabel('Validation MAPE (%)')
    plt.title(f'Best Model ({best_model_type.upper()}) Performance')
    plt.grid(True, alpha=0.3)

    # Plot 3: Polynomial degree analysis
    plt.subplot(2, 2, 3)
    degrees_data = {}
    for model_type, fold_metrics in all_results.items():
        for fold_metric in fold_metrics:
            degree = fold_metric['polynomial_degree']
            if degree not in degrees_data:
                degrees_data[degree] = []
            degrees_data[degree].append(fold_metric['val_mape'])

    degrees = sorted(degrees_data.keys())
    avg_mapes = [np.mean(degrees_data[d]) for d in degrees]
    plt.bar(degrees, avg_mapes, alpha=0.7)
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Average Validation MAPE (%)')
    plt.title('Performance vs Polynomial Degree')
    plt.grid(True, alpha=0.3)

    # Plot 4: Training time comparison
    plt.subplot(2, 2, 4)
    training_times = [comp['avg_training_time'] for comp in model_comparison]
    plt.bar(model_names, training_times, alpha=0.7, color='purple')
    plt.ylabel('Average Training Time (seconds)')
    plt.title('Training Time Comparison')
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('polynomial_training_results.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print_with_timestamp(
        "Basic visualization saved to: polynomial_training_results.png")

print_with_timestamp(f"\n🎉 Polynomial Regression Training Complete!")
print_with_timestamp(
    f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print_with_timestamp(f"🏆 Best model: {best_model_type.upper()}")
print_with_timestamp(f"📊 Best performance: {best_fold['val_mape']:.2f}% MAPE")
print_with_timestamp(
    f"📐 Best polynomial degree: {best_fold['polynomial_degree']}")
print_with_timestamp(f"Script completed at: {datetime.now()}")

print_with_timestamp(f"\n📁 Generated Files:")
print_with_timestamp(f"  • best_polynomial_model.joblib - Best trained model")
print_with_timestamp(f"  • polynomial_training_results.csv - Detailed results")
print_with_timestamp(f"  • polynomial_summary_metrics.csv - Summary metrics")
print_with_timestamp(f"  • Individual fold models: poly_*_model_fold_*.joblib")
print_with_timestamp(f"  • Visualization plots: polynomial_*.png")
