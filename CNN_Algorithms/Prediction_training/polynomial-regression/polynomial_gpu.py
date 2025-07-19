import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import os
import sys
from datetime import datetime

# PyTorch imports for GPU computation
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Sklearn imports (only for preprocessing and metrics)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# For saving models
import joblib

# Function to print with timestamp


def print_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()


print_with_timestamp(
    "Starting GPU Polynomial Regression CNN Execution Time Prediction Training")

# Check for GPU availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print_with_timestamp(f"Using device: {device}")
if torch.cuda.is_available():
    print_with_timestamp(f"GPU: {torch.cuda.get_device_name(0)}")
    print_with_timestamp(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print_with_timestamp("WARNING: CUDA not available, falling back to CPU")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)


class PolynomialFeatures(nn.Module):
    """PyTorch implementation of polynomial features"""

    def __init__(self, degree=2, include_bias=False):
        super(PolynomialFeatures, self).__init__()
        self.degree = degree
        self.include_bias = include_bias

    def forward(self, x):
        # x shape: (batch_size, num_features)
        batch_size, num_features = x.shape
        features = [x]  # degree 1 features

        # Generate polynomial features up to specified degree
        for d in range(2, self.degree + 1):
            # For simplicity, we'll create powers of individual features
            # and some interaction terms
            power_features = torch.pow(x, d)
            features.append(power_features)

        # Add interaction terms for degree >= 2
        if self.degree >= 2:
            for i in range(num_features):
                for j in range(i + 1, num_features):
                    interaction = x[:, i:i+1] * x[:, j:j+1]
                    features.append(interaction)

        # Concatenate all features
        poly_features = torch.cat(features, dim=1)

        # Add bias term if requested
        if self.include_bias:
            bias = torch.ones(batch_size, 1, device=x.device)
            poly_features = torch.cat([bias, poly_features], dim=1)

        return poly_features


class PolynomialRegressionModel(nn.Module):
    """GPU-accelerated Polynomial Regression with regularization"""

    def __init__(self, input_dim, degree=2, reg_type='ridge', alpha=1.0, l1_ratio=0.5):
        super(PolynomialRegressionModel, self).__init__()
        self.degree = degree
        self.reg_type = reg_type
        self.alpha = alpha
        self.l1_ratio = l1_ratio

        # Create polynomial features layer
        self.poly_features = PolynomialFeatures(
            degree=degree, include_bias=False)

        # Calculate output dimension after polynomial expansion
        # This is an approximation - we'll set it properly after seeing the data
        self.poly_dim = None
        self.linear = None

    def forward(self, x):
        # Generate polynomial features
        poly_x = self.poly_features(x)

        # Initialize linear layer if not done yet
        if self.linear is None:
            self.poly_dim = poly_x.shape[1]
            self.linear = nn.Linear(self.poly_dim, 1, bias=True).to(x.device)

        # Linear transformation
        output = self.linear(poly_x)
        return output.squeeze()

    def get_regularization_loss(self):
        """Calculate regularization loss"""
        if self.linear is None:
            return torch.tensor(0.0)

        weights = self.linear.weight

        if self.reg_type == 'ridge':
            return self.alpha * torch.sum(weights ** 2)
        elif self.reg_type == 'lasso':
            return self.alpha * torch.sum(torch.abs(weights))
        elif self.reg_type == 'elasticnet':
            l2_penalty = (1 - self.l1_ratio) * torch.sum(weights ** 2)
            l1_penalty = self.l1_ratio * torch.sum(torch.abs(weights))
            return self.alpha * (l2_penalty + l1_penalty)
        else:  # linear (no regularization)
            return torch.tensor(0.0)


def train_gpu_model(model, train_loader, val_loader, num_epochs=1000, lr=0.01, patience=50):
    """Train polynomial regression model on GPU"""
    model = model.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for batch_x, batch_y in train_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            outputs = model(batch_x)

            # Main loss
            mse_loss = criterion(outputs, batch_y)
            # Regularization loss
            reg_loss = model.get_regularization_loss()
            total_loss = mse_loss + reg_loss

            total_loss.backward()
            optimizer.step()

            train_loss += total_loss.item()

        train_loss /= len(train_loader)
        train_losses.append(train_loss)

        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                batch_x, batch_y = batch_x.to(device), batch_y.to(device)
                outputs = model(batch_x)
                val_loss += criterion(outputs, batch_y).item()

        val_loss /= len(val_loader)
        val_losses.append(val_loss)

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= patience:
            print_with_timestamp(f"Early stopping at epoch {epoch+1}")
            break

    return train_losses, val_losses, best_val_loss


def evaluate_gpu_model(model, data_loader):
    """Evaluate GPU model and return predictions"""
    model.eval()
    all_predictions = []
    all_actuals = []

    with torch.no_grad():
        for batch_x, batch_y in data_loader:
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)
            outputs = model(batch_x)

            all_predictions.extend(outputs.cpu().numpy())
            all_actuals.extend(batch_y.cpu().numpy())

    return np.array(all_predictions), np.array(all_actuals)


def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def evaluate_model_metrics(y_true, y_pred):
    """Calculate evaluation metrics"""
    mape = calculate_mape(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    return {
        'mape': mape,
        'mae': mae,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2,
        'predictions': y_pred
    }


def print_metrics(metrics, title="Results"):
    print_with_timestamp(f"{title}:")
    print_with_timestamp(f"  MAPE: {metrics['mape']:.2f}%")
    print_with_timestamp(f"  MAE:  {metrics['mae']:.4f}")
    print_with_timestamp(f"  RMSE: {metrics['rmse']:.4f}")
    print_with_timestamp(f"  R²:   {metrics['r2']:.4f}")


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
X = df_scaled[feature_cols].values
y = df_scaled[target_col].values

print_with_timestamp(
    f"Data preprocessing complete. Features shape: {X.shape}, Target shape: {y.shape}")

# Convert to PyTorch tensors
X_tensor = torch.FloatTensor(X)
y_tensor = torch.FloatTensor(y)

# K-Fold Cross Validation Setup
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

print_with_timestamp(f"Training with {k}-Fold Cross Validation on GPU")
print_with_timestamp(f"Total samples: {len(X)}")
print_with_timestamp(f"Features: {X.shape[1]}")

# GPU Hyperparameter configurations
gpu_param_configs = {
    'ridge': [
        {'degree': 1, 'alpha': 0.01, 'lr': 0.01},
        {'degree': 1, 'alpha': 0.1, 'lr': 0.01},
        {'degree': 1, 'alpha': 1.0, 'lr': 0.01},
        {'degree': 2, 'alpha': 0.01, 'lr': 0.005},
        {'degree': 2, 'alpha': 0.1, 'lr': 0.005},
        {'degree': 2, 'alpha': 1.0, 'lr': 0.005},
        {'degree': 3, 'alpha': 0.1, 'lr': 0.001},
        {'degree': 3, 'alpha': 1.0, 'lr': 0.001},
        {'degree': 4, 'alpha': 1.0, 'lr': 0.001},
    ],
    'lasso': [
        {'degree': 1, 'alpha': 0.01, 'lr': 0.01},
        {'degree': 1, 'alpha': 0.1, 'lr': 0.01},
        {'degree': 1, 'alpha': 1.0, 'lr': 0.01},
        {'degree': 2, 'alpha': 0.01, 'lr': 0.005},
        {'degree': 2, 'alpha': 0.1, 'lr': 0.005},
        {'degree': 2, 'alpha': 1.0, 'lr': 0.005},
        {'degree': 3, 'alpha': 0.1, 'lr': 0.001},
        {'degree': 3, 'alpha': 1.0, 'lr': 0.001},
    ],
    'elasticnet': [
        {'degree': 1, 'alpha': 0.01, 'l1_ratio': 0.1, 'lr': 0.01},
        {'degree': 1, 'alpha': 0.1, 'l1_ratio': 0.5, 'lr': 0.01},
        {'degree': 1, 'alpha': 1.0, 'l1_ratio': 0.7, 'lr': 0.01},
        {'degree': 2, 'alpha': 0.01, 'l1_ratio': 0.3, 'lr': 0.005},
        {'degree': 2, 'alpha': 0.1, 'l1_ratio': 0.5, 'lr': 0.005},
        {'degree': 2, 'alpha': 1.0, 'l1_ratio': 0.7, 'lr': 0.005},
        {'degree': 3, 'alpha': 0.1, 'l1_ratio': 0.5, 'lr': 0.001},
        {'degree': 3, 'alpha': 1.0, 'l1_ratio': 0.7, 'lr': 0.001},
    ],
    'linear': [
        {'degree': 1, 'alpha': 0.0, 'lr': 0.01},
        {'degree': 2, 'alpha': 0.0, 'lr': 0.005},
        {'degree': 3, 'alpha': 0.0, 'lr': 0.001},
        {'degree': 4, 'alpha': 0.0, 'lr': 0.001},
        {'degree': 5, 'alpha': 0.0, 'lr': 0.0005},
    ]
}

# Store results from all folds and models
all_results = {}
trained_models = {}
model_types = ['ridge', 'lasso', 'elasticnet', 'linear']

print_with_timestamp(
    f"Will test {len(model_types)} model types: {model_types}")

# Start timing
start_time = time.time()

# Loop through each model type
for model_type in model_types:
    print_with_timestamp(f"\n{'='*60}")
    print_with_timestamp(
        f"🚀 TRAINING {model_type.upper()} POLYNOMIAL REGRESSION ON GPU")
    print_with_timestamp(f"{'='*60}")

    fold_metrics = []
    fold_models = []

    # Loop through each fold for this model type
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print_with_timestamp(f"\n🚀 {model_type.upper()} - FOLD {fold}")
        print_with_timestamp("-" * 40)

        # Get train and validation data for this fold
        X_train_fold = X_tensor[train_idx]
        X_val_fold = X_tensor[val_idx]
        y_train_fold = y_tensor[train_idx]
        y_val_fold = y_tensor[val_idx]

        print_with_timestamp(f"Training samples: {len(X_train_fold)}")
        print_with_timestamp(f"Validation samples: {len(X_val_fold)}")

        # Create data loaders
        batch_size = min(64, len(X_train_fold) // 4)  # Adaptive batch size
        train_dataset = TensorDataset(X_train_fold, y_train_fold)
        val_dataset = TensorDataset(X_val_fold, y_val_fold)
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False)

        # Hyperparameter search for this model type
        best_val_loss = float('inf')
        best_model = None
        best_params = None

        configs = gpu_param_configs[model_type]
        print_with_timestamp(
            f"Testing {len(configs)} hyperparameter configurations...")

        for config_idx, config in enumerate(configs):
            # Create model
            degree = config['degree']
            alpha = config['alpha']
            lr = config['lr']
            l1_ratio = config.get('l1_ratio', 0.5)

            model = PolynomialRegressionModel(
                input_dim=X.shape[1],
                degree=degree,
                reg_type=model_type,
                alpha=alpha,
                l1_ratio=l1_ratio
            )

            # Train model
            try:
                train_losses, val_losses, val_loss = train_gpu_model(
                    model, train_loader, val_loader,
                    num_epochs=500, lr=lr, patience=30
                )

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model = model
                    best_params = config.copy()

            except Exception as e:
                print_with_timestamp(f"Config {config_idx+1} failed: {e}")
                continue

        if best_model is None:
            print_with_timestamp(
                f"❌ All configurations failed for fold {fold}")
            continue

        fold_training_time = time.time() - start_time

        print_with_timestamp(f"✅ Training complete")
        print_with_timestamp(f"Best parameters: {best_params}")
        print_with_timestamp(f"Best validation loss: {best_val_loss:.4f}")

        # Evaluate on training and validation sets
        train_pred, train_actual = evaluate_gpu_model(best_model, train_loader)
        val_pred, val_actual = evaluate_gpu_model(best_model, val_loader)

        train_metrics = evaluate_model_metrics(train_actual, train_pred)
        val_metrics = evaluate_model_metrics(val_actual, val_pred)

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
            'best_val_loss': best_val_loss,
            'training_time': fold_training_time,
            'polynomial_degree': best_params['degree']
        }

        fold_metrics.append(fold_result)
        fold_models.append(best_model)

        # Print fold results
        print_with_timestamp(f"🚀 {model_type.upper()} Fold {fold} Results:")
        print_metrics(train_metrics, f"  Training")
        print_metrics(val_metrics, f"  Validation")
        print_with_timestamp(f"  Polynomial Degree: {best_params['degree']}")
        print_with_timestamp(f"  Alpha: {best_params['alpha']}")

        # Save model for this fold
        model_filename = f'poly_gpu_{model_type}_model_fold_{fold}.pt'
        torch.save(best_model.state_dict(), model_filename)
        print_with_timestamp(f"GPU model saved: {model_filename}")

    # Store results for this model type
    all_results[model_type] = fold_metrics
    trained_models[model_type] = fold_models

    if fold_metrics:  # Only calculate if we have results
        # Calculate average performance for this model type
        avg_val_mape = np.mean([f['val_mape'] for f in fold_metrics])
        avg_val_r2 = np.mean([f['val_r2'] for f in fold_metrics])
        std_val_mape = np.std([f['val_mape'] for f in fold_metrics])
        avg_training_time = np.mean([f['training_time'] for f in fold_metrics])

        print_with_timestamp(f"\n📊 {model_type.upper()} GPU SUMMARY:")
        print_with_timestamp(
            f"Average Val MAPE: {avg_val_mape:.2f}% ± {std_val_mape:.2f}%")
        print_with_timestamp(f"Average Val R²: {avg_val_r2:.4f}")
        print_with_timestamp(
            f"Average Training Time: {avg_training_time:.1f} seconds per fold")

# Calculate total training time
total_time = time.time() - start_time

# Find best overall model
print_with_timestamp(f"\n{'='*80}")
print_with_timestamp("🚀 GPU POLYNOMIAL REGRESSION CROSS VALIDATION SUMMARY")
print_with_timestamp(f"{'='*80}")

# Compare all models
model_comparison = []
for model_type, fold_metrics in all_results.items():
    if not fold_metrics:  # Skip if no successful results
        continue

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

if not model_comparison:
    print_with_timestamp("❌ No successful model training completed")
    sys.exit(1)

# Sort by validation MAPE (lower is better)
model_comparison.sort(key=lambda x: x['avg_val_mape'])

print_with_timestamp(
    "📊 GPU Model Performance Comparison (sorted by Val MAPE):")
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

print_with_timestamp(f"\n🏆 Best Overall GPU Model: {best_model_type.upper()}")
print_with_timestamp(
    f"🎯 Best Fold: Fold {best_fold['fold']} (Val MAPE: {best_fold['val_mape']:.2f}%)")
print_with_timestamp(
    f"📐 Best Polynomial Degree: {best_fold['polynomial_degree']}")
print_with_timestamp(
    f"⏱️  Total Training Time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")

# Save the best model
best_model_idx = best_fold['fold'] - 1
best_model = trained_models[best_model_type][best_model_idx]
torch.save(best_model.state_dict(), 'best_polynomial_gpu_model.pt')
print_with_timestamp(f"Best GPU model saved as: best_polynomial_gpu_model.pt")

# Save model architecture info
model_info = {
    'model_type': best_model_type,
    'input_dim': X.shape[1],
    'degree': best_fold['polynomial_degree'],
    'alpha': best_fold['best_params']['alpha'],
    'l1_ratio': best_fold['best_params'].get('l1_ratio', 0.5),
    'feature_cols': feature_cols,
    'scaler_params': {
        'mean_': scaler.mean_.tolist(),
        'scale_': scaler.scale_.tolist()
    }
}
joblib.dump(model_info, 'best_polynomial_gpu_model_info.pkl')
print_with_timestamp(
    f"Model info saved as: best_polynomial_gpu_model_info.pkl")

# Save all results to CSV
all_fold_results = []
for model_type, fold_metrics in all_results.items():
    all_fold_results.extend(fold_metrics)

if all_fold_results:
    results_df = pd.DataFrame(all_fold_results)
    results_df.to_csv('polynomial_gpu_training_results.csv', index=False)
    print_with_timestamp(
        f"Results saved to: polynomial_gpu_training_results.csv")

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
        'device': str(device),
        'timestamp': datetime.now().isoformat()
    })

summary_df = pd.DataFrame(summary_metrics)
summary_df.to_csv('polynomial_gpu_summary_metrics.csv', index=False)
print_with_timestamp(
    f"Summary metrics saved to: polynomial_gpu_summary_metrics.csv")

# ===============================
# GPU VISUALIZATION
# ===============================


def create_gpu_evaluation_dashboard():
    """Create comprehensive GPU polynomial regression evaluation dashboard"""
    print_with_timestamp(
        "📊 Creating GPU polynomial regression evaluation dashboard...")

    # Collect all predictions and actuals from best model
    if best_model_type not in trained_models or not trained_models[best_model_type]:
        print_with_timestamp("No trained models available for visualization")
        return

    best_models = trained_models[best_model_type]
    all_predictions = []
    all_actuals = []

    for fold, (train_idx, val_idx) in enumerate(kf.split(X)):
        if fold >= len(best_models):
            break

        # Get validation data for this fold
        X_val_fold = X_tensor[val_idx]
        y_val_fold = y_tensor[val_idx]

        val_dataset = TensorDataset(X_val_fold, y_val_fold)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

        # Get predictions
        val_pred, val_actual = evaluate_gpu_model(
            best_models[fold], val_loader)

        all_predictions.extend(val_pred)
        all_actuals.extend(val_actual)

    all_predictions = np.array(all_predictions)
    all_actuals = np.array(all_actuals)

    # Create main dashboard
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle(f'GPU Polynomial Regression Model Evaluation Dashboard\nBest Model: {best_model_type.upper()}',
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
    ax.set_title(f'GPU Predictions vs Actual ({best_model_type.upper()})')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Add R² annotation
    r2_overall = r2_score(all_actuals, all_predictions)
    ax.text(0.05, 0.95, f'R² = {r2_overall:.4f}', transform=ax.transAxes,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))

    # 2. Model Comparison
    ax = axes[0, 1]
    model_names = [comp['model_type'].upper() for comp in model_comparison]
    val_mapes = [comp['avg_val_mape'] for comp in model_comparison]

    bars = ax.bar(model_names, val_mapes, alpha=0.7, color='skyblue')
    bars[0].set_color('red')
    bars[0].set_alpha(1.0)

    ax.set_xlabel('GPU Model Type')
    ax.set_ylabel('Average Validation MAPE (%)')
    ax.set_title('GPU Model Performance Comparison')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45)

    # 3. Error Distribution
    ax = axes[0, 2]
    percentage_errors = np.abs(
        (all_predictions - all_actuals) / all_actuals) * 100
    ax.hist(percentage_errors, bins=30, alpha=0.7,
            edgecolor='black', color='lightcoral')
    ax.axvline(x=np.mean(percentage_errors), color='red', linestyle='--', lw=2,
               label=f'Mean: {np.mean(percentage_errors):.2f}%')
    ax.axvline(x=np.median(percentage_errors), color='green', linestyle='--', lw=2,
               label=f'Median: {np.median(percentage_errors):.2f}%')
    ax.set_xlabel('Percentage Error (%)')
    ax.set_ylabel('Frequency')
    ax.set_title('GPU Model Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # 4. Training Time Comparison
    ax = axes[1, 0]
    training_times = [comp['avg_training_time'] for comp in model_comparison]
    bars = ax.bar(model_names, training_times, alpha=0.7, color='purple')
    ax.set_xlabel('GPU Model Type')
    ax.set_ylabel('Average Training Time (seconds)')
    ax.set_title('GPU Training Time Comparison')
    ax.grid(True, alpha=0.3)
    plt.setp(ax.get_xticklabels(), rotation=45)

    # 5. Cross-Validation Results for Best Model
    ax = axes[1, 1]
    if best_model_type in all_results and all_results[best_model_type]:
        best_fold_results = all_results[best_model_type]
        folds = [f['fold'] for f in best_fold_results]
        fold_mapes = [f['val_mape'] for f in best_fold_results]

        ax.plot(folds, fold_mapes, 'o-', linewidth=3,
                markersize=8, color='orange')
        ax.set_xlabel('Fold')
        ax.set_ylabel('Validation MAPE (%)')
        ax.set_title(f'GPU CV Performance ({best_model_type.upper()})')
        ax.grid(True, alpha=0.3)

    # 6. Device and Performance Summary
    ax = axes[1, 2]
    ax.axis('off')

    summary_text = f"""
GPU Polynomial Regression Summary:
─────────────────────────────────
Device: {device}
Best Model: {best_model_type.upper()}
Best Fold: {best_fold['fold']}
Best Polynomial Degree: {best_fold['polynomial_degree']}

Performance Metrics:
• Best Val MAPE: {best_fold['val_mape']:.2f}%
• Best Val R²: {best_fold['val_r2']:.4f}
• Best Val MAE: {best_fold['val_mae']:.4f}

Training Efficiency:
• Total Time: {total_time:.1f}s ({total_time/60:.1f} min)
• Samples: {len(all_actuals)} total
• CV Folds: {k}
• Models Tested: {len([m for m in all_results.keys() if all_results[m]])}

GPU Acceleration:
• Faster training with PyTorch
• Batch processing enabled
• Memory efficient operations
"""

    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightcyan", alpha=0.8))

    plt.tight_layout()
    plt.savefig('polynomial_gpu_evaluation_dashboard.png',
                dpi=300, bbox_inches='tight')
    plt.close()
    print_with_timestamp(
        "✅ GPU evaluation dashboard saved: polynomial_gpu_evaluation_dashboard.png")


# Generate visualization
try:
    create_gpu_evaluation_dashboard()
except Exception as e:
    print_with_timestamp(f"Error creating GPU visualization: {e}")

print_with_timestamp(f"\n🎉 GPU Polynomial Regression Training Complete!")
print_with_timestamp(
    f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
print_with_timestamp(f"🚀 Device used: {device}")
if model_comparison:
    print_with_timestamp(f"🏆 Best model: {best_model_type.upper()}")
    print_with_timestamp(
        f"📊 Best performance: {best_fold['val_mape']:.2f}% MAPE")
    print_with_timestamp(
        f"📐 Best polynomial degree: {best_fold['polynomial_degree']}")
print_with_timestamp(f"Script completed at: {datetime.now()}")

print_with_timestamp(f"\n📁 Generated GPU Files:")
print_with_timestamp(
    f"  • best_polynomial_gpu_model.pt - Best trained GPU model")
print_with_timestamp(
    f"  • best_polynomial_gpu_model_info.pkl - Model architecture info")
print_with_timestamp(
    f"  • polynomial_gpu_training_results.csv - Detailed results")
print_with_timestamp(
    f"  • polynomial_gpu_summary_metrics.csv - Summary metrics")
print_with_timestamp(f"  • Individual fold models: poly_gpu_*_model_fold_*.pt")
print_with_timestamp(
    f"  • Visualization: polynomial_gpu_evaluation_dashboard.png")
