# Improved Neural Network for CNN Execution Time Prediction
# Features: Skip connections, MAPE loss, advanced regularization

from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import KFold
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import os
import time
import sys
from datetime import datetime
import torch.nn.functional as F

# Function to print with timestamp


def print_with_timestamp(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()


print_with_timestamp(
    "Starting Improved CNN Execution Time Prediction Training")

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print_with_timestamp(f"Using device: {device}")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# File path handling
csv_file = 'combined.csv'
if not os.path.exists(csv_file):
    print_with_timestamp(f"Error: {csv_file} not found in current directory")
    sys.exit(1)

print_with_timestamp(f"Loading data from {csv_file}")
df = pd.read_csv(csv_file)
print_with_timestamp(f"Data loaded successfully. Shape: {df.shape}")

# One-hot encode the Algorithm column
df_encoded = pd.get_dummies(
    df, columns=['Algorithm'], prefix='Algorithm', dtype=int)
df = df_encoded

# Enhanced feature scaling using RobustScaler (better for outliers)
numerical_cols = ['Batch_Size', 'Input_Size', 'In_Channels', 'Out_Channels',
                  'Kernel_Size', 'Stride', 'Padding']
print_with_timestamp(
    f"Scaling numerical features with RobustScaler: {numerical_cols}")

# Use RobustScaler for better outlier handling
scaler = RobustScaler()  # More robust to outliers than StandardScaler
df_scaled = df.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Create polynomial features for non-linear interactions
print_with_timestamp(
    "Creating polynomial features for better non-linear modeling")
poly_features = PolynomialFeatures(
    degree=2, interaction_only=True, include_bias=False)
poly_cols = ['Batch_Size', 'Input_Size', 'In_Channels', 'Out_Channels']
poly_data = poly_features.fit_transform(df_scaled[poly_cols])

# Add polynomial features to the dataframe
poly_feature_names = poly_features.get_feature_names_out(poly_cols)
for i, name in enumerate(poly_feature_names):
    if name not in poly_cols:  # Don't duplicate original features
        df_scaled[f'poly_{name}'] = poly_data[:, i]

# Define feature columns
feature_cols = [col for col in df_scaled.columns if col != 'Execution_Time_ms']
target_col = 'Execution_Time_ms'

print_with_timestamp(f"Features: {len(feature_cols)}")
print_with_timestamp(
    f"Total features (including polynomial): {len(feature_cols)}")

# Create features and target arrays
X = df_scaled[feature_cols]
y = df_scaled[target_col]

print_with_timestamp(
    f"Final data shape: Features: {X.shape}, Target: {y.shape}")

# Enhanced Dataset class


class CNNExecutionDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# K-Fold Cross Validation Setup
k = 5
kf = KFold(n_splits=k, shuffle=True, random_state=42)

fold_results = []
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]

    train_dataset = CNNExecutionDataset(X_train_fold, y_train_fold)
    val_dataset = CNNExecutionDataset(X_val_fold, y_val_fold)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False,
                            pin_memory=True, num_workers=0)

    fold_results.append({
        'fold': fold,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset)
    })

# Enhanced model with skip connections and better architecture


class ImprovedCNNPredictor(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super(ImprovedCNNPredictor, self).__init__()

        # Input layer
        self.input_layer = nn.Linear(input_size, 512)
        self.input_bn = nn.BatchNorm1d(512)

        # Block 1: Skip connection
        self.block1_layer1 = nn.Linear(512, 512)
        self.block1_bn1 = nn.BatchNorm1d(512)
        self.block1_layer2 = nn.Linear(512, 512)
        self.block1_bn2 = nn.BatchNorm1d(512)

        # Block 2: Dimension reduction with skip
        self.block2_layer1 = nn.Linear(512, 256)
        self.block2_bn1 = nn.BatchNorm1d(256)
        self.block2_layer2 = nn.Linear(256, 256)
        self.block2_bn2 = nn.BatchNorm1d(256)

        # Block 3: Further reduction
        self.block3_layer1 = nn.Linear(256, 128)
        self.block3_bn1 = nn.BatchNorm1d(128)
        self.block3_layer2 = nn.Linear(128, 128)
        self.block3_bn2 = nn.BatchNorm1d(128)

        # Final layers
        self.final_layer1 = nn.Linear(128, 64)
        self.final_bn1 = nn.BatchNorm1d(64)
        self.final_layer2 = nn.Linear(64, 32)
        self.final_bn2 = nn.BatchNorm1d(32)
        self.output = nn.Linear(32, 1)

        # Activation functions
        self.relu = nn.ReLU()
        self.swish = nn.SiLU()  # Swish activation (often better than ReLU)
        self.dropout = nn.Dropout(dropout_rate)

        # Advanced regularization
        self.layer_norm = nn.LayerNorm(512)

        # Initialize weights
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        # Input processing
        x = self.input_layer(x)
        x = self.input_bn(x)
        x = self.swish(x)
        x = self.dropout(x)

        # Block 1 with skip connection
        identity1 = x
        x = self.block1_layer1(x)
        x = self.block1_bn1(x)
        x = self.swish(x)
        x = self.dropout(x)

        x = self.block1_layer2(x)
        x = self.block1_bn2(x)
        x = x + identity1  # Skip connection
        x = self.swish(x)
        x = self.dropout(x)

        # Block 2 with dimension change
        x = self.block2_layer1(x)
        x = self.block2_bn1(x)
        x = self.swish(x)
        x = self.dropout(x)

        identity2 = x
        x = self.block2_layer2(x)
        x = self.block2_bn2(x)
        x = x + identity2  # Skip connection
        x = self.swish(x)
        x = self.dropout(x)

        # Block 3
        x = self.block3_layer1(x)
        x = self.block3_bn1(x)
        x = self.swish(x)
        x = self.dropout(x)

        identity3 = x
        x = self.block3_layer2(x)
        x = self.block3_bn2(x)
        x = x + identity3  # Skip connection
        x = self.swish(x)
        x = self.dropout(x)

        # Final layers
        x = self.final_layer1(x)
        x = self.final_bn1(x)
        x = self.swish(x)
        x = self.dropout(x)

        x = self.final_layer2(x)
        x = self.final_bn2(x)
        x = self.swish(x)
        x = self.dropout(x)

        x = self.output(x)
        return x

# Custom MAPE Loss Function


class MAPELoss(nn.Module):
    def __init__(self, epsilon=1e-8):
        super(MAPELoss, self).__init__()
        self.epsilon = epsilon

    def forward(self, y_pred, y_true):
        return torch.mean(torch.abs((y_true - y_pred) / (torch.abs(y_true) + self.epsilon)))

# Combined Loss Function (MSE + MAPE)


class CombinedLoss(nn.Module):
    def __init__(self, mse_weight=0.7, mape_weight=0.3):
        super(CombinedLoss, self).__init__()
        self.mse_loss = nn.MSELoss()
        self.mape_loss = MAPELoss()
        self.mse_weight = mse_weight
        self.mape_weight = mape_weight

    def forward(self, y_pred, y_true):
        mse = self.mse_loss(y_pred, y_true)
        mape = self.mape_loss(y_pred, y_true)
        return self.mse_weight * mse + self.mape_weight * mape

# Evaluation functions


def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    if np.sum(mask) == 0:
        return 0.0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions, targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(
                device, non_blocking=True), y_batch.to(device, non_blocking=True)
            pred = model(X_batch).squeeze()
            loss = criterion(pred, y_batch)

            total_loss += loss.item()
            predictions.extend(pred.cpu().numpy())
            targets.extend(y_batch.cpu().numpy())

    avg_loss = total_loss / len(val_loader)
    mape = calculate_mape(targets, predictions)
    mae = mean_absolute_error(targets, predictions)
    mse = mean_squared_error(targets, predictions)
    r2 = r2_score(targets, predictions)

    return {
        'loss': avg_loss,
        'mape': mape,
        'mae': mae,
        'mse': mse,
        'rmse': np.sqrt(mse),
        'r2': r2
    }


# Enhanced Training Parameters
epochs = 300
patience = 50
input_size = X.shape[1]
learning_rate = 0.0003  # Slightly higher for better convergence

# Store results
fold_metrics = []
trained_models = []
all_fold_histories = []

print_with_timestamp(
    f"Starting Enhanced {len(fold_results)}-Fold Cross Validation Training")
print_with_timestamp(
    f"Architecture: Improved with skip connections and advanced features")
print_with_timestamp(
    f"Epochs: {epochs}, Patience: {patience}, Learning Rate: {learning_rate}")
print_with_timestamp("=" * 80)

total_start_time = time.time()

for fold_data in fold_results:
    fold = fold_data['fold']
    train_loader = fold_data['train_loader']
    val_loader = fold_data['val_loader']

    print_with_timestamp(f"Starting FOLD {fold}")
    print_with_timestamp("-" * 40)

    fold_start_time = time.time()

    # Create improved model
    model = ImprovedCNNPredictor(input_size).to(device)

    # Use combined loss function
    criterion = CombinedLoss()

    # Enhanced optimizer with better parameters
    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=1e-4)

    # Advanced learning rate scheduling
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=learning_rate*3, epochs=epochs,
        steps_per_epoch=len(train_loader), pct_start=0.3
    )

    # Early stopping
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training history
    fold_history = {
        'fold': fold,
        'train_losses': [],
        'val_losses': [],
        'train_mapes': [],
        'val_mapes': [],
        'val_r2s': [],
        'learning_rates': [],
        'epochs': []
    }

    # Training loop
    for epoch in range(epochs):
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(
                device, non_blocking=True), y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            scheduler.step()

            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation
        train_metrics = validate_model(model, train_loader, criterion, device)
        val_metrics = validate_model(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']

        # Store metrics
        fold_history['epochs'].append(epoch + 1)
        fold_history['train_losses'].append(avg_train_loss)
        fold_history['val_losses'].append(val_loss)
        fold_history['train_mapes'].append(train_metrics['mape'])
        fold_history['val_mapes'].append(val_metrics['mape'])
        fold_history['val_r2s'].append(val_metrics['r2'])
        fold_history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Early stopping based on validation MAPE
        if val_metrics['mape'] < best_val_loss:
            best_val_loss = val_metrics['mape']
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        # Print progress
        if (epoch + 1) % 20 == 0:
            print_with_timestamp(f"  Epoch {epoch+1:3d}/{epochs} | "
                                 f"Train Loss: {avg_train_loss:.4f} | "
                                 f"Val MAPE: {val_metrics['mape']:.2f}% | "
                                 f"Val R²: {val_metrics['r2']:.4f}")

        # Early stopping
        if patience_counter >= patience:
            print_with_timestamp(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation
    train_metrics = validate_model(model, train_loader, criterion, device)
    val_metrics = validate_model(model, val_loader, criterion, device)

    fold_time = time.time() - fold_start_time

    # Store results
    fold_result = {
        'fold': fold,
        'train_mape': train_metrics['mape'],
        'val_mape': val_metrics['mape'],
        'train_r2': train_metrics['r2'],
        'val_r2': val_metrics['r2'],
        'epochs_trained': epoch + 1,
        'training_time': fold_time
    }

    fold_metrics.append(fold_result)
    trained_models.append(model)
    all_fold_histories.append(fold_history)

    print_with_timestamp(f"  Fold {fold} Complete in {fold_time:.2f} seconds")
    print_with_timestamp(
        f"     Train MAPE: {train_metrics['mape']:.2f}% | Val MAPE: {val_metrics['mape']:.2f}%")
    print_with_timestamp(
        f"     Train R²: {train_metrics['r2']:.4f} | Val R²: {val_metrics['r2']:.4f}")

total_time = time.time() - total_start_time

# Results summary
print_with_timestamp("\n" + "=" * 80)
print_with_timestamp("📊 ENHANCED MODEL CROSS VALIDATION SUMMARY")
print_with_timestamp("=" * 80)

avg_val_mape = np.mean([f['val_mape'] for f in fold_metrics])
std_val_mape = np.std([f['val_mape'] for f in fold_metrics])
avg_val_r2 = np.mean([f['val_r2'] for f in fold_metrics])

print_with_timestamp(
    f"Average Val MAPE: {avg_val_mape:.2f}% ± {std_val_mape:.2f}%")
print_with_timestamp(f"Average Val R²: {avg_val_r2:.4f}")
print_with_timestamp(f"Total Training Time: {total_time/60:.2f} minutes")

best_fold = min(fold_metrics, key=lambda x: x['val_mape'])
print_with_timestamp(
    f"🎉 Best performing fold: Fold {best_fold['fold']} (Val MAPE: {best_fold['val_mape']:.2f}%)")

# Save results
results_df = pd.DataFrame(fold_metrics)
results_df.to_csv('improved_training_results.csv', index=False)

# Save best model
best_model = trained_models[best_fold['fold'] - 1]
torch.save(best_model.state_dict(), 'best_improved_model.pth')

print_with_timestamp(f"Results saved to improved_training_results.csv")
print_with_timestamp(f"Best model saved to best_improved_model.pth")

# GPU cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()

print_with_timestamp("Enhanced training completed successfully!")
