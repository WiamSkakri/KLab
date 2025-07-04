import pandas as pd
from sklearn.preprocessing import StandardScaler
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

# Set up logging and output


def print_with_timestamp(message):
    """Print message with timestamp for better logging"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {message}")
    sys.stdout.flush()


print_with_timestamp("Starting CNN Execution Time Prediction Training")

# Check CUDA availability
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print_with_timestamp(f"Using device: {device}")

if torch.cuda.is_available():
    print_with_timestamp(f"GPU: {torch.cuda.get_device_name(0)}")
    print_with_timestamp(
        f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print_with_timestamp(f"CUDA version: {torch.version.cuda}")
else:
    print_with_timestamp("CUDA not available, using CPU")

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# File path handling for HPC
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

# Dataset class optimized for HPC


class CNNExecutionDataset(Dataset):
    def __init__(self, X, y):
        """
        X: DataFrame with features (one-hot encoded + scaled numerical)
        y: Series with target values (execution times)
        """
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# K-Fold Cross Validation Setup
k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

print_with_timestamp(f"Training with {k}-Fold Cross Validation")
print_with_timestamp(f"Total samples: {len(X)}")
print_with_timestamp(f"Features: {X.shape[1]}")

# Store results for each fold
fold_results = []

# Train model for each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print_with_timestamp(f"Preparing FOLD {fold}")

    # Get train and validation data for this fold
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]

    # Create datasets for this fold
    train_dataset = CNNExecutionDataset(X_train_fold, y_train_fold)
    val_dataset = CNNExecutionDataset(X_val_fold, y_val_fold)

    # Create data loaders with pin_memory for GPU optimization
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True,
                              pin_memory=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False,
                            pin_memory=True, num_workers=0)

    print_with_timestamp(f"Training samples: {len(train_dataset)}")
    print_with_timestamp(f"Validation samples: {len(val_dataset)}")

    # Store for training
    fold_results.append({
        'fold': fold,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset)
    })

# Enhanced model class with better GPU utilization


class CNNExecutionPredictor(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super(CNNExecutionPredictor, self).__init__()

        # Enhanced architecture for better GPU utilization
        self.layer1 = nn.Linear(input_size, 256)
        self.batch_norm1 = nn.BatchNorm1d(256)
        self.layer2 = nn.Linear(256, 128)
        self.batch_norm2 = nn.BatchNorm1d(128)
        self.layer3 = nn.Linear(128, 64)
        self.batch_norm3 = nn.BatchNorm1d(64)
        self.layer4 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Layer 1
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.layer3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 4
        x = self.layer4(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output (no activation for regression)
        x = self.output(x)
        return x

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

    # Calculate metrics
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


# Training parameters
epochs = 150
patience = 20  # Early stopping patience
input_size = X.shape[1]
learning_rate = 0.001

# Store results from all folds
fold_metrics = []
trained_models = []

print_with_timestamp(
    f"Starting {len(fold_results)}-Fold Cross Validation Training")
print_with_timestamp(f"Epochs: {epochs}, Early Stopping Patience: {patience}")
print_with_timestamp(f"Learning Rate: {learning_rate}")
print_with_timestamp("=" * 80)

# Track total training time
total_start_time = time.time()

for fold_data in fold_results:
    fold = fold_data['fold']
    train_loader = fold_data['train_loader']
    val_loader = fold_data['val_loader']

    print_with_timestamp(f"🔄 Starting FOLD {fold}")
    print_with_timestamp("-" * 40)

    # Track fold training time
    fold_start_time = time.time()

    # Create fresh model for this fold
    model = CNNExecutionPredictor(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, verbose=True)

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training loop for this fold
    for epoch in range(epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(
                device, non_blocking=True), y_batch.to(device, non_blocking=True)

            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            loss.backward()

            # Gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)

        # Validation phase
        val_metrics = validate_model(model, val_loader, criterion, device)
        val_loss = val_metrics['loss']

        # Learning rate scheduling
        scheduler.step(val_loss)

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        epoch_time = time.time() - epoch_start_time

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print_with_timestamp(f"  Epoch {epoch+1:3d}/{epochs} | "
                                 f"Train Loss: {avg_train_loss:.4f} | "
                                 f"Val Loss: {val_loss:.4f} | "
                                 f"Val MAPE: {val_metrics['mape']:.2f}% | "
                                 f"Time: {epoch_time:.2f}s")

        # Early stopping
        if patience_counter >= patience:
            print_with_timestamp(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation on this fold
    train_metrics = validate_model(model, train_loader, criterion, device)
    val_metrics = validate_model(model, val_loader, criterion, device)

    fold_time = time.time() - fold_start_time

    # Store results
    fold_result = {
        'fold': fold,
        'train_mape': train_metrics['mape'],
        'val_mape': val_metrics['mape'],
        'train_mae': train_metrics['mae'],
        'val_mae': val_metrics['mae'],
        'train_r2': train_metrics['r2'],
        'val_r2': val_metrics['r2'],
        'epochs_trained': epoch + 1,
        'best_val_loss': best_val_loss,
        'training_time': fold_time
    }

    fold_metrics.append(fold_result)
    trained_models.append(model)

    # Print fold results
    print_with_timestamp(
        f"  ✅ Fold {fold} Complete in {fold_time:.2f} seconds")
    print_with_timestamp(
        f"     Train MAPE: {train_metrics['mape']:.2f}% | Val MAPE: {val_metrics['mape']:.2f}%")
    print_with_timestamp(
        f"     Train MAE:  {train_metrics['mae']:.4f} | Val MAE:  {val_metrics['mae']:.4f}")
    print_with_timestamp(
        f"     Train R²:   {train_metrics['r2']:.4f} | Val R²:   {val_metrics['r2']:.4f}")
    print_with_timestamp(f"     Epochs:     {epoch + 1}/{epochs}")

total_time = time.time() - total_start_time

# Calculate average performance across all folds
print_with_timestamp("\n" + "=" * 80)
print_with_timestamp("📊 CROSS VALIDATION SUMMARY")
print_with_timestamp("=" * 80)

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

print_with_timestamp(f"Average Train MAPE: {avg_train_mape:.2f}%")
print_with_timestamp(
    f"Average Val MAPE:   {avg_val_mape:.2f}% ± {std_val_mape:.2f}%")
print_with_timestamp(f"Average Train MAE:  {avg_train_mae:.4f}")
print_with_timestamp(
    f"Average Val MAE:    {avg_val_mae:.4f} ± {std_val_mae:.4f}")
print_with_timestamp(f"Average Train R²:   {avg_train_r2:.4f}")
print_with_timestamp(
    f"Average Val R²:     {avg_val_r2:.4f} ± {std_val_r2:.4f}")
print_with_timestamp(
    f"Average Training Time per Fold: {avg_training_time:.2f} seconds")
print_with_timestamp(
    f"Total Training Time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")

# Detailed results table
print_with_timestamp(f"\nDetailed Results by Fold:")
print_with_timestamp(
    f"{'Fold':<4} {'Train MAPE':<11} {'Val MAPE':<9} {'Train R²':<8} {'Val R²':<7} {'Epochs':<7} {'Time(s)':<8}")
print_with_timestamp("-" * 70)
for f in fold_metrics:
    print_with_timestamp(f"{f['fold']:<4} {f['train_mape']:<11.2f} {f['val_mape']:<9.2f} "
                         f"{f['train_r2']:<8.4f} {f['val_r2']:<7.4f} {f['epochs_trained']:<7} "
                         f"{f['training_time']:<8.2f}")

best_fold = min(fold_metrics, key=lambda x: x['val_mape'])
print_with_timestamp(f"\n🎉 Training Complete!")
print_with_timestamp(f"📈 Best performing fold: Fold {best_fold['fold']} "
                     f"(Val MAPE: {best_fold['val_mape']:.2f}%)")

# Save results to CSV
results_df = pd.DataFrame(fold_metrics)
results_df.to_csv('training_results.csv', index=False)
print_with_timestamp(f"Results saved to training_results.csv")

# Save the best model
best_model = trained_models[best_fold['fold'] - 1]
torch.save(best_model.state_dict(), 'best_model.pth')
print_with_timestamp(f"Best model saved to best_model.pth")

# GPU memory cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print_with_timestamp("GPU memory cleared")

print_with_timestamp("Script completed successfully!")
