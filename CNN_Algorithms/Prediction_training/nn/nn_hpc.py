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

# Function to print with timestamp


def print_with_timestamp(message):
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

# Set random seeds for reproducibility when we train agains, that way we know that the changes we make are not due to random variations but to the code
torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed(42)

# File path handling for HPC
csv_file = 'combined.csv'
# in case the file is not found, we print the error and the current directory and the available files
if not os.path.exists(csv_file):
    print_with_timestamp(f"Error: {csv_file} not found in current directory")
    print_with_timestamp(f"Current directory: {os.getcwd()}")
    print_with_timestamp("Available files:")
    for file in os.listdir('.'):
        print(f"  - {file}")
    sys.exit(1)

# If the file is found, we print the message and the shape of the data
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

# Printing the number of features and the feature columns
print_with_timestamp(f"Features: {len(feature_cols)}")
print_with_timestamp(f"Feature columns: {feature_cols}")

# Create features and target arrays
X = df_scaled[feature_cols]
y = df_scaled[target_col]

print_with_timestamp(
    f"Data preprocessing complete. Features shape: {X.shape}, Target shape: {y.shape}")


# # ===========================
# Custom MAPE Loss Function
# ===========================
class MAPELoss(nn.Module):
    def __init__(self, eps=1e-8):
        super(MAPELoss, self).__init__()
        self.eps = eps

    def forward(self, y_pred, y_true):
        mask = y_true != 0
        return torch.mean(torch.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + self.eps))) * 100

# ===========================
# Custom hybrid loss function
# ===========================


class HybridLoss(nn.Module):
    def __init__(self, alpha=0.8, eps=1e-8):
        super(HybridLoss, self).__init__()
        self.alpha = alpha
        self.mape = MAPELoss(eps)
        self.mae = nn.L1Loss()

    def forward(self, y_pred, y_true):
        return self.alpha * self.mape(y_pred, y_true) + (1 - self.alpha) * self.mae(y_pred, y_true)


# Pytorch Dataset wrapper that prepare the data for the model
"""
Pandas DataFrame → CNNExecutionDataset → DataLoader → Neural Network
     (CSV data)      (PyTorch tensors)    (batches)    (training)
"""


class CNNExecutionDataset(Dataset):
    def __init__(self, X, y):
        """
        X: DataFrame with features (one-hot encoded + scaled numerical)
        y: Series with target values (execution times)
        """
        # FloatTensor is a Pytorch tensor that is a 2D tensor of floats
        # easily transfered to the GPU
        self.X = torch.FloatTensor(X.values)
        self.y = torch.FloatTensor(y.values)

    def __len__(self):
        # the length of the dataset is the number of rows in the dataframe
        # required by the Pytorch Dataset class for data processing
        return len(self.X)

    def __getitem__(self, idx):
        # returns the features and the target for the given index
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
    # batch_size is the number of samples to be processed in each batch
    # shuffle is a boolean that indicates if the data should be shuffled before each epoch
    # pin_memory is a boolean that indicates if the data should be pinned to the CPU memory
    # num_workers is the number of workers to be used to load the data, it is 0 here to not use multiple threads in the HPC
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

        # Enhanced architecture with much larger hidden dimensions for better capacity

        # Transforms the input features into a 2048-dimensional vector
        # Much wider first layer to capture more complex patterns in the data
        self.layer1 = nn.Linear(input_size, 2048)
        # Batch normalization is used to stabilize the training process
        # by normalizing the input to have zero mean and unit variance
        # Avoids vanishing gradients problem and exploding gradients problem
        self.batch_norm1 = nn.BatchNorm1d(2048)
        # The second layer reduces the dimensionality to 1024
        self.layer2 = nn.Linear(2048, 1024)
        self.batch_norm2 = nn.BatchNorm1d(1024)
        # Third layer reduces to 512
        self.layer3 = nn.Linear(1024, 512)
        self.batch_norm3 = nn.BatchNorm1d(512)
        # Fourth layer reduces to 256
        self.layer4 = nn.Linear(512, 256)
        self.batch_norm4 = nn.BatchNorm1d(256)
        # Fifth layer reduces to 128
        self.layer5 = nn.Linear(256, 128)
        self.batch_norm5 = nn.BatchNorm1d(128)
        # Sixth layer reduces to 64
        self.layer6 = nn.Linear(128, 64)
        self.output = nn.Linear(64, 1)

        # Activation and regularization
        self.relu = nn.ReLU()
        # Randomly dropping out 30% of the neurons to prevent overfitting
        # Active during training, inactive during inference
        # Avoids vanishing gradients problem
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Layer 1 (2048 units)
        x = self.layer1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 2 (1024 units)
        x = self.layer2(x)
        x = self.batch_norm2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 3 (512 units)
        x = self.layer3(x)
        x = self.batch_norm3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 4 (256 units)
        x = self.layer4(x)
        x = self.batch_norm4(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 5 (128 units)
        x = self.layer5(x)
        x = self.batch_norm5(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 6 (64 units)
        x = self.layer6(x)
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
min_epochs = 110
epochs = 250  # Increased from 150 for better convergence
patience = 50
input_size = X.shape[1]
learning_rate = 0.001

# Store results from all folds
fold_metrics = []
trained_models = []

# Store detailed training history for plotting
all_fold_histories = []

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

    print_with_timestamp(f"Starting FOLD {fold}")
    print_with_timestamp("-" * 40)

    # Track fold training time
    fold_start_time = time.time()

    # Create fresh model for this fold
    model = CNNExecutionPredictor(input_size).to(device)
    criterion = HybridLoss()
    optimizer = optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5, verbose=True)

    # Early stopping variables
    best_val_mape = float('inf')
    patience_counter = 0
    best_model_state = None

    # Track training history for plotting
    fold_history = {
        'fold': fold,
        'train_losses': [],
        'val_mapees': [],
        'train_mapes': [],
        'val_mapes': [],
        'train_maes': [],
        'val_maes': [],
        'train_r2s': [],
        'val_r2s': [],
        'learning_rates': [],
        'epochs': []
    }

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
        train_metrics = validate_model(model, train_loader, criterion, device)
        val_metrics = validate_model(model, val_loader, criterion, device)
        # val_mape = val_metrics['loss']
        val_mape = val_metrics['mape']

        # Store metrics for plotting
        fold_history['epochs'].append(epoch + 1)
        fold_history['train_losses'].append(avg_train_loss)
        fold_history['val_mapees'].append(val_mape)
        fold_history['train_mapes'].append(train_metrics['mape'])
        fold_history['val_mapes'].append(val_metrics['mape'])
        fold_history['train_maes'].append(train_metrics['mae'])
        fold_history['val_maes'].append(val_metrics['mae'])
        fold_history['train_r2s'].append(train_metrics['r2'])
        fold_history['val_r2s'].append(val_metrics['r2'])
        fold_history['learning_rates'].append(optimizer.param_groups[0]['lr'])

        # Learning rate scheduling
        # scheduler.step(val_mape)
        scheduler.step(val_mape)

        # Early stopping check
        if val_mape < best_val_mape:
            best_val_mape = val_mape
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1

        epoch_time = time.time() - epoch_start_time

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0:
            print_with_timestamp(f"  Epoch {epoch+1:3d}/{epochs} | "
                                 f"Train Loss: {avg_train_loss:.4f} | "
                                 f"Val Loss: {val_mape:.4f} | "
                                 f"Val MAPE: {val_metrics['mape']:.2f}% | "
                                 f"Time: {epoch_time:.2f}s")

        # Early stopping with min_epochs condition
        if (epoch + 1) > min_epochs and patience_counter >= patience:
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
        'best_val_mape': best_val_mape,
        'training_time': fold_time
    }

    fold_metrics.append(fold_result)
    trained_models.append(model)
    all_fold_histories.append(fold_history)

    # Print fold results
    print_with_timestamp(
        f"  Fold {fold} Complete in {fold_time:.2f} seconds")
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

# ==========================================
# COMPREHENSIVE PLOTTING AND VISUALIZATION
# ==========================================

print_with_timestamp("Creating evaluation plots...")

# Get the best fold history for detailed plotting
best_fold_history = all_fold_histories[best_fold['fold'] - 1]

# Collect predictions vs actual values from the best model for scatter plot
print_with_timestamp("Collecting predictions for visualization...")
best_model.eval()
all_predictions = []
all_actuals = []

# Use all data for the final evaluation plots
full_dataset = CNNExecutionDataset(X, y)
full_loader = DataLoader(full_dataset, batch_size=64,
                         shuffle=False, pin_memory=True)

with torch.no_grad():
    for X_batch, y_batch in full_loader:
        X_batch, y_batch = X_batch.to(
            device, non_blocking=True), y_batch.to(device, non_blocking=True)
        pred = best_model(X_batch).squeeze()
        all_predictions.extend(pred.cpu().numpy())
        all_actuals.extend(y_batch.cpu().numpy())

all_predictions = np.array(all_predictions)
all_actuals = np.array(all_actuals)
residuals = all_actuals - all_predictions

# 1. TRAINING & VALIDATION LOSS CURVES
plt.figure(figsize=(15, 10))

# Loss curves for best fold
plt.subplot(2, 3, 1)
plt.plot(best_fold_history['epochs'], best_fold_history['train_losses'],
         label='Training Loss', color='blue', linewidth=2)
plt.plot(best_fold_history['epochs'], best_fold_history['val_mapees'],
         label='Validation Loss', color='red', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.title(f'Training vs Validation Loss (Best Fold {best_fold["fold"]})')
plt.legend()
plt.grid(True, alpha=0.3)

# 2. PREDICTION VS ACTUAL SCATTER PLOT
plt.subplot(2, 3, 2)
plt.scatter(all_actuals, all_predictions, alpha=0.6, s=10)
min_val = min(all_actuals.min(), all_predictions.min())
max_val = max(all_actuals.max(), all_predictions.max())
plt.plot([min_val, max_val], [min_val, max_val],
         'r--', lw=2, label='Perfect Prediction')
plt.xlabel('Actual Execution Time (ms)')
plt.ylabel('Predicted Execution Time (ms)')
plt.title('Predictions vs Actual Values')
plt.legend()
plt.grid(True, alpha=0.3)

# Calculate R² for the plot
r2_score_val = r2_score(all_actuals, all_predictions)
plt.text(0.05, 0.95, f'R² = {r2_score_val:.4f}', transform=plt.gca().transAxes,
         bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

# 3. RESIDUAL PLOT
plt.subplot(2, 3, 3)
plt.scatter(all_predictions, residuals, alpha=0.6, s=10)
plt.axhline(y=0, color='r', linestyle='--', linewidth=2)
plt.xlabel('Predicted Values')
plt.ylabel('Residuals (Actual - Predicted)')
plt.title('Residual Plot')
plt.grid(True, alpha=0.3)

# 4. PERFORMANCE METRICS OVER EPOCHS (Best Fold)
plt.subplot(2, 3, 4)
plt.plot(best_fold_history['epochs'], best_fold_history['train_mapes'],
         label='Train MAPE', color='blue', linewidth=2)
plt.plot(best_fold_history['epochs'], best_fold_history['val_mapes'],
         label='Val MAPE', color='red', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MAPE (%)')
plt.title(f'MAPE Over Epochs (Best Fold {best_fold["fold"]})')
plt.legend()
plt.grid(True, alpha=0.3)

# 5. CROSS-VALIDATION RESULTS
plt.subplot(2, 3, 5)
folds = [f['fold'] for f in fold_metrics]
val_mapes = [f['val_mape'] for f in fold_metrics]
bars = plt.bar(folds, val_mapes, color=[
               'lightcoral' if f != best_fold['fold'] else 'darkgreen' for f in folds])
plt.xlabel('Fold')
plt.ylabel('Validation MAPE (%)')
plt.title('Cross-Validation Results')
plt.grid(True, alpha=0.3, axis='y')

# Highlight best fold
for i, (fold, mape) in enumerate(zip(folds, val_mapes)):
    if fold == best_fold['fold']:
        plt.text(fold, mape + 0.1, f'Best\n{mape:.2f}%', ha='center', va='bottom',
                 fontweight='bold', color='darkgreen')

# 6. ERROR DISTRIBUTION
plt.subplot(2, 3, 6)
percentage_errors = np.abs(residuals / all_actuals) * 100
plt.hist(percentage_errors, bins=30, alpha=0.7,
         color='skyblue', edgecolor='black')
plt.xlabel('Absolute Percentage Error (%)')
plt.ylabel('Frequency')
plt.title('Distribution of Absolute Percentage Errors')
plt.grid(True, alpha=0.3)

# Add statistics
mean_ape = np.mean(percentage_errors)
median_ape = np.median(percentage_errors)
plt.axvline(mean_ape, color='red', linestyle='--',
            linewidth=2, label=f'Mean: {mean_ape:.2f}%')
plt.axvline(median_ape, color='orange', linestyle='--',
            linewidth=2, label=f'Median: {median_ape:.2f}%')
plt.legend()

plt.tight_layout()
plt.savefig('nn_training_evaluation.png', dpi=300, bbox_inches='tight')
print_with_timestamp(
    "Main evaluation plot saved to nn_training_evaluation.png")

# ADDITIONAL DETAILED PLOTS

# Plot 7: R² and MAE over epochs
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.plot(best_fold_history['epochs'], best_fold_history['train_r2s'],
         label='Train R²', color='blue', linewidth=2)
plt.plot(best_fold_history['epochs'], best_fold_history['val_r2s'],
         label='Val R²', color='red', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('R² Score')
plt.title(f'R² Over Epochs (Best Fold {best_fold["fold"]})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 2)
plt.plot(best_fold_history['epochs'], best_fold_history['train_maes'],
         label='Train MAE', color='blue', linewidth=2)
plt.plot(best_fold_history['epochs'], best_fold_history['val_maes'],
         label='Val MAE', color='red', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('MAE')
plt.title(f'MAE Over Epochs (Best Fold {best_fold["fold"]})')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
plt.plot(best_fold_history['epochs'], best_fold_history['learning_rates'],
         color='green', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Learning Rate')
plt.title(f'Learning Rate Schedule (Best Fold {best_fold["fold"]})')
plt.grid(True, alpha=0.3)
plt.yscale('log')

plt.subplot(2, 2, 4)
plt.hist(residuals, bins=30, alpha=0.7, color='lightgreen', edgecolor='black')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.title('Distribution of Residuals')
plt.grid(True, alpha=0.3)

# Add statistics
mean_residual = np.mean(residuals)
std_residual = np.std(residuals)
plt.axvline(mean_residual, color='red', linestyle='--', linewidth=2,
            label=f'Mean: {mean_residual:.4f}')
plt.axvline(mean_residual + std_residual, color='orange', linestyle=':', linewidth=2,
            label=f'+1σ: {mean_residual + std_residual:.4f}')
plt.axvline(mean_residual - std_residual, color='orange', linestyle=':', linewidth=2,
            label=f'-1σ: {mean_residual - std_residual:.4f}')
plt.legend()

plt.tight_layout()
plt.savefig('nn_detailed_metrics.png', dpi=300, bbox_inches='tight')
print_with_timestamp("Detailed metrics plot saved to nn_detailed_metrics.png")

# Plot 8: Cross-validation comparison across all metrics
plt.figure(figsize=(15, 10))

# MAPE comparison
plt.subplot(2, 3, 1)
val_mapes = [f['val_mape'] for f in fold_metrics]
train_mapes = [f['train_mape'] for f in fold_metrics]
x = np.arange(len(folds))
width = 0.35
plt.bar(x - width/2, train_mapes, width, label='Train MAPE', alpha=0.8)
plt.bar(x + width/2, val_mapes, width, label='Val MAPE', alpha=0.8)
plt.xlabel('Fold')
plt.ylabel('MAPE (%)')
plt.title('MAPE Comparison Across Folds')
plt.xticks(x, folds)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# R² comparison
plt.subplot(2, 3, 2)
val_r2s = [f['val_r2'] for f in fold_metrics]
train_r2s = [f['train_r2'] for f in fold_metrics]
plt.bar(x - width/2, train_r2s, width, label='Train R²', alpha=0.8)
plt.bar(x + width/2, val_r2s, width, label='Val R²', alpha=0.8)
plt.xlabel('Fold')
plt.ylabel('R² Score')
plt.title('R² Comparison Across Folds')
plt.xticks(x, folds)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# MAE comparison
plt.subplot(2, 3, 3)
val_maes = [f['val_mae'] for f in fold_metrics]
train_maes = [f['train_mae'] for f in fold_metrics]
plt.bar(x - width/2, train_maes, width, label='Train MAE', alpha=0.8)
plt.bar(x + width/2, val_maes, width, label='Val MAE', alpha=0.8)
plt.xlabel('Fold')
plt.ylabel('MAE')
plt.title('MAE Comparison Across Folds')
plt.xticks(x, folds)
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

# Training times
plt.subplot(2, 3, 4)
training_times = [f['training_time'] for f in fold_metrics]
plt.bar(folds, training_times, color='lightblue', alpha=0.8)
plt.xlabel('Fold')
plt.ylabel('Training Time (seconds)')
plt.title('Training Time by Fold')
plt.grid(True, alpha=0.3, axis='y')

# Epochs trained
plt.subplot(2, 3, 5)
epochs_trained = [f['epochs_trained'] for f in fold_metrics]
plt.bar(folds, epochs_trained, color='lightcoral', alpha=0.8)
plt.xlabel('Fold')
plt.ylabel('Epochs Trained')
plt.title('Epochs Trained by Fold')
plt.grid(True, alpha=0.3, axis='y')

# Summary statistics
plt.subplot(2, 3, 6)
plt.text(0.1, 0.9, f"Cross-Validation Summary", fontsize=14, fontweight='bold')
plt.text(
    0.1, 0.8, f"Average Val MAPE: {avg_val_mape:.2f}% ± {std_val_mape:.2f}%")
plt.text(0.1, 0.7, f"Average Val MAE: {avg_val_mae:.4f} ± {std_val_mae:.4f}")
plt.text(0.1, 0.6, f"Average Val R²: {avg_val_r2:.4f} ± {std_val_r2:.4f}")
plt.text(
    0.1, 0.5, f"Best Fold: {best_fold['fold']} (MAPE: {best_fold['val_mape']:.2f}%)")
plt.text(0.1, 0.4, f"Total Training Time: {total_time/60:.2f} minutes")
plt.text(0.1, 0.3, f"Average Time per Fold: {avg_training_time:.2f} seconds")
plt.axis('off')

plt.tight_layout()
plt.savefig('nn_cross_validation_comparison.png', dpi=300, bbox_inches='tight')
print_with_timestamp(
    "Cross-validation comparison plot saved to nn_cross_validation_comparison.png")

# Print summary of generated plots
print_with_timestamp("\n" + "=" * 80)
print_with_timestamp("📊 VISUALIZATION SUMMARY")
print_with_timestamp("=" * 80)
print_with_timestamp("Generated evaluation plots:")
print_with_timestamp(
    "  1. nn_training_evaluation.png - Main evaluation dashboard (6 plots)")
print_with_timestamp("     • Training/Validation loss curves")
print_with_timestamp("     • Prediction vs Actual scatter plot")
print_with_timestamp("     • Residual analysis")
print_with_timestamp("     • MAPE over epochs")
print_with_timestamp("     • Cross-validation results")
print_with_timestamp("     • Error distribution")
print_with_timestamp(
    "  2. nn_detailed_metrics.png - Detailed metrics (4 plots)")
print_with_timestamp("     • R² progression")
print_with_timestamp("     • MAE progression")
print_with_timestamp("     • Learning rate schedule")
print_with_timestamp("     • Residual distribution")
print_with_timestamp(
    "  3. nn_cross_validation_comparison.png - CV analysis (6 plots)")
print_with_timestamp("     • Metric comparisons across folds")
print_with_timestamp("     • Training times and epochs")
print_with_timestamp("     • Summary statistics")

# GPU memory cleanup
if torch.cuda.is_available():
    torch.cuda.empty_cache()
    print_with_timestamp("GPU memory cleared")

print_with_timestamp("Script completed successfully!")
