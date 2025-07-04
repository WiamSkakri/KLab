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
import torch.optim as optim

df = pd.read_csv('combined.csv')

# One-hot encode the Algorithm column
df_encoded = pd.get_dummies(
    df, columns=['Algorithm'], prefix='Algorithm', dtype=int)

# Update df to use encoded version
df = df_encoded

# Scaling features
numerical_cols = ['Batch_Size', 'Input_Size', 'In_Channels',
                  'Out_Channels', 'Kernel_Size', 'Stride', 'Padding']
# Apply Standard Scaling to numerical columns only
scaler = StandardScaler()
df_scaled = df.copy()
df_scaled[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# Define feature columns
feature_cols = [col for col in df_scaled.columns if col != 'Execution_Time_ms']
target_col = 'Execution_Time_ms'

# Create features and target arrays
X = df_scaled[feature_cols].values
y = df_scaled[target_col].values

# Preparing the data


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
X = df_scaled.drop('Execution_Time_ms', axis=1)  # Features
y = df_scaled['Execution_Time_ms']  # Target

k = 5  # Number of folds
kf = KFold(n_splits=k, shuffle=True, random_state=42)

print(f"Training with {k}-Fold Cross Validation")
print(f"Total samples: {len(X)}")
print(f"Features: {X.shape[1]}")

# Store results for each fold
fold_results = []

# Train model for each fold
for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
    print(f"\n=== FOLD {fold} ===")

    # Get train and validation data for this fold
    X_train_fold = X.iloc[train_idx]
    X_val_fold = X.iloc[val_idx]
    y_train_fold = y.iloc[train_idx]
    y_val_fold = y.iloc[val_idx]

    # Create datasets for this fold
    train_dataset = CNNExecutionDataset(X_train_fold, y_train_fold)
    val_dataset = CNNExecutionDataset(X_val_fold, y_val_fold)

    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")

    # Store for training (you'll train one model per fold)
    fold_results.append({
        'fold': fold,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'train_size': len(train_dataset),
        'val_size': len(val_dataset)
    })

# Create a model class that inherits nn


class CNNExecutionPredictor(nn.Module):
    def __init__(self, input_size, dropout_rate=0.3):
        super(CNNExecutionPredictor, self).__init__()

        # Architecture: input -> 128 -> 64 -> 32 -> 1
        self.layer1 = nn.Linear(input_size, 128)
        self.layer2 = nn.Linear(128, 64)
        self.layer3 = nn.Linear(64, 32)
        self.output = nn.Linear(32, 1)

        # Activation and regularization
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, x):
        # Layer 1
        x = self.layer1(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 2
        x = self.layer2(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Layer 3
        x = self.layer3(x)
        x = self.relu(x)
        x = self.dropout(x)

        # Output (no activation for regression)
        x = self.output(x)
        return x


# Initialize model
input_size = X.shape[1]  # Number of features
model = CNNExecutionPredictor(input_size)

# Training setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=10, factor=0.5)

# Training function


def train_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    total_loss = 0

    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        predictions = model(X_batch).squeeze()
        loss = criterion(predictions, y_batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)

# Validation function


def validate(model, test_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions = []
    actuals = []

    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            pred = model(X_batch).squeeze()
            loss = criterion(pred, y_batch)
            total_loss += loss.item()

            predictions.extend(pred.cpu().numpy())
            actuals.extend(y_batch.cpu().numpy())

    return total_loss / len(test_loader), predictions, actuals

# Evaluation


def calculate_mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def validate_model(model, val_loader, criterion, device):
    model.eval()
    total_loss = 0
    predictions, targets = [], []

    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
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


def print_metrics(metrics, title="Results"):
    print(f"\n{title}:")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")


# The training loop
# Training parameters
epochs = 100
patience = 15  # Early stopping patience
input_size = X.shape[1]

# Store results from all folds
fold_metrics = []
trained_models = []

print(f"Starting {len(fold_results)}-Fold Cross Validation Training")
print(f"Epochs: {epochs}, Early Stopping Patience: {patience}")
print("=" * 60)

for fold_data in fold_results:
    fold = fold_data['fold']
    train_loader = fold_data['train_loader']
    val_loader = fold_data['val_loader']

    print(f"\n🔄 FOLD {fold}")
    print("-" * 20)

    # Create fresh model for this fold
    model = CNNExecutionPredictor(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=10, factor=0.5)

    # Early stopping variables
    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    # Training loop for this fold
    for epoch in range(epochs):
        # Training phase
        model.train()
        train_loss = 0

        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)

            optimizer.zero_grad()
            predictions = model(X_batch).squeeze()
            loss = criterion(predictions, y_batch)
            loss.backward()
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

        # Print progress every 20 epochs
        if (epoch + 1) % 20 == 0:
            print(f"  Epoch {epoch+1:3d}/{epochs} | Train Loss: {avg_train_loss:.4f} | "
                  f"Val Loss: {val_loss:.4f} | Val MAPE: {val_metrics['mape']:.2f}%")

        # Early stopping
        if patience_counter >= patience:
            print(f"  Early stopping at epoch {epoch+1}")
            break

    # Load best model state
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    # Final evaluation on this fold
    train_metrics = validate_model(model, train_loader, criterion, device)
    val_metrics = validate_model(model, val_loader, criterion, device)

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
        'best_val_loss': best_val_loss
    }

    fold_metrics.append(fold_result)
    trained_models.append(model)

    # Print fold results
    print(f"  ✅ Fold {fold} Complete")
    print(
        f"     Train MAPE: {train_metrics['mape']:.2f}% | Val MAPE: {val_metrics['mape']:.2f}%")
    print(
        f"     Train MAE:  {train_metrics['mae']:.4f} | Val MAE:  {val_metrics['mae']:.4f}")
    print(
        f"     Train R²:   {train_metrics['r2']:.4f} | Val R²:   {val_metrics['r2']:.4f}")

# Calculate average performance across all folds
print("\n" + "=" * 60)
print("📊 CROSS VALIDATION SUMMARY")
print("=" * 60)

avg_train_mape = np.mean([f['train_mape'] for f in fold_metrics])
avg_val_mape = np.mean([f['val_mape'] for f in fold_metrics])
std_val_mape = np.std([f['val_mape'] for f in fold_metrics])

avg_train_mae = np.mean([f['train_mae'] for f in fold_metrics])
avg_val_mae = np.mean([f['val_mae'] for f in fold_metrics])
std_val_mae = np.std([f['val_mae'] for f in fold_metrics])

avg_train_r2 = np.mean([f['train_r2'] for f in fold_metrics])
avg_val_r2 = np.mean([f['val_r2'] for f in fold_metrics])
std_val_r2 = np.std([f['val_r2'] for f in fold_metrics])

print(f"Average Train MAPE: {avg_train_mape:.2f}%")
print(f"Average Val MAPE:   {avg_val_mape:.2f}% ± {std_val_mape:.2f}%")
print(f"Average Train MAE:  {avg_train_mae:.4f}")
print(f"Average Val MAE:    {avg_val_mae:.4f} ± {std_val_mae:.4f}")
print(f"Average Train R²:   {avg_train_r2:.4f}")
print(f"Average Val R²:     {avg_val_r2:.4f} ± {std_val_r2:.4f}")

# Detailed results table
print(f"\nDetailed Results by Fold:")
print(f"{'Fold':<4} {'Train MAPE':<11} {'Val MAPE':<9} {'Train R²':<8} {'Val R²':<7} {'Epochs':<7}")
print("-" * 60)
for f in fold_metrics:
    print(f"{f['fold']:<4} {f['train_mape']:<11.2f} {f['val_mape']:<9.2f} "
          f"{f['train_r2']:<8.4f} {f['val_r2']:<7.4f} {f['epochs_trained']:<7}")

print(f"\n🎉 Training Complete!")
print(f"📈 Best performing fold: Fold {min(fold_metrics, key=lambda x: x['val_mape'])['fold']} "
      f"(Val MAPE: {min(fold_metrics, key=lambda x: x['val_mape'])['val_mape']:.2f}%)")
