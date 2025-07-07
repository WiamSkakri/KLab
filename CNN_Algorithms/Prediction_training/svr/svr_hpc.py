from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
import torch.optim as optim
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch
import joblib
import pandas as pd
import numpy as np
import time
import matplotlib.pyplot as plt
import sys
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Try to import GPU-accelerated libraries first, fall back to CPU versions
try:
    from cuml.svm import SVR
    from cuml.model_selection import GridSearchCV
    from cuml.preprocessing import StandardScaler
    from cuml.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import KFold  # KFold not available in cuML
    GPU_AVAILABLE = True
    print("🚀 Using GPU-accelerated cuML libraries")
except ImportError:
    from sklearn.svm import SVR
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    from sklearn.model_selection import KFold, GridSearchCV
    from sklearn.preprocessing import StandardScaler
    GPU_AVAILABLE = False
    print("⚠️  cuML not available, using CPU-based sklearn libraries")

# Additional sklearn imports for fallback


def setup_logging():
    """Setup logging configuration"""
    print("=" * 80)
    print("SVR HYPERPARAMETER OPTIMIZATION AND CROSS-VALIDATION")
    print("=" * 80)
    print(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Python Version: {sys.version}")
    print(f"Working Directory: {os.getcwd()}")
    print(f"Script: {os.path.basename(__file__)}")

    # GPU information
    print(f"PyTorch CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU Device: {torch.cuda.get_device_name(0)}")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    print(f"Using GPU-accelerated libraries: {GPU_AVAILABLE}")
    print()


def load_and_preprocess_data(csv_file='combined.csv'):
    """Load and preprocess the dataset"""
    print("📊 LOADING AND PREPROCESSING DATA")
    print("-" * 50)

    try:
        # Load data
        print(f"Loading data from: {csv_file}")
        df = pd.read_csv(csv_file)
        print(
            f"✅ Data loaded successfully: {df.shape[0]} rows, {df.shape[1]} columns")

        # Display basic info
        print(f"Original columns: {list(df.columns)}")
        print(
            f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

        # One-hot encode the Algorithm column
        print("Encoding categorical variables...")
        df_encoded = pd.get_dummies(
            df, columns=['Algorithm'], prefix='Algorithm', dtype=int)
        print(
            f"After encoding: {df_encoded.shape[0]} rows, {df_encoded.shape[1]} columns")

        # Update df to use encoded version
        df = df_encoded

        # Scaling features
        numerical_cols = ['Batch_Size', 'Input_Size', 'In_Channels',
                          'Out_Channels', 'Kernel_Size', 'Stride', 'Padding']
        print(f"Scaling numerical columns: {numerical_cols}")

        # Apply Standard Scaling to numerical columns only
        scaler = StandardScaler()
        df_scaled = df.copy()

        if GPU_AVAILABLE and torch.cuda.is_available():
            # Use GPU for scaling if available
            print("Using GPU for data scaling...")
            numerical_data = torch.tensor(
                df[numerical_cols].values, dtype=torch.float32).cuda()
            numerical_mean = numerical_data.mean(dim=0)
            numerical_std = numerical_data.std(dim=0)
            numerical_scaled = (
                numerical_data - numerical_mean) / numerical_std
            df_scaled[numerical_cols] = numerical_scaled.cpu().numpy()
        else:
            # Use standard CPU scaling
            df_scaled[numerical_cols] = scaler.fit_transform(
                df[numerical_cols])

        # Define feature columns
        feature_cols = [
            col for col in df_scaled.columns if col != 'Execution_Time_ms']
        target_col = 'Execution_Time_ms'

        print(f"Feature columns ({len(feature_cols)}): {feature_cols}")
        print(f"Target column: {target_col}")

        # Create features and target arrays
        X = df_scaled[feature_cols].values
        y = df_scaled[target_col].values

        print(f"✅ Feature matrix shape: {X.shape}")
        print(f"✅ Target vector shape: {y.shape}")
        print(
            f"Target statistics: min={y.min():.4f}, max={y.max():.4f}, mean={y.mean():.4f}, std={y.std():.4f}")

        return X, y, scaler, feature_cols, target_col

    except Exception as e:
        print(f"❌ Error loading data: {str(e)}")
        sys.exit(1)


def setup_kfold_splits(X, y, k=5):
    """Setup K-Fold cross validation splits"""
    print(f"\n🔄 SETTING UP {k}-FOLD CROSS VALIDATION")
    print("-" * 50)

    # K-Fold Cross Validation Setup
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    # Store results for each fold
    fold_data = []

    # Prepare data splits for each fold
    for fold, (train_idx, val_idx) in enumerate(kf.split(X), 1):
        print(
            f"Fold {fold}: Training={len(train_idx)}, Validation={len(val_idx)}")

        # Simple array slicing
        X_train = X[train_idx]
        X_val = X[val_idx]
        y_train = y[train_idx]
        y_val = y[val_idx]

        # Store for SVR training
        fold_data.append({
            'fold': fold,
            'X_train': X_train,
            'X_val': X_val,
            'y_train': y_train,
            'y_val': y_val,
            'train_size': len(X_train),
            'val_size': len(X_val)
        })

    print(f"✅ {k}-Fold splits prepared successfully")
    return fold_data


def setup_hyperparameter_grid():
    """Setup hyperparameter grids for SVR"""
    print(f"\n⚙️ SETTING UP HYPERPARAMETER GRIDS")
    print("-" * 50)

    # Define hyperparameter grid for SVR
    param_grid = {
        'kernel': ['rbf', 'linear'],  # Start with RBF and Linear
        'C': [0.1, 1, 10, 100],       # Regularization strength
        # Kernel coefficient (RBF only)
        'gamma': ['scale', 'auto', 0.001, 0.01, 0.1],
        'epsilon': [0.01, 0.1, 0.2]   # Tolerance
    }

    # For faster search with large dataset, use smaller grid
    param_grid_small = {
        'kernel': ['rbf'],
        'C': [1, 10, 100],
        'gamma': ['scale', 0.01, 0.1],
        'epsilon': [0.1]
    }

    full_combinations = len(param_grid['kernel']) * len(param_grid['C']) * len(
        param_grid['gamma']) * len(param_grid['epsilon'])
    small_combinations = len(param_grid_small['kernel']) * len(param_grid_small['C']) * len(
        param_grid_small['gamma']) * len(param_grid_small['epsilon'])

    print(f"Full grid combinations: {full_combinations}")
    print(f"Small grid combinations: {small_combinations}")
    print("Using small grid for faster computation")

    return param_grid_small


def calculate_mape(y_true, y_pred):
    """Calculate Mean Absolute Percentage Error"""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true != 0
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def create_svr_with_gridsearch(param_grid, cv_folds=3):
    """Create SVR with GridSearchCV"""
    # Create base SVR model
    if GPU_AVAILABLE:
        # cuML SVR doesn't need n_jobs parameter
        svr = SVR()
        grid_search = GridSearchCV(
            estimator=svr,
            param_grid=param_grid,
            cv=cv_folds,           # Internal CV for hyperparameter tuning
            scoring='neg_mean_squared_error',  # Use MSE for tuning
            verbose=1              # Show progress
        )
    else:
        # sklearn SVR with n_jobs for CPU parallelization
        svr = SVR()
        grid_search = GridSearchCV(
            estimator=svr,
            param_grid=param_grid,
            cv=cv_folds,           # Internal CV for hyperparameter tuning
            scoring='neg_mean_squared_error',  # Use MSE for tuning
            n_jobs=-1,             # Use all available cores
            verbose=1              # Show progress
        )

    return grid_search


def evaluate_svr(model, X, y):
    """Evaluate SVR model"""
    # Make predictions
    predictions = model.predict(X)

    # Calculate metrics
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
    """Print evaluation metrics"""
    print(f"{title}:")
    print(f"  MAPE: {metrics['mape']:.2f}%")
    print(f"  MAE:  {metrics['mae']:.4f}")
    print(f"  RMSE: {metrics['rmse']:.4f}")
    print(f"  R²:   {metrics['r2']:.4f}")


def train_svr_cross_validation(fold_data, param_grid):
    """Train SVR with K-Fold Cross Validation"""
    print(f"\n🚀 STARTING SVR CROSS VALIDATION TRAINING")
    print("=" * 80)

    if GPU_AVAILABLE and torch.cuda.is_available():
        print(f"🎯 Using GPU acceleration with cuML")
        print(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
    else:
        print(f"🔧 Using CPU-based training with sklearn")

    # Store results from all folds
    fold_metrics = []
    trained_models = []

    # Start timing
    start_time = time.time()

    # Loop through each fold
    for fold_info in fold_data:
        fold = fold_info['fold']
        X_train = fold_info['X_train']
        X_val = fold_info['X_val']
        y_train = fold_info['y_train']
        y_val = fold_info['y_val']

        print(f"\n🔄 FOLD {fold}")
        print("-" * 30)
        print(f"Training samples: {len(X_train)}")
        print(f"Validation samples: {len(X_val)}")

        # Create SVR with GridSearchCV
        print("Running hyperparameter search...")
        svr_grid = create_svr_with_gridsearch(param_grid, cv_folds=3)

        # Fit the model (this does hyperparameter search + training)
        fold_start_time = time.time()
        svr_grid.fit(X_train, y_train)
        fold_training_time = time.time() - fold_start_time

        # Get the best model
        best_model = svr_grid.best_estimator_
        best_params = svr_grid.best_params_
        best_score = svr_grid.best_score_

        print(f"✅ Training complete in {fold_training_time:.1f} seconds")
        print(f"Best parameters: {best_params}")
        print(f"Best CV score (neg MSE): {best_score:.4f}")

        # Evaluate on training and validation sets
        train_metrics = evaluate_svr(best_model, X_train, y_train)
        val_metrics = evaluate_svr(best_model, X_val, y_val)

        # Store results for this fold
        fold_result = {
            'fold': fold,
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
            'training_time': fold_training_time
        }

        fold_metrics.append(fold_result)
        trained_models.append(best_model)

        # Print fold results
        print(f"📊 Fold {fold} Results:")
        print_metrics(train_metrics, f"  Training")
        print_metrics(val_metrics, f"  Validation")

        # Clear GPU cache if using CUDA
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Calculate total training time
    total_time = time.time() - start_time

    return fold_metrics, trained_models, total_time


def analyze_and_save_results(fold_metrics, trained_models, total_time):
    """Analyze results and save to files"""
    print(f"\n📊 ANALYZING AND SAVING RESULTS")
    print("=" * 80)

    # Calculate average performance across all folds
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

    # Print summary
    print(f"CROSS VALIDATION SUMMARY:")
    print(f"Average Train MAPE: {avg_train_mape:.2f}%")
    print(f"Average Val MAPE:   {avg_val_mape:.2f}% ± {std_val_mape:.2f}%")
    print(f"Average Train MAE:  {avg_train_mae:.4f}")
    print(f"Average Val MAE:    {avg_val_mae:.4f} ± {std_val_mae:.4f}")
    print(f"Average Train R²:   {avg_train_r2:.4f}")
    print(f"Average Val R²:     {avg_val_r2:.4f} ± {std_val_r2:.4f}")
    print(f"Average Training Time: {avg_training_time:.1f} seconds per fold")
    print(f"Total Training Time: {total_time:.1f} seconds")

    # Detailed results table
    print(f"\nDetailed Results by Fold:")
    print(f"{'Fold':<4} {'Train MAPE':<11} {'Val MAPE':<9} {'Train R²':<8} {'Val R²':<7} {'Time(s)':<8}")
    print("-" * 60)
    for f in fold_metrics:
        print(f"{f['fold']:<4} {f['train_mape']:<11.2f} {f['val_mape']:<9.2f} "
              f"{f['train_r2']:<8.4f} {f['val_r2']:<7.4f} {f['training_time']:<8.1f}")

    # Best performing fold
    best_fold = min(fold_metrics, key=lambda x: x['val_mape'])
    print(
        f"\n🏆 Best performing fold: Fold {best_fold['fold']} (Val MAPE: {best_fold['val_mape']:.2f}%)")

    # Most common hyperparameters
    print(f"\nMost common best parameters:")
    all_params = [f['best_params'] for f in fold_metrics]
    for param in ['kernel', 'C', 'gamma', 'epsilon']:
        values = [p.get(param) for p in all_params if param in p]
        if values:
            most_common = max(set(values), key=values.count)
            print(
                f"  {param}: {most_common} (appeared in {values.count(most_common)}/{len(values)} folds)")

    # Save results to CSV
    results_df = pd.DataFrame(fold_metrics)
    results_csv = 'svr_training_results.csv'
    results_df.to_csv(results_csv, index=False)
    print(f"\n💾 Results saved to: {results_csv}")

    # Save best model
    best_model_idx = np.argmin([f['val_mape'] for f in fold_metrics])
    best_model = trained_models[best_model_idx]
    best_model_file = 'best_svr_model.pkl'
    joblib.dump(best_model, best_model_file)
    print(f"💾 Best model saved to: {best_model_file}")

    # Create summary dictionary
    summary = {
        'avg_train_mape': avg_train_mape,
        'avg_val_mape': avg_val_mape,
        'std_val_mape': std_val_mape,
        'avg_train_mae': avg_train_mae,
        'avg_val_mae': avg_val_mae,
        'std_val_mae': std_val_mae,
        'avg_train_r2': avg_train_r2,
        'avg_val_r2': avg_val_r2,
        'std_val_r2': std_val_r2,
        'avg_training_time': avg_training_time,
        'total_time': total_time,
        'best_fold': best_fold['fold'],
        'best_val_mape': best_fold['val_mape']
    }

    return summary


def main():
    """Main function to run the SVR training pipeline"""
    # Setup logging
    setup_logging()

    try:
        # Load and preprocess data
        X, y, scaler, feature_cols, target_col = load_and_preprocess_data()

        # Setup K-Fold cross validation
        fold_data = setup_kfold_splits(X, y, k=5)

        # Setup hyperparameter grid
        param_grid = setup_hyperparameter_grid()

        # Train SVR with cross validation
        fold_metrics, trained_models, total_time = train_svr_cross_validation(
            fold_data, param_grid)

        # Analyze and save results
        summary = analyze_and_save_results(
            fold_metrics, trained_models, total_time)

        print(f"\n🎉 SVR TRAINING COMPLETED SUCCESSFULLY!")
        print(
            f"🚀 Acceleration: {'GPU (cuML)' if GPU_AVAILABLE else 'CPU (sklearn)'}")
        print(
            f"⏱️  Total time: {total_time:.1f} seconds ({total_time/60:.1f} minutes)")
        print(
            f"📊 Final validation MAPE: {summary['avg_val_mape']:.2f}% ± {summary['std_val_mape']:.2f}%")
        print(f"🏆 Best fold achieved: {summary['best_val_mape']:.2f}% MAPE")

        print(f"\n📁 Output files generated:")
        print(f"  - svr_training_results.csv: Detailed results for each fold")
        print(f"  - best_svr_model.pkl: Best trained model")

        return 0

    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
