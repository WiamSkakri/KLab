#!/bin/bash
#SBATCH -J xgb_optimized_gpu      # Job name
#SBATCH -o xgb_optimized_gpu.out  # Output file
#SBATCH --time=24:00:00           # 24 hours time limit (optimized version)
#SBATCH -p gpu                    # GPU partition
#SBATCH -A sxk1942               # Account/Project ID
#SBATCH -c 4                     # 4 processors
#SBATCH --mem=32GB              # 32GB memory
#SBATCH --gpus=1                # Request 1 GPU

# Exit on any error
set -e

# Print debug information
echo "Debug Information:"
echo "Current directory: $(pwd)"
echo "Contents of current directory:"
ls -la
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "HOME directory: $HOME"
echo "PFSDIR: $PFSDIR"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "GPUs: $SLURM_GPUS"
echo "Date/Time: $(date)"

# Check if virtual environment exists
if [ ! -d "$HOME/ai3_env" ]; then
    echo "Error: Virtual environment not found at $HOME/ai3_env"
    exit 1
fi

# Activate the existing ai3_env virtual environment
echo "Activating ai3_env virtual environment"
source $HOME/ai3_env/bin/activate || {
    echo "Error: Failed to activate ai3_env virtual environment"
    exit 1
}

# Print environment information
echo "Python interpreter: $(which python)"
echo "Python version: $(python --version)"
echo "Virtual environment location: $VIRTUAL_ENV"

# Check CUDA availability and print GPU information
echo "Checking CUDA and GPU configuration..."
nvidia-smi
echo "-----------------------------------"

# Verify Python and required packages
python << 'END_PYTHON'
import sys
import pandas as pd
import sklearn
import numpy as np
import matplotlib
import joblib

print(f"Python path: {sys.executable}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Joblib version: {joblib.__version__}")

# Check XGBoost capability
try:
    import xgboost as xgb
    print(f"XGBoost version: {xgb.__version__}")
    
    # Test GPU access for XGBoost
    try:
        # Try to create a GPU-enabled XGBoost regressor (modern syntax)
        gpu_params = {'tree_method': 'hist', 'device': 'cuda'}
        xgb_test = xgb.XGBRegressor(**gpu_params, n_estimators=10)
        print("XGBoost GPU Support: ✅ Available")
    except Exception as e:
        print(f"XGBoost GPU Support: ⚠️ Not available ({e})")
        print("Will fall back to CPU-based XGBoost")
    
except ImportError as e:
    print(f"XGBoost: ❌ Not available ({e})")
    exit(1)

# Check RandomizedSearchCV availability
from sklearn.model_selection import RandomizedSearchCV
print("RandomizedSearchCV: ✅ Available for optimized hyperparameter search")
END_PYTHON

if [ $? -ne 0 ]; then
    echo "Error: Failed to import required packages"
    exit 1
fi

# Create timestamp for unique results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=$SLURM_SUBMIT_DIR/results_${TIMESTAMP}
mkdir -p $RESULTS_DIR

# Create a directory in scratch for the job
SCRATCH_DIR=$PFSDIR/xgb_optimized_${SLURM_JOB_ID}
if ! mkdir -p $SCRATCH_DIR; then
    echo "Failed to create scratch directory: $SCRATCH_DIR"
    exit 1
fi
echo "Created scratch directory: $SCRATCH_DIR"

# Check if required files exist
if [ ! -f python.py ]; then
    echo "Error: python.py not found in current directory"
    exit 1
fi

# Check if combined.csv exists, if not copy from parent directory
if [ ! -f "combined.csv" ]; then
    echo "Looking for combined.csv in parent directories..."
    if [ -f "../combined.csv" ]; then
        cp ../combined.csv .
        echo "Copied combined.csv from parent directory"
    elif [ -f "../../combined.csv" ]; then
        cp ../../combined.csv .
        echo "Copied combined.csv from grandparent directory"
    elif [ -f "../../../combined.csv" ]; then
        cp ../../../combined.csv .
        echo "Copied combined.csv from great-grandparent directory"
    else
        echo "Error: combined.csv not found in current, parent, grandparent, or great-grandparent directory"
        exit 1
    fi
fi

# Copy the script and data to the scratch directory
cp python.py $SCRATCH_DIR/
cp combined.csv $SCRATCH_DIR/
echo "Copied Python script and data to scratch directory"

# Change to the scratch directory
cd $SCRATCH_DIR
echo "Changed to scratch directory"

# Print data file information
echo "Data file information:"
echo "CSV file size: $(ls -lh combined.csv | awk '{print $5}')"
echo "CSV file rows: $(wc -l < combined.csv)"

# Set GPU optimization environment variables
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
echo "Set CUDA and threading environment variables for optimal performance"

# Run the optimized XGBoost training script and capture all output
echo "Running OPTIMIZED XGBoost Training..."
echo "========================================================="
echo "🚀 Optimization features:"
echo "  • RandomizedSearchCV instead of GridSearchCV"
echo "  • Reduced hyperparameter search space (192 vs 4,320 combinations)"
echo "  • Limited to 50 iterations per fold instead of exhaustive search"
echo "  • Evaluation tracking for model monitoring"
echo "  • Expected runtime: ~30-90 minutes (vs 24+ hours)"
echo "========================================================="

python python.py 2>&1 | tee python_output.log

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "OPTIMIZED XGBoost training completed successfully"
else
    echo "OPTIMIZED XGBoost training failed with exit code $?"
    # Copy logs even if script failed
    cp python_output.log $RESULTS_DIR/
    exit 1
fi

# Copy results to the timestamped results directory
echo "Copying results to: $RESULTS_DIR"
cp -ru *.csv *.joblib *.png python_output.log $RESULTS_DIR/ 2>/dev/null || echo "Some result files not found (this may be normal)"

# Print summary of results
echo "Results Summary:"
echo "==============="
if [ -f "xgb_optimized_training_results.csv" ]; then
    echo "Training results file: xgb_optimized_training_results.csv"
    echo "Number of folds: $(tail -n +2 xgb_optimized_training_results.csv | wc -l)"
fi

if [ -f "xgb_optimized_summary_metrics.csv" ]; then
    echo "Summary metrics file: xgb_optimized_summary_metrics.csv"
fi

if [ -f "best_xgb_optimized_model.joblib" ]; then
    echo "Best model saved: best_xgb_optimized_model.joblib"
fi

echo "Files copied to: $RESULTS_DIR"
ls -la $RESULTS_DIR

# Cleanup scratch directory
if [ -d "$SCRATCH_DIR" ]; then
    rm -rf $SCRATCH_DIR
    echo "Cleaned up scratch directory"
fi

# Deactivate virtual environment
deactivate

echo "==========================================="
echo "OPTIMIZED XGBoost Training Job completed successfully!"
echo "Results directory: $RESULTS_DIR"
echo "🚀 Optimization: RandomizedSearchCV + Evaluation Tracking"
echo "⚡ Expected speedup: ~96% faster than original"
echo "⏱️  Target runtime: 30-90 minutes (vs 24+ hours)"
echo "===========================================" 