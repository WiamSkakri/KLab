#!/bin/bash
#SBATCH -J poly_ridge_training          # Job name
#SBATCH -o poly_ridge_training.out      # Output file
#SBATCH --time=24:00:00                  # 24 hours (Ridge has hyperparameter search)
#SBATCH -N 1                            # 1 Node
#SBATCH -A sxk1942                      # Account/Project ID
#SBATCH -c 6                            # 6 processors (more for grid search)
#SBATCH --mem=12GB                      # 12GB memory

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

# Print CPU information
echo "Checking CPU configuration..."
echo "Number of CPU cores: $(nproc)"
echo "CPU info:"
lscpu | head -10
echo "Memory info:"
free -h
echo "-----------------------------------"

# Verify Python and required packages
python << 'END_PYTHON'
import sys
import pandas as pd
import sklearn
import numpy as np
import matplotlib
import joblib
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

print(f"Python path: {sys.executable}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Joblib version: {joblib.__version__}")
print("All required packages successfully imported!")
END_PYTHON

if [ $? -ne 0 ]; then
    echo "Error: Failed to import required packages"
    exit 1
fi

# Create timestamp for unique results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=$SLURM_SUBMIT_DIR/results_ridge_${TIMESTAMP}
mkdir -p $RESULTS_DIR

# Create a directory in scratch for the job
SCRATCH_DIR=$PFSDIR/poly_ridge_training_${SLURM_JOB_ID}
if ! mkdir -p $SCRATCH_DIR; then
    echo "Failed to create scratch directory: $SCRATCH_DIR"
    exit 1
fi
echo "Created scratch directory: $SCRATCH_DIR"

# Check if required files exist
if [ ! -f polynomial_ridge.py ]; then
    echo "Error: polynomial_ridge.py not found in current directory"
    exit 1
fi

if [ ! -f combined.csv ]; then
    echo "Error: combined.csv not found in current directory"
    echo "Please ensure the combined.csv file is in the same directory as this job script"
    exit 1
fi

# Copy files to scratch
cp polynomial_ridge.py $SCRATCH_DIR/
cp combined.csv $SCRATCH_DIR/

# Change to scratch directory for processing
cd $SCRATCH_DIR

echo "Starting Ridge Polynomial Regression Training..."
echo "Working directory: $(pwd)"
echo "Time started: $(date)"
echo "Model: Ridge Regression (degree 2, L2 regularization)"
echo "Hyperparameters: alpha=[0.01, 0.1, 1.0, 10.0, 100.0]"
echo "Expected runtime: 10-60 minutes"
echo "=================================================================================="

# Run the Ridge polynomial regression training with output logging
python polynomial_ridge.py 2>&1 | tee training_output.log

# Check if training completed successfully
if [ $? -ne 0 ]; then
    echo "Error: Training failed"
    exit 1
fi

echo "=================================================================================="
echo "Training completed successfully!"
echo "Time finished: $(date)"

# Copy all results back to submit directory
cp -r * $RESULTS_DIR/

# Count generated files
GENERATED_FILES=$(find $RESULTS_DIR -type f | wc -l)

echo ""
echo "🎉 Ridge Polynomial Regression Training Complete!"
echo "📊 Model: Ridge Regression (L2 regularization)"
echo "📐 Polynomial Degree: 2"
echo "🎯 Regularization: L2 penalty (shrinks coefficients)"
echo "⏱️  Total execution time: $SECONDS seconds"
echo "📁 Results Directory: $RESULTS_DIR"
echo "📄 Generated $GENERATED_FILES files"
echo ""

# Cleanup scratch directory
if [ -d "$SCRATCH_DIR" ]; then
    rm -rf $SCRATCH_DIR
    echo "Cleaned up scratch directory"
fi

# Deactivate virtual environment
deactivate

echo "Job completed successfully. Results are in: $RESULTS_DIR"
echo "Job finished at: $(date)"
echo ""
echo "📋 To view results:"
echo "  Training log:     cat $RESULTS_DIR/training_output.log | less"
echo "  Results CSV:      cat $RESULTS_DIR/ridge_training_results.csv"
echo "  Visualization:    ls $RESULTS_DIR/ridge_polynomial_results.png"
echo "  Job output:       cat poly_ridge_training.out"
echo ""
echo "🔄 To load the best model in Python:"
echo "  import joblib"
echo "  model = joblib.load('$RESULTS_DIR/best_ridge_model.joblib')"
echo ""
echo "🎯 Ridge Regression Characteristics:"
echo "  ✅ L2 regularization prevents overfitting"
echo "  ✅ Handles multicollinearity well"
echo "  ✅ Good for correlated features"
echo "  ✅ Shrinks coefficients toward zero"
echo ""
echo "🚀 Next steps:"
echo "  1. Run: sbatch job_lasso.sh"
echo "  2. Run: sbatch job_elasticnet.sh"
echo "  3. Compare: sbatch job_compare.sh" 