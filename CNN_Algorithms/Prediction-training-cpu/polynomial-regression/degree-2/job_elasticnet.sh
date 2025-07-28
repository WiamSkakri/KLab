#!/bin/bash
#SBATCH -J poly_elasticnet_training     # Job name
#SBATCH -o poly_elasticnet_training.out # Output file
#SBATCH --time=24:00:00                  # 24 hours (ElasticNet has most hyperparameters)
#SBATCH -N 1                            # 1 Node
#SBATCH -A sxk1942                      # Account/Project ID
#SBATCH -c 8                            # 8 processors (most for extensive grid search)
#SBATCH --mem=16GB                      # 16GB memory (most memory needed)

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
from sklearn.linear_model import ElasticNet
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
RESULTS_DIR=$SLURM_SUBMIT_DIR/results_elasticnet_${TIMESTAMP}
mkdir -p $RESULTS_DIR

# Create a directory in scratch for the job
SCRATCH_DIR=$PFSDIR/poly_elasticnet_training_${SLURM_JOB_ID}
if ! mkdir -p $SCRATCH_DIR; then
    echo "Failed to create scratch directory: $SCRATCH_DIR"
    exit 1
fi
echo "Created scratch directory: $SCRATCH_DIR"

# Check if required files exist
if [ ! -f polynomial_elasticnet.py ]; then
    echo "Error: polynomial_elasticnet.py not found in current directory"
    exit 1
fi

if [ ! -f combined.csv ]; then
    echo "Error: combined.csv not found in current directory"
    echo "Please ensure the combined.csv file is in the same directory as this job script"
    exit 1
fi

# Copy files to scratch
cp polynomial_elasticnet.py $SCRATCH_DIR/
cp combined.csv $SCRATCH_DIR/

# Change to scratch directory for processing
cd $SCRATCH_DIR

echo "Starting ElasticNet Polynomial Regression Training..."
echo "Working directory: $(pwd)"
echo "Time started: $(date)"
echo "Model: ElasticNet Regression (degree 2, L1+L2 regularization)"
echo "Hyperparameters: alpha=[0.01, 0.1, 1.0, 10.0]"
echo "                 l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9]"
echo "Regularization: Combined L1 (feature selection) + L2 (shrinkage)"
echo "Expected runtime: 15-90 minutes"
echo "=================================================================================="

# Run the ElasticNet polynomial regression training with output logging
python polynomial_elasticnet.py 2>&1 | tee training_output.log

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
echo "🎉 ElasticNet Polynomial Regression Training Complete!"
echo "📊 Model: ElasticNet Regression (L1 + L2 regularization)"
echo "📐 Polynomial Degree: 2"
echo "🎯 Regularization: Combined L1 + L2 penalty"
echo "⚖️  L1/L2 Mix: Optimized via l1_ratio parameter"
echo "🔍 Feature Selection: Automatic + coefficient shrinkage"
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
echo "  Results CSV:      cat $RESULTS_DIR/elasticnet_training_results.csv"
echo "  Visualization:    ls $RESULTS_DIR/elasticnet_polynomial_results.png"
echo "  Job output:       cat poly_elasticnet_training.out"
echo ""
echo "🔄 To load the best model in Python:"
echo "  import joblib"
echo "  model = joblib.load('$RESULTS_DIR/best_elasticnet_model.joblib')"
echo ""
echo "🎯 ElasticNet Regression Characteristics:"
echo "  ✅ Best of both Ridge and Lasso"
echo "  ✅ Combined L1 + L2 regularization"
echo "  ✅ Feature selection + coefficient shrinkage"
echo "  ✅ Robust to multicollinearity"
echo "  ✅ Most flexible regularization approach"
echo ""
echo "🚀 Next step:"
echo "  1. Compare all models: sbatch job_compare.sh" 