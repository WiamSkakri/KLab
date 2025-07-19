#!/bin/bash
#SBATCH -J poly_prediction_training     # Job name
#SBATCH -o poly_prediction_training.out # Output file
#SBATCH --time=2:00:00                  # 2 hours of wall time (polynomial is faster than NN)
#SBATCH -N 1                            # 1 Node
#SBATCH -A sxk1942                      # Account/Project ID
#SBATCH -c 8                            # 8 processors (more CPUs for parallel grid search)
#SBATCH --mem=16GB                      # 16GB memory (less than NN since no GPU tensors)

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
from sklearn.linear_model import Ridge, Lasso, ElasticNet, LinearRegression
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
RESULTS_DIR=$SLURM_SUBMIT_DIR/results_${TIMESTAMP}
mkdir -p $RESULTS_DIR

# Create a directory in scratch for the job
SCRATCH_DIR=$PFSDIR/poly_prediction_training_${SLURM_JOB_ID}
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

if [ ! -f combined.csv ]; then
    echo "Error: combined.csv not found in current directory"
    echo "Please ensure the combined.csv file is in the same directory as this job script"
    exit 1
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
echo "CSV file columns: $(head -1 combined.csv | tr ',' '\n' | wc -l)"

# Set environment variables for optimal CPU performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

echo "Set CPU threads to: $SLURM_CPUS_PER_TASK"

# Run the training script and capture all output
echo "Starting Polynomial Regression Training..."
echo "=========================================="
echo "Testing 4 polynomial regression models:"
echo "  - Ridge Regression (L2 regularization)"
echo "  - Lasso Regression (L1 regularization)" 
echo "  - ElasticNet (L1 + L2 regularization)"
echo "  - Linear Regression (no regularization)"
echo "=========================================="
python python.py 2>&1 | tee training_output.log

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Polynomial Regression training completed successfully"
else
    echo "Polynomial Regression training failed with exit code $?"
    # Copy logs even if script failed
    cp training_output.log $RESULTS_DIR/
    exit 1
fi

# Copy all results to the timestamped results directory
echo "Copying results to: $RESULTS_DIR"
cp -v training_output.log $RESULTS_DIR/
cp -v polynomial_training_results.csv $RESULTS_DIR/ 2>/dev/null || echo "polynomial_training_results.csv not found"
cp -v polynomial_summary_metrics.csv $RESULTS_DIR/ 2>/dev/null || echo "polynomial_summary_metrics.csv not found"
cp -v best_polynomial_model.joblib $RESULTS_DIR/ 2>/dev/null || echo "best_polynomial_model.joblib not found"
cp -v poly_*_model_fold_*.joblib $RESULTS_DIR/ 2>/dev/null || echo "No individual fold models found"
cp -v *.png $RESULTS_DIR/ 2>/dev/null || echo "No PNG files found"

# Count generated files
GENERATED_FILES=$(ls -1 $RESULTS_DIR/ | wc -l)
echo "Copied $GENERATED_FILES files to results directory"

# Extract key results from training output
echo "Extracting key results..."
BEST_MODEL=""
BEST_MAPE=""
BEST_DEGREE=""
TRAINING_TIME=""

if [ -f "training_output.log" ]; then
    BEST_MODEL=$(grep "🏆 Best Overall Model:" training_output.log | sed 's/.*Best Overall Model: //' | head -1)
    BEST_MAPE=$(grep "🎯 Best Fold:" training_output.log | sed 's/.*Val MAPE: //' | sed 's/%).*//' | head -1)
    BEST_DEGREE=$(grep "📐 Best Polynomial Degree:" training_output.log | sed 's/.*Best Polynomial Degree: //' | head -1)
    TRAINING_TIME=$(grep "⏱️  Total Training Time:" training_output.log | sed 's/.*Total Training Time: //' | sed 's/ seconds.*//' | head -1)
fi

# Create a comprehensive summary report
echo "Creating summary report..."
cat > $RESULTS_DIR/job_summary.txt << EOF
Polynomial Regression Training Job Summary
==========================================
Job ID: $SLURM_JOB_ID
Start Time: $(date)
Submit Directory: $SLURM_SUBMIT_DIR
Results Directory: $RESULTS_DIR

Job Configuration:
- Partition: compute (CPU)
- CPU Cores: $SLURM_CPUS_PER_TASK
- Memory: 16GB
- Wall Time: 2 hours

Training Results:
- Best Model: $BEST_MODEL
- Best MAPE: $BEST_MAPE%
- Best Polynomial Degree: $BEST_DEGREE
- Total Training Time: $TRAINING_TIME seconds

Models Tested:
1. Ridge Regression (L2 regularization)
2. Lasso Regression (L1 regularization) 
3. ElasticNet (L1 + L2 regularization)
4. Linear Regression (no regularization)

Cross-Validation: 5-fold CV for each model type

Files Generated:
- training_output.log: Complete training log with timestamps
- polynomial_training_results.csv: Detailed results for all models and folds
- polynomial_summary_metrics.csv: Summary statistics by model type
- best_polynomial_model.joblib: Best performing trained model
- poly_*_model_fold_*.joblib: Individual models for each fold and type
- polynomial_evaluation_dashboard.png: Main evaluation dashboard (9 plots)
- polynomial_detailed_analysis.png: Detailed analysis (6 plots)
- polynomial_training_results.png: Basic visualization (fallback)
- job_summary.txt: This summary

System Information:
- CPU: $(lscpu | grep "Model name" | sed 's/Model name: *//')
- Cores: $(nproc)
- Memory: $(free -h | grep Mem | awk '{print $2}')
- Python: $(python --version)
- Scikit-learn: $(python -c "import sklearn; print(sklearn.__version__)")

Training completed successfully!
Polynomial regression provides fast, interpretable models for CNN execution time prediction.
EOF

# Print final system resource usage
echo "Final system resource usage:"
echo "Memory usage:"
free -h
echo "CPU load average:"
uptime

# Print a quick summary of results
echo ""
echo "🎉 TRAINING COMPLETED SUCCESSFULLY! 🎉"
echo "======================================="
if [ ! -z "$BEST_MODEL" ]; then
    echo "🏆 Best Model: $BEST_MODEL"
fi
if [ ! -z "$BEST_MAPE" ]; then
    echo "📊 Best MAPE: $BEST_MAPE%"
fi
if [ ! -z "$BEST_DEGREE" ]; then
    echo "📐 Best Polynomial Degree: $BEST_DEGREE"
fi
if [ ! -z "$TRAINING_TIME" ]; then
    echo "⏱️  Training Time: $TRAINING_TIME seconds"
fi
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
echo "  Results CSV:      cat $RESULTS_DIR/polynomial_training_results.csv"
echo "  Summary metrics:  cat $RESULTS_DIR/polynomial_summary_metrics.csv"
echo "  Job summary:      cat $RESULTS_DIR/job_summary.txt"
echo "  Visualizations:   ls $RESULTS_DIR/*.png"
echo ""
echo "🔄 To load the best model in Python:"
echo "  import joblib"
echo "  model = joblib.load('$RESULTS_DIR/best_polynomial_model.joblib')"
echo ""
echo "🎯 Key Advantages of Polynomial Regression:"
echo "  ✅ Fast training (CPU-only, no GPU required)"
echo "  ✅ Interpretable results"
echo "  ✅ Good for nonlinear patterns"
echo "  ✅ Multiple regularization options tested" 