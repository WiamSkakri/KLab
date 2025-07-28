#!/bin/bash
#SBATCH -J poly_compare_models          # Job name
#SBATCH -o poly_compare_models.out      # Output file
#SBATCH --time=24:00:00                  # 24 hours (comparison is fast)
#SBATCH -N 1                            # 1 Node
#SBATCH -A sxk1942                      # Account/Project ID
#SBATCH -c 2                            # 2 processors (minimal for comparison)
#SBATCH --mem=4GB                       # 4GB memory (minimal for comparison)

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
import numpy as np
import matplotlib
import os

print(f"Python path: {sys.executable}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print("All required packages successfully imported!")
END_PYTHON

if [ $? -ne 0 ]; then
    echo "Error: Failed to import required packages"
    exit 1
fi

# Create timestamp for unique results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=$SLURM_SUBMIT_DIR/results_comparison_${TIMESTAMP}
mkdir -p $RESULTS_DIR

# Create a directory in scratch for the job
SCRATCH_DIR=$PFSDIR/poly_compare_${SLURM_JOB_ID}
if ! mkdir -p $SCRATCH_DIR; then
    echo "Failed to create scratch directory: $SCRATCH_DIR"
    exit 1
fi
echo "Created scratch directory: $SCRATCH_DIR"

# Check if required files exist
if [ ! -f compare_models.py ]; then
    echo "Error: compare_models.py not found in current directory"
    exit 1
fi

# Check for result files from individual models
echo "Checking for individual model results..."
FOUND_RESULTS=0

if [ -f linear_training_results.csv ]; then
    echo "✅ Found Linear results: linear_training_results.csv"
    FOUND_RESULTS=$((FOUND_RESULTS + 1))
else
    echo "❌ Missing Linear results: linear_training_results.csv"
fi

if [ -f ridge_training_results.csv ]; then
    echo "✅ Found Ridge results: ridge_training_results.csv"
    FOUND_RESULTS=$((FOUND_RESULTS + 1))
else
    echo "❌ Missing Ridge results: ridge_training_results.csv"
fi

if [ -f lasso_training_results.csv ]; then
    echo "✅ Found Lasso results: lasso_training_results.csv"
    FOUND_RESULTS=$((FOUND_RESULTS + 1))
else
    echo "❌ Missing Lasso results: lasso_training_results.csv"
fi

if [ -f elasticnet_training_results.csv ]; then
    echo "✅ Found ElasticNet results: elasticnet_training_results.csv"
    FOUND_RESULTS=$((FOUND_RESULTS + 1))
else
    echo "❌ Missing ElasticNet results: elasticnet_training_results.csv"
fi

if [ $FOUND_RESULTS -eq 0 ]; then
    echo ""
    echo "❌ ERROR: No individual model results found!"
    echo "Please run the individual model training jobs first:"
    echo "  sbatch job_linear.sh"
    echo "  sbatch job_ridge.sh"
    echo "  sbatch job_lasso.sh"
    echo "  sbatch job_elasticnet.sh"
    exit 1
fi

echo ""
echo "Found $FOUND_RESULTS model result files. Proceeding with comparison..."

# Copy files to scratch
cp compare_models.py $SCRATCH_DIR/

# Copy any available result files
for file in *_training_results.csv; do
    if [ -f "$file" ]; then
        cp "$file" $SCRATCH_DIR/
    fi
done

# Change to scratch directory for processing
cd $SCRATCH_DIR

echo "Starting Polynomial Regression Model Comparison..."
echo "Working directory: $(pwd)"
echo "Time started: $(date)"
echo "Comparing $FOUND_RESULTS polynomial regression models"
echo "Expected runtime: 1-5 minutes"
echo "=================================================================================="

# Run the model comparison with output logging
python compare_models.py 2>&1 | tee comparison_output.log

# Check if comparison completed successfully
if [ $? -ne 0 ]; then
    echo "Error: Model comparison failed"
    exit 1
fi

echo "=================================================================================="
echo "Comparison completed successfully!"
echo "Time finished: $(date)"

# Copy all results back to submit directory
cp -r * $RESULTS_DIR/

# Count generated files
GENERATED_FILES=$(find $RESULTS_DIR -type f | wc -l)

echo ""
echo "🎉 Polynomial Regression Model Comparison Complete!"
echo "📊 Models Compared: $FOUND_RESULTS"
echo "📈 Comprehensive analysis and visualization generated"
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
echo "  Comparison log:   cat $RESULTS_DIR/comparison_output.log | less"
echo "  Comparison CSV:   cat $RESULTS_DIR/polynomial_models_comparison.csv"
echo "  Visualization:    ls $RESULTS_DIR/polynomial_models_comparison.png"
echo "  Job output:       cat poly_compare_models.out"
echo ""
echo "🏆 Key Outputs:"
echo "  • Performance ranking of all models"
echo "  • Training time comparisons"
echo "  • Detailed model characteristics"
echo "  • Recommendations for best model"
echo ""
echo "🎯 Model Comparison Benefits:"
echo "  ✅ Side-by-side performance metrics"
echo "  ✅ Training efficiency analysis"
echo "  ✅ Model selection recommendations"
echo "  ✅ Comprehensive visualization dashboard"
echo ""
echo "📊 The comparison results will help you choose the best polynomial regression model"
echo "    for your CNN execution time prediction task!" 