#!/bin/bash
#SBATCH -J svr_prediction_training      # Job name
#SBATCH -o svr_prediction_training.out  # Output file
#SBATCH --time=4:00:00                 # 4 hours of wall time
#SBATCH -p gpu                          # GPU partition
#SBATCH -A sxk1942                      # Account/Project ID
#SBATCH -c 4                            # 4 processors
#SBATCH --mem=32GB                      # 32GB memory
#SBATCH --gpus=1                        # 1 GPU
#SBATCH -N 1                            # 1 Node

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

# Set CPU optimization environment variables for GPU node
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
export NUMEXPR_NUM_THREADS=4
export SCIKIT_LEARN_ASSUME_FINITE=1  # Slight performance boost for scikit-learn
export CUDA_VISIBLE_DEVICES=0        # Use first GPU
echo "Set CPU threads to 4 and GPU environment for optimization"

# Verify Python and required packages
python << 'END_PYTHON'
import sys
import os
import pandas as pd
import numpy as np
import sklearn
import matplotlib
import joblib
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch
import ai3

print(f"Python path: {sys.executable}")
print(f"Pandas version: {pd.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"PyTorch version: {torch.__version__}")
print(f"Joblib version: {joblib.__version__}")
if hasattr(ai3, '__version__'):
    print(f"AI3 version: {ai3.__version__}")
else:
    print("AI3 version: version not available")

# Check sklearn components
print("\nScikit-learn components check:")
print(f"  SVR: {SVR}")
print(f"  StandardScaler: {StandardScaler}")
print(f"  GridSearchCV: {GridSearchCV}")
print(f"  KFold: {KFold}")

# Check CPU and GPU configuration
print(f"\nCPU configuration:")
print(f"  OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
print(f"  MKL_NUM_THREADS: {os.environ.get('MKL_NUM_THREADS', 'not set')}")
print(f"  OPENBLAS_NUM_THREADS: {os.environ.get('OPENBLAS_NUM_THREADS', 'not set')}")

print(f"\nGPU configuration:")
print(f"  CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'not set')}")
print(f"  PyTorch CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"  GPU device: {torch.cuda.get_device_name(0)}")
    print(f"  GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
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
SCRATCH_DIR=$PFSDIR/svr_prediction_training_${SLURM_JOB_ID}
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

# Print system information
echo "System information:"
echo "Available memory: $(free -h | grep '^Mem:' | awk '{print $2}')"
echo "Available processors: $(nproc)"
echo "Load average: $(uptime | awk -F'load average:' '{print $2}')"

# Print GPU information
echo "GPU information:"
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader,nounits
else
    echo "nvidia-smi not available"
fi

# Run the training script and capture all output
echo "Starting SVR Training..."
echo "====================================="
python python.py 2>&1 | tee training_output.log

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "SVR training completed successfully"
else
    echo "SVR training failed with exit code $?"
    # Copy logs even if script failed
    cp training_output.log $RESULTS_DIR/
    exit 1
fi

# Copy all results to the timestamped results directory
echo "Copying results to: $RESULTS_DIR"
cp -v training_output.log $RESULTS_DIR/
cp -v svr_training_results.csv $RESULTS_DIR/ 2>/dev/null || echo "svr_training_results.csv not found"
cp -v best_svr_model.pkl $RESULTS_DIR/ 2>/dev/null || echo "best_svr_model.pkl not found"
cp -v *.png $RESULTS_DIR/ 2>/dev/null || echo "No PNG files found"

# Create a summary report
echo "Creating summary report..."
cat > $RESULTS_DIR/job_summary.txt << EOF
SVR Training Job Summary
========================
Job ID: $SLURM_JOB_ID
Start Time: $(date)
Submit Directory: $SLURM_SUBMIT_DIR
Results Directory: $RESULTS_DIR
Processors Used: 8
Memory Allocated: 32GB

Files Generated:
- training_output.log: Complete training log
- svr_training_results.csv: Cross-validation results
- best_svr_model.pkl: Best trained SVR model
- svr_main_evaluation.png: Main evaluation dashboard (8 plots)
- svr_hyperparameter_analysis.png: Hyperparameter analysis (6 plots)
- svr_detailed_metrics.png: Detailed metrics analysis (6 plots)
- job_summary.txt: This summary

System Information:
$(free -h | grep '^Mem:')
$(nproc) processors available
Load average: $(uptime | awk -F'load average:' '{print $2}')

Training completed successfully!
EOF

# Print final system status
echo "Final system status:"
echo "Memory usage: $(free -h | grep '^Mem:')"
echo "Load average: $(uptime | awk -F'load average:' '{print $2}')"

# Extract key results from the log if available
if [ -f training_output.log ]; then
    echo "Extracting key results..."
    echo "Key Results Summary:" >> $RESULTS_DIR/job_summary.txt
    echo "==================" >> $RESULTS_DIR/job_summary.txt
    
    # Try to extract the final validation MAPE from the log
    FINAL_MAPE=$(grep "Final validation MAPE:" training_output.log | tail -1 | awk '{print $4}')
    if [ ! -z "$FINAL_MAPE" ]; then
        echo "Final Validation MAPE: $FINAL_MAPE" >> $RESULTS_DIR/job_summary.txt
    fi
    
    # Extract total training time
    TOTAL_TIME=$(grep "Total time:" training_output.log | tail -1 | awk '{print $3}')
    if [ ! -z "$TOTAL_TIME" ]; then
        echo "Total Training Time: $TOTAL_TIME seconds" >> $RESULTS_DIR/job_summary.txt
    fi
    
    # Extract best fold performance
    BEST_FOLD=$(grep "Best fold achieved:" training_output.log | tail -1)
    if [ ! -z "$BEST_FOLD" ]; then
        echo "Best Fold: $BEST_FOLD" >> $RESULTS_DIR/job_summary.txt
    fi
fi

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
echo "To view results:"
echo "  Training log: cat $RESULTS_DIR/training_output.log"
echo "  Results CSV:  cat $RESULTS_DIR/svr_training_results.csv"
echo "  Best model:   $RESULTS_DIR/best_svr_model.pkl"
echo "  Job summary:  cat $RESULTS_DIR/job_summary.txt"
echo "  Visualizations: open $RESULTS_DIR/*.png"
echo ""
echo "To submit this job, run:"
echo "  sbatch job.sh" 