#!/bin/bash
#SBATCH -J nn_prediction_training      # Job name
#SBATCH -o nn_prediction_training.out  # Output file
#SBATCH --time=4:00:00                # 4 hours of wall time
#SBATCH -p gpu                         # GPU partition
#SBATCH -A sxk1942                     # Account/Project ID
#SBATCH -c 4                           # 4 processors
#SBATCH --mem=32GB                     # 32GB memory
#SBATCH --gpus=1                       # Request 1 GPU

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

# Check CUDA availability and print GPU information
echo "Checking CUDA and GPU configuration..."
nvidia-smi
echo "-----------------------------------"

# Verify Python and required packages, including CUDA support
python << 'END_PYTHON'
import sys
import torch
import torchvision
import pandas as pd
import sklearn
import numpy as np
import matplotlib
import ai3

print(f"Python path: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
if hasattr(ai3, '__version__'):
    print(f"AI3 version: {ai3.__version__}")
else:
    print("AI3 version: version not available")
END_PYTHON

if [ $? -ne 0 ]; then
    echo "Error: Failed to import required packages or CUDA not available"
    exit 1
fi

# Create timestamp for unique results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=$SLURM_SUBMIT_DIR/results_${TIMESTAMP}
mkdir -p $RESULTS_DIR

# Create a directory in scratch for the job
SCRATCH_DIR=$PFSDIR/nn_prediction_training_${SLURM_JOB_ID}
if ! mkdir -p $SCRATCH_DIR; then
    echo "Failed to create scratch directory: $SCRATCH_DIR"
    exit 1
fi
echo "Created scratch directory: $SCRATCH_DIR"

# Check if required files exist
if [ ! -f nn_hpc.py ]; then
    echo "Error: nn_hpc.py not found in current directory"
    exit 1
fi

if [ ! -f combined.csv ]; then
    echo "Error: combined.csv not found in current directory"
    echo "Please ensure the combined.csv file is in the same directory as this job script"
    exit 1
fi

# Copy the script and data to the scratch directory
cp nn_hpc.py $SCRATCH_DIR/
cp combined.csv $SCRATCH_DIR/
echo "Copied Python script and data to scratch directory"

# Change to the scratch directory
cd $SCRATCH_DIR
echo "Changed to scratch directory"

# Print data file information
echo "Data file information:"
echo "CSV file size: $(ls -lh combined.csv | awk '{print $5}')"
echo "CSV file rows: $(wc -l < combined.csv)"

# Run the training script and capture all output
echo "Starting Neural Network Training..."
echo "====================================="
python nn_hpc.py 2>&1 | tee training_output.log

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Neural Network training completed successfully"
else
    echo "Neural Network training failed with exit code $?"
    # Copy logs even if script failed
    cp training_output.log $RESULTS_DIR/
    exit 1
fi

# Copy all results to the timestamped results directory
echo "Copying results to: $RESULTS_DIR"
cp -v training_output.log $RESULTS_DIR/
cp -v training_results.csv $RESULTS_DIR/ 2>/dev/null || echo "training_results.csv not found"
cp -v best_model.pth $RESULTS_DIR/ 2>/dev/null || echo "best_model.pth not found"
cp -v *.png $RESULTS_DIR/ 2>/dev/null || echo "No PNG files found"

# Create a summary report
echo "Creating summary report..."
cat > $RESULTS_DIR/job_summary.txt << EOF
Neural Network Training Job Summary
===================================
Job ID: $SLURM_JOB_ID
Start Time: $(date)
Submit Directory: $SLURM_SUBMIT_DIR
Results Directory: $RESULTS_DIR

Files Generated:
- training_output.log: Complete training log
- training_results.csv: Cross-validation results
- best_model.pth: Best trained model
- job_summary.txt: This summary

GPU Information:
$(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits)

Training completed successfully!
EOF

# Print final GPU memory usage
echo "Final GPU memory usage:"
nvidia-smi

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
echo "  Results CSV:  cat $RESULTS_DIR/training_results.csv"
echo "  Job summary:  cat $RESULTS_DIR/job_summary.txt" 