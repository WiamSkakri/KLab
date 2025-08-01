#!/bin/bash
#SBATCH -J densenet_smm_cpu      # Job name
#SBATCH -o densenet_smm_cpu.out  # Output file
#SBATCH --time=24:00:00       # 24 hours of wall time
#SBATCH -N 1                  # 1 Node
#SBATCH -c 4                  # 4 processors
#SBATCH --mem=64gb            # 64GB memory

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

# Set library paths to help find Python shared libraries
export LD_LIBRARY_PATH=$HOME/ai3_env/lib:$LD_LIBRARY_PATH
echo "Set LD_LIBRARY_PATH to include virtual environment libraries"

# Print environment information
echo "Python interpreter: $(which python)"
echo "LD_LIBRARY_PATH: $LD_LIBRARY_PATH"
echo "Virtual environment location: $VIRTUAL_ENV"
echo "Checking Python executable..."
ldd $(which python) || echo "ldd failed for python executable"
echo "Python version: $(python --version)"

# Verify Python and required packages
python << 'END_PYTHON'
import sys
import torch
import torchvision
import ai3

print(f"Python path: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
if hasattr(ai3, '__version__'):
    print(f"AI3 version: {ai3.__version__}")
else:
    print("AI3 version: version not available")
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
SCRATCH_DIR=$PFSDIR/densenet_smm_cpu_${SLURM_JOB_ID}
if ! mkdir -p $SCRATCH_DIR; then
    echo "Failed to create scratch directory: $SCRATCH_DIR"
    exit 1
fi
echo "Created scratch directory: $SCRATCH_DIR"

# Check if Python script exists
if [ ! -f python.py ]; then
    echo "Error: python.py not found in current directory"
    exit 1
fi

# Copy the test script to the scratch directory
cp python.py $SCRATCH_DIR/
echo "Copied Python script to scratch directory"

# Change to the scratch directory
cd $SCRATCH_DIR
echo "Changed to scratch directory"

# Set CPU optimization environment variables
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export OPENBLAS_NUM_THREADS=4
echo "Set CPU threads to 4 for optimization"

# Run the test script and capture all output
echo "Running Python script..."
python python.py 2>&1 | tee python_output.log

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "Python script executed successfully"
else
    echo "Python script failed with exit code $?"
    # Copy logs even if script failed
    cp python_output.log $RESULTS_DIR/
    exit 1
fi

# Copy results to the timestamped results directory
echo "Copying results to: $RESULTS_DIR"
cp -ru *.csv python_output.log $RESULTS_DIR/

# Cleanup scratch directory
if [ -d "$SCRATCH_DIR" ]; then
    rm -rf $SCRATCH_DIR
    echo "Cleaned up scratch directory"
fi

# Deactivate virtual environment
deactivate

echo "Job completed successfully. Results are in: $RESULTS_DIR"
