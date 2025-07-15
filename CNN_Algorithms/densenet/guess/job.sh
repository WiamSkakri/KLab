#!/bin/bash
#SBATCH -J densenet_guess_gpu    # Job name
#SBATCH -o densenet_guess_gpu.out  # Output file
#SBATCH --time=20:00:00          # 20 hours of wall time
#SBATCH -p gpu                   # GPU partition
#SBATCH -A sxk1942              # Account/Project ID
#SBATCH -c 4                    # 4 processors
#SBATCH --mem=32GB             # 32GB memory
#SBATCH --gpus=1               # Request 1 GPU

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
import ai3

print(f"Python path: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
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
SCRATCH_DIR=$PFSDIR/densenet_guess_gpu_${SLURM_JOB_ID}
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

# Set GPU optimization environment variables
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export CUDA_VISIBLE_DEVICES=0
echo "Set CUDA environment variables for optimal GPU performance"

# Run the test script and capture all output
echo "Running DenseNet Guess Algorithm Python script..."
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

# Print summary of results
echo "Results Summary:"
echo "==============="
if [ -f "DenseNet121_guess_cuda_overall.csv" ]; then
    echo "Overall results file: DenseNet121_guess_cuda_overall.csv"
    echo "Number of test cases: $(tail -n +2 DenseNet121_guess_cuda_overall.csv | wc -l)"
fi

if [ -f "DenseNet121_guess_cuda_layers.csv" ]; then
    echo "Layer-wise results file: DenseNet121_guess_cuda_layers.csv"
    echo "Number of layer measurements: $(tail -n +2 DenseNet121_guess_cuda_layers.csv | wc -l)"
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
echo "DenseNet Guess Algorithm Job completed successfully!"
echo "Results directory: $RESULTS_DIR"
echo "===========================================" 