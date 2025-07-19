#!/bin/bash
#SBATCH -J vgg16_implicit_gemm_gpuL40s  # Job name
#SBATCH -o vgg16_implicit_gemm_gpuL40s.out  # Output file
#SBATCH --time=20:00:00                 # 20 hours of wall time
#SBATCH -p gpu                          # GPU partition
#SBATCH -A sxk1942                     # Account/Project ID
#SBATCH -c 8                           # 8 processors
#SBATCH --mem=64GB                     # 64GB memory
#SBATCH -C gpul40s                     # Constraint for L40s GPU
#SBATCH --gres=gpu:1                   # Request 1 GPU

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
    echo "Error: Failed to import required packages or CUDA not available"
    exit 1
fi

# Create timestamp for unique results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=$SLURM_SUBMIT_DIR/results_${TIMESTAMP}
mkdir -p $RESULTS_DIR

# Create a directory in scratch for the job
SCRATCH_DIR=$PFSDIR/vgg16_implicit_gemm_gpuL40s_${SLURM_JOB_ID}
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

# Run the test script and capture all output
echo "Running VGG16 Implicit GEMM performance test..."
python python.py 2>&1 | tee python_output.log

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "VGG16 Implicit GEMM script executed successfully"
else
    echo "VGG16 Implicit GEMM script failed with exit code $?"
    # Copy logs even if script failed
    cp python_output.log $RESULTS_DIR/
    exit 1
fi

# Copy results to the timestamped results directory
echo "Copying results to: $RESULTS_DIR"
cp -ru *.csv python_output.log $RESULTS_DIR/

# Print summary of generated files
echo "Generated files:"
ls -la $RESULTS_DIR/

# Cleanup scratch directory
if [ -d "$SCRATCH_DIR" ]; then
    rm -rf $SCRATCH_DIR
    echo "Cleaned up scratch directory"
fi

# Deactivate virtual environment
deactivate

echo "VGG16 Implicit GEMM L40s job completed successfully. Results are in: $RESULTS_DIR"
echo "Expected output files:"
echo "  - VGG16_implicit_gemm_cuda_overall.csv"
echo "  - VGG16_implicit_gemm_cuda_layers.csv" 
echo "  - python_output.log" 