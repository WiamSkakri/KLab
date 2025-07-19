#!/bin/bash
#SBATCH --job-name=poly_gpu_regression
#SBATCH --output=poly_gpu_regression_%j.out
#SBATCH --error=poly_gpu_regression_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Load necessary modules
module load python/3.9
module load cuda/11.8
module load pytorch

# Activate virtual environment if you have one
# source /path/to/your/venv/bin/activate

# Set CUDA device (optional, PyTorch will automatically detect)
export CUDA_VISIBLE_DEVICES=0

# Navigate to the working directory
cd $SLURM_SUBMIT_DIR

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Job ID: $SLURM_JOB_ID"
echo "GPU allocated: $CUDA_VISIBLE_DEVICES"

# Check GPU availability
nvidia-smi

# Install required packages if not already installed
pip install pandas numpy matplotlib scikit-learn torch torchvision joblib

echo "Starting GPU Polynomial Regression Training..."

# Run the GPU polynomial regression script
python polynomial_gpu.py

echo "Job completed at: $(date)"

# Print some job statistics
echo "Job Statistics:"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Node: $SLURM_NODELIST"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo "Start Time: $SLURM_JOB_START_TIME" 