#!/bin/bash
#SBATCH --account=sxk1942
#SBATCH --job-name=googlenet_implicit_precomp_gemm
#SBATCH --output=googlenet_implicit_precomp_gemm_%j.out
#SBATCH --error=googlenet_implicit_precomp_gemm_%j.err
#SBATCH --time=20:00:00
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --gres=gpu:1

echo "=========================================="
echo "SLURM job ID: $SLURB_JOB_ID"
echo "SLURM job name: $SLURM_JOB_NAME"
echo "Start time: $(date)"
echo "Working directory: $(pwd)"
echo "=========================================="

# Environment setup
source activate ai3_env

# Display environment information
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version)"
echo "Hostname: $(hostname)"

# Load environment modules if needed
module load cuda

# Run the GoogleNet Implicit Precomputed GEMM implementation
echo "Running GoogleNet Implicit Precomputed GEMM implementation..."
python python.py

echo "=========================================="
echo "Job completed at: $(date)"
echo "==========================================" 