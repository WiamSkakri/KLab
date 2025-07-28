#!/bin/bash
#SBATCH -J rdf_l40s_gpu_training         # Job name for L40s GPU training
#SBATCH -o rdf_l40s_training.out         # Output file
#SBATCH --time=6:00:00                   # 6 hours of wall time (L40s optimized)
#SBATCH -p gpu                           # GPU partition for L40s
#SBATCH -A sxk1942                       # Account/Project ID
#SBATCH -c 8                             # 8 processors (L40s benefits from more CPUs)
#SBATCH --mem=64GB                       # 64GB memory (L40s has 48GB VRAM)
#SBATCH --gpus=rtx_l40s:1                # Request 1 L40s GPU specifically
#SBATCH --constraint=l40s                # Ensure L40s GPU allocation

# Exit on any error
set -e

# Print debug information
echo "============================================="
echo "L40s GPU Random Forest Training Job Started"
echo "============================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Job name: $SLURM_JOB_NAME"
echo "Current directory: $(pwd)"
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "Date/Time: $(date)"
echo "Hostname: $(hostname)"

echo -e "\nResource Allocation:"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "GPUs: $SLURM_GPUS"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "Partition: $SLURM_JOB_PARTITION"

echo -e "\nDirectory Contents:"
ls -la

# Check L40s GPU availability and configuration
echo -e "\n========================================="
echo "L40s GPU Configuration Check"
echo "========================================="
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi
    
    echo -e "\nDetailed GPU Memory Info:"
    nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used --format=csv
    
    echo -e "\nCUDA Version:"
    nvcc --version 2>/dev/null || echo "NVCC not available"
    
    echo -e "\nCUDA Runtime Version:"
    python -c "import torch; print(f'PyTorch CUDA version: {torch.version.cuda}')" 2>/dev/null || echo "PyTorch not available"
else
    echo "Error: nvidia-smi not found. L40s GPU may not be available."
    exit 1
fi

# Verify L40s GPU specifically
echo -e "\nVerifying L40s GPU..."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)
if [[ "$GPU_NAME" == *"L40S"* ]] || [[ "$GPU_NAME" == *"L40s"* ]]; then
    echo "✅ L40s GPU detected: $GPU_NAME"
else
    echo "⚠️  Warning: Expected L40s GPU but found: $GPU_NAME"
    echo "Continuing with available GPU..."
fi

# Check if virtual environment exists
if [ ! -d "$HOME/ai3_env" ]; then
    echo "Error: Virtual environment not found at $HOME/ai3_env"
    echo "Please create and configure the ai3_env virtual environment first"
    exit 1
fi

# Activate the ai3_env virtual environment
echo -e "\n========================================="
echo "Environment Setup"
echo "========================================="
echo "Activating ai3_env virtual environment..."
source $HOME/ai3_env/bin/activate || {
    echo "Error: Failed to activate ai3_env virtual environment"
    exit 1
}

echo "✅ Virtual environment activated"
echo "Python path: $(which python)"
echo "Python version: $(python --version)"

# Verify required packages for L40s GPU acceleration
echo -e "\nChecking L40s GPU-optimized packages..."

echo "Checking cuML (GPU Random Forest)..."
python -c "import cuml; print(f'✅ cuML version: {cuml.__version__}')" 2>/dev/null || echo "❌ cuML not available"

echo "Checking CuPy (GPU arrays)..."
python -c "import cupy as cp; print(f'✅ CuPy version: {cp.__version__}'); print(f'✅ CUDA version: {cp.cuda.runtime.runtimeGetVersion()}')" 2>/dev/null || echo "❌ CuPy not available"

echo "Checking PyTorch..."
python -c "import torch; print(f'✅ PyTorch version: {torch.__version__}'); print(f'✅ CUDA available: {torch.cuda.is_available()}'); print(f'✅ Device count: {torch.cuda.device_count()}')" 2>/dev/null || echo "❌ PyTorch not available"

echo "Checking standard ML packages..."
python -c "import pandas; print(f'✅ Pandas version: {pandas.__version__}')" 2>/dev/null || echo "❌ Pandas not available"
python -c "import numpy; print(f'✅ NumPy version: {numpy.__version__}')" 2>/dev/null || echo "❌ NumPy not available"
python -c "import sklearn; print(f'✅ Scikit-learn version: {sklearn.__version__}')" 2>/dev/null || echo "❌ Scikit-learn not available"
python -c "import matplotlib; print(f'✅ Matplotlib version: {matplotlib.__version__}')" 2>/dev/null || echo "❌ Matplotlib not available"
python -c "import joblib; print(f'✅ Joblib version: {joblib.__version__}')" 2>/dev/null || echo "❌ Joblib not available"

# Check for the L40s data file
echo -e "\n========================================="
echo "Data File Verification"
echo "========================================="
if [ -f "combined_l40s.csv" ]; then
    echo "✅ L40s data file found: combined_l40s.csv"
    echo "File size: $(ls -lh combined_l40s.csv | awk '{print $5}')"
    echo "Number of lines: $(wc -l < combined_l40s.csv)"
    echo "First few lines:"
    head -n 3 combined_l40s.csv
else
    echo "❌ Error: combined_l40s.csv not found!"
    echo "Available CSV files:"
    ls -la *.csv 2>/dev/null || echo "No CSV files found in current directory"
    echo "Please ensure combined_l40s.csv is in the same directory as this job script"
    exit 1
fi

# Set environment variables for optimal L40s performance
echo -e "\n========================================="
echo "L40s GPU Optimization Settings"
echo "========================================="
export CUDA_VISIBLE_DEVICES=0
export RAPIDS_NO_INITIALIZE=1
export CUML_LOG_LEVEL=INFO
export CUPY_CACHE_DIR=/tmp/cupy_cache_$SLURM_JOB_ID

# Create temporary cache directory
mkdir -p $CUPY_CACHE_DIR

echo "✅ Environment variables set for L40s optimization"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "CUPY_CACHE_DIR: $CUPY_CACHE_DIR"

# Set output directory for results
OUTPUT_DIR="output_l40s_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo -e "\n========================================="
echo "Starting L40s GPU Random Forest Training"
echo "========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Script: python.py"
echo "Data: combined_l40s.csv"
echo "Hardware: L40s GPU"
echo "Start time: $(date)"

# Change to output directory and copy files
cd $OUTPUT_DIR
cp ../python.py .
cp ../combined_l40s.csv .

echo -e "\nWorking directory: $(pwd)"

# Run the L40s GPU training script with comprehensive logging
echo -e "\n🚀 Executing L40s GPU Random Forest training..."
python python.py 2>&1 | tee training_output.log

TRAINING_EXIT_CODE=$?

echo -e "\n========================================="
echo "L40s GPU Training Completion"
echo "========================================="
echo "End time: $(date)"
echo "Training exit code: $TRAINING_EXIT_CODE"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ L40s GPU training completed successfully!"
    
    echo -e "\nGenerated files:"
    ls -la *.csv *.png *.joblib *.log 2>/dev/null || echo "No output files found"
    
    echo -e "\nFile sizes:"
    du -h * 2>/dev/null | sort -hr || echo "No files to show"
    
    # Display training summary if available
    if [ -f "rdf_l40s_summary_metrics.csv" ]; then
        echo -e "\n📊 Training Summary:"
        cat rdf_l40s_summary_metrics.csv
    fi
    
    # Show the best performance achieved
    if [ -f "rdf_l40s_training_results.csv" ]; then
        echo -e "\n🏆 Best Performance:"
        python -c "
import pandas as pd
try:
    df = pd.read_csv('rdf_l40s_training_results.csv')
    best_fold = df.loc[df['val_mape'].idxmin()]
    print(f'Best Fold: {best_fold[\"fold\"]}')
    print(f'Validation MAPE: {best_fold[\"val_mape\"]:.2f}%')
    print(f'Validation R²: {best_fold[\"val_r2\"]:.4f}')
    print(f'Training Time: {best_fold[\"training_time\"]:.1f} seconds')
except Exception as e:
    print(f'Could not read results: {e}')
" 2>/dev/null || echo "Could not display performance summary"
    fi
    
else
    echo "❌ L40s GPU training failed with exit code: $TRAINING_EXIT_CODE"
    echo -e "\nLast 50 lines of output:"
    tail -n 50 training_output.log 2>/dev/null || echo "No log file available"
fi

# GPU memory status after training
echo -e "\n========================================="
echo "Final L40s GPU Status"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu --format=csv

# Cleanup
echo -e "\nCleaning up temporary files..."
rm -rf $CUPY_CACHE_DIR 2>/dev/null || echo "Cache directory already cleaned"

# Final summary
echo -e "\n========================================="
echo "L40s GPU Job Summary"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Hardware: L40s GPU"
echo "Dataset: combined_l40s.csv"
echo "Output directory: $OUTPUT_DIR"
echo "Final status: $([ $TRAINING_EXIT_CODE -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
echo "Job completed at: $(date)"
echo "============================================="

# Exit with the same code as the training script
exit $TRAINING_EXIT_CODE 