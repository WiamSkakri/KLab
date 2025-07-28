#!/bin/bash
#SBATCH -J xgb_l40s_gpu_training          # Job name for L40s XGBoost training
#SBATCH -o xgb_l40s_training.out          # Output file
#SBATCH --time=4:00:00                    # 4 hours of wall time (XGBoost typically faster)
#SBATCH -p gpu                            # GPU partition for L40s
#SBATCH -A sxk1942                        # Account/Project ID
#SBATCH -c 8                              # 8 processors (L40s benefits from more CPUs)
#SBATCH --mem=64GB                        # 64GB memory (L40s has 48GB VRAM)
#SBATCH --gpus=rtx_l40s:1                 # Request 1 L40s GPU specifically
#SBATCH --constraint=l40s                 # Ensure L40s GPU allocation

# Exit on any error
set -e

# Print debug information
echo "============================================="
echo "L40s GPU XGBoost Training Job Started"
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

echo "Checking XGBoost (GPU)..."
python -c "
import xgboost as xgb
print(f'✅ XGBoost version: {xgb.__version__}')
# Test GPU functionality
import numpy as np
try:
    test_data = xgb.DMatrix(np.random.rand(10, 5), label=np.random.rand(10))
    test_params = {'tree_method': 'hist', 'device': 'cuda:0'}
    xgb.train(test_params, test_data, num_boost_round=1, verbose_eval=False)
    print('✅ XGBoost GPU support confirmed')
except Exception as e:
    print(f'❌ XGBoost GPU not available: {e}')
" 2>/dev/null || echo "❌ XGBoost not available"

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

# Set environment variables for optimal L40s XGBoost performance
echo -e "\n========================================="
echo "L40s XGBoost GPU Optimization Settings"
echo "========================================="
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_LAUNCH_BLOCKING=0  # Allow asynchronous GPU operations for better performance

echo "✅ Environment variables set for L40s XGBoost optimization"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"

# Set output directory for results
OUTPUT_DIR="output_xgb_l40s_$(date +%Y%m%d_%H%M%S)"
mkdir -p $OUTPUT_DIR

echo -e "\n========================================="
echo "Starting L40s GPU XGBoost Training"
echo "========================================="
echo "Output directory: $OUTPUT_DIR"
echo "Script: python.py"
echo "Data: combined_l40s.csv"
echo "Hardware: L40s GPU"
echo "Algorithm: XGBoost"
echo "Start time: $(date)"

# Change to output directory and copy files
cd $OUTPUT_DIR
cp ../python.py .
cp ../combined_l40s.csv .

echo -e "\nWorking directory: $(pwd)"

# Run the L40s GPU XGBoost training script with comprehensive logging
echo -e "\n🚀 Executing L40s GPU XGBoost training..."
python python.py 2>&1 | tee training_output.log

TRAINING_EXIT_CODE=$?

echo -e "\n========================================="
echo "L40s GPU XGBoost Training Completion"
echo "========================================="
echo "End time: $(date)"
echo "Training exit code: $TRAINING_EXIT_CODE"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ L40s GPU XGBoost training completed successfully!"
    
    echo -e "\nGenerated files:"
    ls -la *.csv *.png *.joblib *.log 2>/dev/null || echo "No output files found"
    
    echo -e "\nFile sizes:"
    du -h * 2>/dev/null | sort -hr || echo "No files to show"
    
    # Display training summary if available
    if [ -f "xgb_l40s_summary_metrics.csv" ]; then
        echo -e "\n📊 XGBoost Training Summary:"
        cat xgb_l40s_summary_metrics.csv
    fi
    
    # Show the best performance achieved
    if [ -f "xgb_l40s_training_results.csv" ]; then
        echo -e "\n🏆 Best XGBoost Performance:"
        python -c "
import pandas as pd
try:
    df = pd.read_csv('xgb_l40s_training_results.csv')
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
    echo "❌ L40s GPU XGBoost training failed with exit code: $TRAINING_EXIT_CODE"
    echo -e "\nLast 50 lines of output:"
    tail -n 50 training_output.log 2>/dev/null || echo "No log file available"
fi

# GPU memory status after training
echo -e "\n========================================="
echo "Final L40s GPU Status"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu --format=csv

# Final summary
echo -e "\n========================================="
echo "L40s GPU XGBoost Job Summary"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Hardware: L40s GPU"
echo "Algorithm: XGBoost"
echo "Dataset: combined_l40s.csv"
echo "Output directory: $OUTPUT_DIR"
echo "Final status: $([ $TRAINING_EXIT_CODE -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
echo "Job completed at: $(date)"
echo "============================================="

# Exit with the same code as the training script
exit $TRAINING_EXIT_CODE 