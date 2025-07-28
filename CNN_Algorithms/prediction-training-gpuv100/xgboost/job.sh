#!/bin/bash
#SBATCH -J xgb_prediction_training_v100  # Job name with V100 GPU designation
#SBATCH -o xgb_v100_training.out         # Output file
#SBATCH --time=24:00:00                  # 24 hours of wall time for comprehensive training
#SBATCH -p gpu                           # GPU partition for XGBoost acceleration
#SBATCH -A sxk1942                       # Account/Project ID
#SBATCH -c 4                             # 4 processors (GPU workloads need fewer CPUs)
#SBATCH --mem=28GB                       # 28GB memory (V100 has 32GB VRAM)
#SBATCH --gpus=1                         # Request 1 GPU for XGBoost acceleration

# Exit on any error
set -e

# Print debug information
echo "============================================="
echo "V100 GPU XGBoost Training Job Started"
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

# Check GPU availability and configuration
echo -e "\n========================================="
echo "V100 GPU Configuration Check"
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
    echo "Error: nvidia-smi not found. GPU may not be available."
    exit 1
fi

# Verify GPU allocation (optimized for V100)
echo -e "\nVerifying V100 GPU allocation..."
GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -n1)
if [[ "$GPU_NAME" == *"V100"* ]]; then
    echo "✅ V100 GPU detected: $GPU_NAME"
else
    echo "⚠️  Note: Expected V100 GPU but found: $GPU_NAME"
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

# Verify required packages for V100 GPU acceleration
echo -e "\nChecking V100 GPU-optimized packages..."

echo "Checking XGBoost (GPU)..."
python -c "
import xgboost as xgb
print(f'✅ XGBoost version: {xgb.__version__}')
# Test GPU functionality
import numpy as np
try:
    test_data = xgb.DMatrix(np.random.rand(10, 5), label=np.random.rand(10))
    test_params = {'tree_method': 'hist', 'device': 'cuda'}
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

# Check for the data file
echo -e "\n========================================="
echo "Data File Verification"
echo "========================================="
if [ -f "combined_v100.csv" ]; then
    echo "✅ V100 data file found: combined_v100.csv"
    echo "File size: $(ls -lh combined_v100.csv | awk '{print $5}')"
    echo "Number of lines: $(wc -l < combined_v100.csv)"
    echo "First few lines:"
    head -n 3 combined_v100.csv
else
    echo "❌ Error: combined_v100.csv not found!"
    echo "Available CSV files:"
    ls -la *.csv 2>/dev/null || echo "No CSV files found in current directory"
    echo "Please ensure combined_v100.csv is in the same directory as this job script"
    exit 1
fi

# Set environment variables for optimal V100 XGBoost performance
echo -e "\n========================================="
echo "V100 XGBoost GPU Optimization Settings"
echo "========================================="
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export CUDA_LAUNCH_BLOCKING=0  # Allow asynchronous GPU operations for better performance

echo "✅ Environment variables set for V100 XGBoost optimization"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"

# Create timestamp for unique results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=$SLURM_SUBMIT_DIR/results_v100_xgb_${TIMESTAMP}
mkdir -p $RESULTS_DIR

# Create a directory in scratch for the job
SCRATCH_DIR=$PFSDIR/xgb_v100_training_gpu_${SLURM_JOB_ID}
if ! mkdir -p $SCRATCH_DIR; then
    echo "Failed to create scratch directory: $SCRATCH_DIR"
    exit 1
fi
echo "Created scratch directory: $SCRATCH_DIR"

# Copy the script and data to the scratch directory
cp python.py $SCRATCH_DIR/
cp combined_v100.csv $SCRATCH_DIR/
echo "Copied Python script and data to scratch directory"

# Change to the scratch directory
cd $SCRATCH_DIR
echo "Changed to scratch directory: $(pwd)"

echo -e "\n========================================="
echo "Starting V100 GPU XGBoost Training"
echo "========================================="
echo "Results directory: $RESULTS_DIR"
echo "Scratch directory: $SCRATCH_DIR"
echo "Script: python.py"
echo "Data: combined_v100.csv"
echo "Hardware: V100 GPU (32GB VRAM)"
echo "Algorithm: XGBoost"
echo "Start time: $(date)"

# Run the V100 GPU XGBoost training script with comprehensive logging
echo -e "\n🚀 Executing V100 GPU XGBoost training..."
python python.py 2>&1 | tee training_output.log

TRAINING_EXIT_CODE=$?

echo -e "\n========================================="
echo "V100 GPU XGBoost Training Completion"
echo "========================================="
echo "End time: $(date)"
echo "Training exit code: $TRAINING_EXIT_CODE"

if [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "✅ V100 GPU XGBoost training completed successfully!"
    
    echo -e "\nGenerated files:"
    ls -la *.csv *.png *.joblib *.log 2>/dev/null || echo "No output files found"
    
    echo -e "\nFile sizes:"
    du -h * 2>/dev/null | sort -hr || echo "No files to show"
    
    # Copy all results to the timestamped results directory
    echo "Copying results to: $RESULTS_DIR"
    cp -v training_output.log $RESULTS_DIR/
    cp -v xgb_*training_results.csv $RESULTS_DIR/ 2>/dev/null || echo "Training results CSV not found"
    cp -v xgb_*summary_metrics.csv $RESULTS_DIR/ 2>/dev/null || echo "Summary metrics CSV not found"
    cp -v best_*xgb_model.joblib $RESULTS_DIR/ 2>/dev/null || echo "Best model not found"
    cp -v xgb_model_fold_*.joblib $RESULTS_DIR/ 2>/dev/null || echo "Individual fold models not found"
    
    # Copy visualization files
    cp -v xgb_*.png $RESULTS_DIR/ 2>/dev/null || echo "Visualization plots not found"
    
    # Display training summary if available
    if [ -f "xgb_*summary_metrics.csv" ]; then
        echo -e "\n📊 V100 XGBoost Training Summary:"
        cat xgb_*summary_metrics.csv 2>/dev/null || echo "Could not display summary"
    fi
    
    # Show the best performance achieved
    if [ -f "xgb_*training_results.csv" ]; then
        echo -e "\n🏆 Best V100 XGBoost Performance:"
        python -c "
import pandas as pd
import glob
try:
    csv_files = glob.glob('xgb_*training_results.csv')
    if csv_files:
        df = pd.read_csv(csv_files[0])
        best_fold = df.loc[df['val_mape'].idxmin()]
        print(f'Best Fold: {best_fold[\"fold\"]}')
        print(f'Validation MAPE: {best_fold[\"val_mape\"]:.2f}%')
        print(f'Validation R²: {best_fold[\"val_r2\"]:.4f}')
        print(f'Training Time: {best_fold[\"training_time\"]:.1f} seconds')
    else:
        print('No training results CSV found')
except Exception as e:
    print(f'Could not read results: {e}')
" 2>/dev/null || echo "Could not display performance summary"
    fi
    
else
    echo "❌ V100 GPU XGBoost training failed with exit code: $TRAINING_EXIT_CODE"
    echo -e "\nLast 50 lines of output:"
    tail -n 50 training_output.log 2>/dev/null || echo "No log file available"
    
    # Copy logs even if script failed
    cp training_output.log $RESULTS_DIR/ 2>/dev/null || echo "Could not copy log file"
fi

# Create a summary report
echo "Creating summary report..."
cat > $RESULTS_DIR/job_summary.txt << EOF
V100 GPU-Accelerated XGBoost Training Job Summary
=================================================
Job ID: $SLURM_JOB_ID
Start Time: $(date)
Submit Directory: $SLURM_SUBMIT_DIR
Results Directory: $RESULTS_DIR

Job Resources:
- CPUs: $SLURM_CPUS_PER_TASK
- Memory: $SLURM_MEM_PER_NODE
- GPUs: $SLURM_GPUS
- Time Limit: 24 hours
- Partition: gpu

GPU Information:
$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU info not available")

Algorithm: XGBoost
Training Status: $([ $TRAINING_EXIT_CODE -eq 0 ] && echo "SUCCESS" || echo "FAILED")
EOF

# Extract key metrics from the log file for quick reference
if [ -f training_output.log ] && [ $TRAINING_EXIT_CODE -eq 0 ]; then
    echo "" >> $RESULTS_DIR/job_summary.txt
    echo "Key Performance Metrics:" >> $RESULTS_DIR/job_summary.txt
    echo "========================" >> $RESULTS_DIR/job_summary.txt
    
    # Extract GPU usage confirmation
    grep -i "xgboost.*gpu\|device.*cuda" training_output.log | head -5 >> $RESULTS_DIR/job_summary.txt 2>/dev/null || echo "GPU usage info not found"
    
    # Extract average validation MAPE
    grep "Average Val MAPE:" training_output.log | tail -1 >> $RESULTS_DIR/job_summary.txt 2>/dev/null || echo "Val MAPE not found"
    
    # Extract average validation R²
    grep "Average Val R²:" training_output.log | tail -1 >> $RESULTS_DIR/job_summary.txt 2>/dev/null || echo "Val R² not found"
    
    # Extract total training time
    grep "Total Training Time:" training_output.log | tail -1 >> $RESULTS_DIR/job_summary.txt 2>/dev/null || echo "Training time not found"
fi

# GPU memory status after training
echo -e "\n========================================="
echo "Final V100 GPU Status"
echo "========================================="
nvidia-smi --query-gpu=name,memory.total,memory.free,memory.used,utilization.gpu --format=csv

# Cleanup scratch directory
if [ -d "$SCRATCH_DIR" ]; then
    rm -rf $SCRATCH_DIR
    echo "Cleaned up scratch directory"
fi

# Final summary
echo -e "\n========================================="
echo "V100 GPU XGBoost Job Summary"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Hardware: V100 GPU (32GB VRAM)"
echo "Algorithm: XGBoost"
echo "Dataset: combined_v100.csv"
echo "Results directory: $RESULTS_DIR"
echo "Final status: $([ $TRAINING_EXIT_CODE -eq 0 ] && echo "SUCCESS" || echo "FAILED")"
echo "Job completed at: $(date)"

echo ""
echo "To view results:"
echo "  Training log: cat $RESULTS_DIR/training_output.log"
echo "  Job summary:  cat $RESULTS_DIR/job_summary.txt"
echo "============================================="

# Exit with the same code as the training script
exit $TRAINING_EXIT_CODE 
