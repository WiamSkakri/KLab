#!/bin/bash
#SBATCH -J rdf_prediction_training_gpu  # Job name with GPU designation
#SBATCH -o rdf_prediction_training.out  # Output file
#SBATCH --time=5:00:00                 # 5 hours of wall time (reduced for GPU efficiency)
#SBATCH -p gpu                          # GPU partition for cuML acceleration
#SBATCH -A sxk1942                      # Account/Project ID
#SBATCH -c 4                            # 4 processors (GPU workloads need fewer CPUs)
#SBATCH --mem=32GB                      # 32GB memory (GPU handles compute-heavy tasks)
#SBATCH --gpus=1                        # Request 1 GPU for cuML acceleration

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
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE"
echo "GPUs: $SLURM_GPUS"
echo "GPU devices: $CUDA_VISIBLE_DEVICES"
echo "Date/Time: $(date)"

# Check GPU availability
echo "Checking GPU configuration..."
if command -v nvidia-smi &> /dev/null; then
    echo "GPU Information:"
    nvidia-smi
    echo "CUDA Version:"
    nvcc --version 2>/dev/null || echo "NVCC not available"
else
    echo "Warning: nvidia-smi not found. GPU may not be available."
fi

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

# Print CPU information for hybrid CPU-GPU workload
echo "Checking CPU configuration for hybrid workload..."
echo "Number of CPU cores: $(nproc)"
echo "CPU info:"
lscpu | grep -E "(Model name|CPU\(s\)|Thread|Core)"
echo "-----------------------------------"

# Verify Python and required packages (including GPU libraries)
python << 'END_PYTHON'
import sys
import pandas as pd
import sklearn
import numpy as np
import matplotlib
import joblib
import time
from datetime import datetime

print(f"Python path: {sys.executable}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"Joblib version: {joblib.__version__}")

# Check Random Forest capability
from sklearn.ensemble import RandomForestRegressor
rf_test = RandomForestRegressor(n_estimators=10, n_jobs=-1)
print(f"Random Forest Regressor: Available")

# Check GPU libraries
try:
    import cuml
    from cuml.ensemble import RandomForestRegressor as cuRF
    import cupy as cp
    
    print(f"cuML version: {cuml.__version__}")
    print(f"CuPy version: {cp.__version__}")
    
    # Test GPU access
    try:
        cp.cuda.Device(0).use()
        gpu_name = cp.cuda.runtime.getDeviceProperties(0)['name'].decode()
        gpu_memory = cp.cuda.runtime.memGetInfo()[1] / 1024**3
        print(f"GPU Device: {gpu_name}")
        print(f"GPU Memory: {gpu_memory:.1f} GB")
        print("GPU Libraries: ✅ Available")
    except Exception as e:
        print(f"GPU Access Error: {e}")
        print("GPU Libraries: ⚠️ Installed but not accessible")
    
except ImportError as e:
    print(f"GPU Libraries: ❌ Not available ({e})")
    print("Will fall back to CPU-based sklearn")

# Check multi-threading capability
import multiprocessing
print(f"Available CPU cores: {multiprocessing.cpu_count()}")
END_PYTHON

if [ $? -ne 0 ]; then
    echo "Error: Failed to import required packages"
    exit 1
fi

# Create timestamp for unique results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=$SLURM_SUBMIT_DIR/results_gpu_${TIMESTAMP}
mkdir -p $RESULTS_DIR

# Create a directory in scratch for the job
SCRATCH_DIR=$PFSDIR/rdf_prediction_training_gpu_${SLURM_JOB_ID}
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

# Set environment variables for optimal GPU + CPU performance
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export BLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export LAPACK_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Set CUDA environment variables
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=${SLURM_GPUS:-0}

echo "Set threading environment variables:"
echo "OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo "CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"

# Run the Random Forest training script and capture all output
echo "Starting GPU-Accelerated Random Forest Training..."
echo "=================================================="
python python.py 2>&1 | tee training_output.log

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "GPU-accelerated Random Forest training completed successfully"
else
    echo "Random Forest training failed with exit code $?"
    # Copy logs even if script failed
    cp training_output.log $RESULTS_DIR/
    exit 1
fi

# Copy all results to the timestamped results directory
echo "Copying results to: $RESULTS_DIR"
cp -v training_output.log $RESULTS_DIR/
cp -v rdf_training_results.csv $RESULTS_DIR/ 2>/dev/null || echo "rdf_training_results.csv not found"
cp -v rdf_summary_metrics.csv $RESULTS_DIR/ 2>/dev/null || echo "rdf_summary_metrics.csv not found"
cp -v best_rf_model.joblib $RESULTS_DIR/ 2>/dev/null || echo "best_rf_model.joblib not found"
cp -v rf_model_fold_*.joblib $RESULTS_DIR/ 2>/dev/null || echo "Individual fold models not found"

# Copy visualization files
cp -v rdf_evaluation_dashboard.png $RESULTS_DIR/ 2>/dev/null || echo "Evaluation dashboard plot not found"
cp -v rdf_hyperparameter_analysis.png $RESULTS_DIR/ 2>/dev/null || echo "Hyperparameter analysis plot not found"
cp -v rdf_detailed_metrics_analysis.png $RESULTS_DIR/ 2>/dev/null || echo "Detailed metrics analysis plot not found"
cp -v rdf_training_results.png $RESULTS_DIR/ 2>/dev/null || echo "Basic training results plot not found"

# Create a summary report
echo "Creating summary report..."
cat > $RESULTS_DIR/job_summary.txt << EOF
GPU-Accelerated Random Forest Training Job Summary
==================================================
Job ID: $SLURM_JOB_ID
Start Time: $(date)
Submit Directory: $SLURM_SUBMIT_DIR
Results Directory: $RESULTS_DIR

Job Resources:
- CPUs: $SLURM_CPUS_PER_TASK
- Memory: $SLURM_MEM_PER_NODE
- GPUs: $SLURM_GPUS
- Time Limit: 12 hours
- Partition: gpu

GPU Information:
$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU info not available")

Files Generated:
- training_output.log: Complete training log
- rdf_training_results.csv: Detailed cross-validation results
- rdf_summary_metrics.csv: Summary metrics
- best_rf_model.joblib: Best performing model
- rf_model_fold_*.joblib: Individual fold models
- rdf_evaluation_dashboard.png: Comprehensive evaluation dashboard (9 plots)
- rdf_hyperparameter_analysis.png: Hyperparameter effectiveness analysis (6 plots)
- rdf_detailed_metrics_analysis.png: Detailed metrics comparison (6 plots)
- rdf_training_results.png: Basic training visualization (4 plots)
- job_summary.txt: This summary

CPU Information:
$(lscpu | grep -E "(Model name|CPU\(s\)|Thread|Core)")

Training completed successfully!
EOF

# Extract key metrics from the log file for quick reference
echo "Extracting key performance metrics..."
if [ -f training_output.log ]; then
    echo "" >> $RESULTS_DIR/job_summary.txt
    echo "Key Performance Metrics:" >> $RESULTS_DIR/job_summary.txt
    echo "========================" >> $RESULTS_DIR/job_summary.txt
    
    # Extract backend used (GPU vs CPU)
    grep "Backend used:" training_output.log | tail -1 >> $RESULTS_DIR/job_summary.txt || echo "Backend info not found in log"
    
    # Extract average validation MAPE
    grep "Average Val MAPE:" training_output.log | tail -1 >> $RESULTS_DIR/job_summary.txt || echo "Val MAPE not found in log"
    
    # Extract average validation R²
    grep "Average Val R²:" training_output.log | tail -1 >> $RESULTS_DIR/job_summary.txt || echo "Val R² not found in log"
    
    # Extract total training time
    grep "Total Training Time:" training_output.log | tail -1 >> $RESULTS_DIR/job_summary.txt || echo "Training time not found in log"
    
    # Extract best fold information
    grep "Best performing fold:" training_output.log | tail -1 >> $RESULTS_DIR/job_summary.txt || echo "Best fold not found in log"
fi

# Print resource usage summary
echo "Resource Usage Summary:"
echo "Number of CPU cores used: $SLURM_CPUS_PER_TASK"
echo "Memory allocated: $SLURM_MEM_PER_NODE"
echo "GPUs allocated: $SLURM_GPUS"

# Cleanup scratch directory
if [ -d "$SCRATCH_DIR" ]; then
    rm -rf $SCRATCH_DIR
    echo "Cleaned up scratch directory"
fi

# Deactivate virtual environment
deactivate

echo "GPU job completed successfully. Results are in: $RESULTS_DIR"
echo "Job finished at: $(date)"
echo ""
echo "To view results:"
echo "  Training log: cat $RESULTS_DIR/training_output.log"
echo "  Results CSV:  cat $RESULTS_DIR/rdf_training_results.csv"
echo "  Summary:      cat $RESULTS_DIR/rdf_summary_metrics.csv"
echo "  Job summary:  cat $RESULTS_DIR/job_summary.txt"
echo ""
echo "Visualizations:"
echo "  Main dashboard:     $RESULTS_DIR/rdf_evaluation_dashboard.png"
echo "  Hyperparameters:    $RESULTS_DIR/rdf_hyperparameter_analysis.png"
echo "  Detailed metrics:   $RESULTS_DIR/rdf_detailed_metrics_analysis.png"
echo "  Basic overview:     $RESULTS_DIR/rdf_training_results.png"
echo ""
echo "To load the best model in Python:"
echo "  import joblib"
echo "  model = joblib.load('$RESULTS_DIR/best_rf_model.joblib')" 