#!/bin/bash
#SBATCH -J nn_improved_training        # Job name
#SBATCH -o nn_improved_training.out    # Output file
#SBATCH --time=16:00:00                # 16 hours for enhanced training
#SBATCH -p gpu                         # GPU partition
#SBATCH -A sxk1942                     # Account/Project ID
#SBATCH -c 6                           # 6 processors for enhanced model
#SBATCH --mem=64GB                     # 64GB memory for polynomial features
#SBATCH --gpus=1                       # Request 1 GPU

# Exit on any error
set -e

# Print debug information
echo "🚀 IMPROVED CNN EXECUTION TIME PREDICTION TRAINING"
echo "=================================================="
echo "Current directory: $(pwd)"
echo "Contents of current directory:"
ls -la
echo "SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
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

# Verify Python and required packages
python << 'END_PYTHON'
import sys
import torch
import pandas as pd
import sklearn
import numpy as np
import matplotlib

print(f"Python path: {sys.executable}")
print(f"PyTorch version: {torch.__version__}")
print(f"Pandas version: {pd.__version__}")
print(f"Scikit-learn version: {sklearn.__version__}")
print(f"NumPy version: {np.__version__}")
print(f"Matplotlib version: {matplotlib.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU device: {torch.cuda.get_device_name(0)}")
    print(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
END_PYTHON

if [ $? -ne 0 ]; then
    echo "Error: Failed to import required packages or CUDA not available"
    exit 1
fi

# Create timestamp for unique results directory
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
RESULTS_DIR=$SLURM_SUBMIT_DIR/output-improved-${TIMESTAMP}
mkdir -p $RESULTS_DIR

# Create a directory in scratch for the job
SCRATCH_DIR=$PFSDIR/nn_improved_training_${SLURM_JOB_ID}
if ! mkdir -p $SCRATCH_DIR; then
    echo "Failed to create scratch directory: $SCRATCH_DIR"
    exit 1
fi
echo "Created scratch directory: $SCRATCH_DIR"

# Check if required files exist
if [ ! -f nn_improved.py ]; then
    echo "Error: nn_improved.py not found in current directory"
    exit 1
fi

if [ ! -f combined.csv ]; then
    echo "Error: combined.csv not found in current directory"
    echo "Please ensure the combined.csv file is in the same directory as this job script"
    exit 1
fi

# Copy the script and data to the scratch directory
cp nn_improved.py $SCRATCH_DIR/
cp combined.csv $SCRATCH_DIR/
echo "Copied improved Python script and data to scratch directory"

# Change to the scratch directory
cd $SCRATCH_DIR
echo "Changed to scratch directory"

# Print data file information
echo "Data file information:"
echo "CSV file size: $(ls -lh combined.csv | awk '{print $5}')"
echo "CSV file rows: $(wc -l < combined.csv)"

# Run the improved training script
echo ""
echo "🔥 STARTING IMPROVED NEURAL NETWORK TRAINING"
echo "============================================="
echo "🏗️  Architecture: Enhanced with skip connections"
echo "🎯  Optimization: MAPE-focused loss function"
echo "📈  Features: Polynomial features + RobustScaler"
echo "⚡  Expected: Significantly better MAPE results"
echo "⏱️  Training time: up to 16 hours"
echo "============================================="
echo ""

python nn_improved.py 2>&1 | tee training_output.log

# Check if the script executed successfully
if [ $? -eq 0 ]; then
    echo "✅ Improved Neural Network training completed successfully"
else
    echo "❌ Improved Neural Network training failed with exit code $?"
    cp training_output.log $RESULTS_DIR/
    exit 1
fi

# Copy all results to the timestamped results directory
echo "📁 Copying results to: $RESULTS_DIR"
cp -v training_output.log $RESULTS_DIR/
cp -v improved_training_results.csv $RESULTS_DIR/ 2>/dev/null || echo "improved_training_results.csv not found"
cp -v best_improved_model.pth $RESULTS_DIR/ 2>/dev/null || echo "best_improved_model.pth not found"
cp -v *.png $RESULTS_DIR/ 2>/dev/null || echo "No PNG files found"

# Create a comprehensive summary report
echo "Creating enhanced summary report..."
cat > $RESULTS_DIR/improved_job_summary.txt << EOF
🚀 IMPROVED CNN EXECUTION TIME PREDICTION - JOB SUMMARY
======================================================
Job ID: $SLURM_JOB_ID
Start Time: $(date)
Submit Directory: $SLURM_SUBMIT_DIR
Results Directory: $RESULTS_DIR

🏗️ ENHANCED ARCHITECTURE DETAILS:
- Model: Advanced Neural Network with Skip Connections
- Skip Connections: 3 residual blocks for better gradient flow
- Activation: SiLU (Swish) activation function
- Regularization: Batch normalization + Layer normalization
- Optimizer: AdamW with weight decay
- Learning Rate: OneCycleLR scheduling
- Training: 300 epochs with early stopping (patience=50)

🎯 MAPE OPTIMIZATION FEATURES:
- Combined Loss: 70% MSE + 30% MAPE loss
- Early Stopping: Based on validation MAPE (not loss)
- Feature Engineering: Polynomial features for non-linear interactions
- Robust Scaling: RobustScaler for better outlier handling
- Smaller Batch Size: 32 for better gradient estimates

📊 EXPECTED IMPROVEMENTS:
- Target MAPE: < 40% (vs current ~51%)
- Better R² scores through skip connections
- Reduced overfitting with advanced regularization
- Improved generalization with polynomial features

📁 FILES GENERATED:
- training_output.log: Complete training log
- improved_training_results.csv: Enhanced cross-validation results
- best_improved_model.pth: Best trained model
- improved_job_summary.txt: This comprehensive summary

🖥️ GPU INFORMATION:
$(nvidia-smi --query-gpu=name,memory.total,memory.used --format=csv,noheader,nounits)

Training completed successfully!
EOF

# Extract and display key metrics from training log
echo ""
echo "📊 TRAINING RESULTS SUMMARY"
echo "==========================="
if [ -f training_output.log ]; then
    echo "Key Performance Metrics:"
    echo "------------------------"
    # Extract final MAPE results
    grep -E "(Average Val MAPE|Best performing fold)" training_output.log | tail -5
    echo ""
    echo "Training Progress:"
    echo "------------------"
    # Show training progress
    grep -E "Epoch.*Val MAPE.*Val R²" training_output.log | tail -10
fi

# Print final GPU memory usage
echo ""
echo "Final GPU memory usage:"
nvidia-smi

# Cleanup scratch directory
if [ -d "$SCRATCH_DIR" ]; then
    rm -rf $SCRATCH_DIR
    echo "Cleaned up scratch directory"
fi

# Deactivate virtual environment
deactivate

echo ""
echo "🎉 IMPROVED MODEL TRAINING COMPLETED! 🎉"
echo "========================================"
echo "📊 Results location: $RESULTS_DIR"
echo "📅 Job finished: $(date)"
echo ""
echo "📋 TO VIEW RESULTS:"
echo "  Training log: cat $RESULTS_DIR/training_output.log"
echo "  Results CSV:  cat $RESULTS_DIR/improved_training_results.csv"
echo "  Job summary:  cat $RESULTS_DIR/improved_job_summary.txt"
echo ""
echo "🚀 EXPECTED IMPROVEMENTS:"
echo "  ✅ Lower MAPE (target: < 40%)"
echo "  ✅ Better R² scores"
echo "  ✅ Improved generalization"
echo "  ✅ Enhanced stability"
echo ""
echo "📈 Compare with previous results to validate improvements!" 