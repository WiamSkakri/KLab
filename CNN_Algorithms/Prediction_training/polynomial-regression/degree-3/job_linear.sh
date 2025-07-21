#!/bin/bash
#SBATCH -J poly_linear_degree3         # Job name
#SBATCH -o poly_linear_degree3.out     # Output file
#SBATCH --time=0:30:00                 # 30 minutes walltime
#SBATCH -N 1                           # 1 Node
#SBATCH -A sxk1942                     # Account/Project ID
#SBATCH -c 4                           # 4 processor cores
#SBATCH --mem=8GB                      # 8GB memory

# Exit on any error
set -e

echo "🔢 LINEAR POLYNOMIAL REGRESSION (DEGREE 3) - SLURM JOB"
echo "======================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Job Name: $SLURM_JOB_NAME"
echo "Start Time: $(date)"
echo "Node: $SLURMD_NODENAME"
echo "Cores: $SLURM_CPUS_PER_TASK"
echo "Memory: $SLURM_MEM_PER_NODE MB"
echo ""

# Check if we're on a compute node (has scratch)
if [ -d "/local_scratch/pbs.$SLURM_JOB_ID" ]; then
    echo "📁 Using compute node scratch directory"
    WORK_DIR="/local_scratch/pbs.$SLURM_JOB_ID"
else
    echo "📁 Using current directory (no scratch available)"
    WORK_DIR="."
fi

echo "Working directory: $WORK_DIR"
echo ""

# Load modules and activate environment
echo "🔧 Setting up environment..."
source ~/.bashrc

# Activate conda environment
echo "Activating ai3_env environment..."
conda activate ai3_env
echo "✅ Environment activated"
echo ""

# Copy files to working directory if using scratch
if [ "$WORK_DIR" != "." ]; then
    echo "📋 Copying files to compute node..."
    
    # Check if required files exist
    if [ ! -f "combined.csv" ]; then
        echo "❌ ERROR: combined.csv not found in current directory"
        echo "Current directory: $(pwd)"
        echo "Available files:"
        ls -la
        exit 1
    fi
    
    if [ ! -f "polynomial_linear.py" ]; then
        echo "❌ ERROR: polynomial_linear.py not found in current directory"
        exit 1
    fi
    
    # Copy required files
    cp combined.csv "$WORK_DIR/"
    cp polynomial_linear.py "$WORK_DIR/"
    
    echo "✅ Files copied to compute node"
    echo ""
    
    # Change to working directory
    cd "$WORK_DIR"
fi

echo "📂 Current working directory: $(pwd)"
echo "📋 Available files:"
ls -la
echo ""

# Verify required files
echo "🔍 Verifying required files..."
if [ ! -f "combined.csv" ]; then
    echo "❌ ERROR: combined.csv not found!"
    exit 1
fi

if [ ! -f "polynomial_linear.py" ]; then
    echo "❌ ERROR: polynomial_linear.py not found!"
    exit 1
fi

echo "✅ All required files present"
echo ""

# Show dataset info
echo "📊 Dataset information:"
head -n 5 combined.csv
echo "Dataset shape: $(wc -l combined.csv) rows"
echo ""

# Run the polynomial regression training
echo "🚀 Starting Linear Polynomial Regression (Degree 3) Training..."
echo "Expected runtime: 5-30 minutes"
echo "Model type: Linear Regression (no regularization)"
echo "Polynomial degree: 3"
echo ""

python polynomial_linear.py

# Check if training completed successfully
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Linear Polynomial Regression (Degree 3) training completed successfully!"
    
    # Show generated files
    echo ""
    echo "📁 Generated files:"
    ls -la *.joblib *.csv *.png 2>/dev/null || echo "No output files found"
    
else
    echo "❌ Training failed with exit code: $?"
    exit 1
fi

# Copy results back if using scratch
if [ "$WORK_DIR" != "." ]; then
    echo ""
    echo "📤 Copying results back to submission directory..."
    
    # Copy all result files back
    cp *.joblib "$SLURM_SUBMIT_DIR/" 2>/dev/null || echo "No .joblib files to copy"
    cp *.csv "$SLURM_SUBMIT_DIR/" 2>/dev/null || echo "No .csv files to copy"
    cp *.png "$SLURM_SUBMIT_DIR/" 2>/dev/null || echo "No .png files to copy"
    
    echo "✅ Results copied back to: $SLURM_SUBMIT_DIR"
fi

echo ""
echo "🎉 LINEAR POLYNOMIAL REGRESSION (DEGREE 3) JOB COMPLETED!"
echo "End Time: $(date)"
echo ""
echo "📋 NEXT STEPS:"
echo "1. Check the output files in your submission directory"
echo "2. Review the generated visualization: linear_degree3_polynomial_results.png"
echo "3. Check training results: linear_degree3_training_results.csv"
echo "4. Submit the next model job: sbatch job_ridge.sh"
echo ""
echo "💡 TIP: Use 'seff $SLURM_JOB_ID' to see resource usage statistics" 