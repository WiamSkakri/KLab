#!/bin/bash
#SBATCH -J poly_lasso_degree3          # Job name
#SBATCH -o poly_lasso_degree3.out      # Output file
#SBATCH --time=1:00:00                 # 60 minutes walltime
#SBATCH -N 1                           # 1 Node
#SBATCH -A sxk1942                     # Account/Project ID
#SBATCH -c 6                           # 6 processor cores
#SBATCH --mem=12GB                     # 12GB memory

set -e

echo "🔢 LASSO POLYNOMIAL REGRESSION (DEGREE 3) - SLURM JOB"
echo "===================================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo ""

if [ -d "/local_scratch/pbs.$SLURM_JOB_ID" ]; then
    WORK_DIR="/local_scratch/pbs.$SLURM_JOB_ID"
else
    WORK_DIR="."
fi

source ~/.bashrc
conda activate ai3_env

if [ "$WORK_DIR" != "." ]; then
    cp combined.csv polynomial_lasso.py "$WORK_DIR/"
    cd "$WORK_DIR"
fi

echo "🚀 Starting Lasso Polynomial Regression (Degree 3) Training..."
echo "Expected runtime: 10-60 minutes"
echo "Model type: Lasso Regression (L1 regularization + feature selection)"
echo ""

python polynomial_lasso.py

if [ $? -eq 0 ]; then
    echo "✅ Lasso Polynomial Regression (Degree 3) training completed!"
    if [ "$WORK_DIR" != "." ]; then
        cp *.joblib *.csv *.png "$SLURM_SUBMIT_DIR/" 2>/dev/null
    fi
    echo "📋 NEXT STEPS: Submit ElasticNet job: sbatch job_elasticnet.sh"
else
    echo "❌ Training failed!"
    exit 1
fi

echo "End Time: $(date)" 