#!/bin/bash
#SBATCH -J poly_elasticnet_degree3     # Job name
#SBATCH -o poly_elasticnet_degree3.out # Output file
#SBATCH --time=1:30:00                 # 90 minutes walltime (longer due to more hyperparams)
#SBATCH -N 1                           # 1 Node
#SBATCH -A sxk1942                     # Account/Project ID
#SBATCH -c 8                           # 8 processor cores
#SBATCH --mem=16GB                     # 16GB memory

set -e

echo "🔢 ELASTICNET POLYNOMIAL REGRESSION (DEGREE 3) - SLURM JOB"
echo "=========================================================="
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
    cp combined.csv polynomial_elasticnet.py "$WORK_DIR/"
    cd "$WORK_DIR"
fi

echo "🚀 Starting ElasticNet Polynomial Regression (Degree 3) Training..."
echo "Expected runtime: 15-90 minutes"
echo "Model type: ElasticNet Regression (L1 + L2 regularization)"
echo ""

python polynomial_elasticnet.py

if [ $? -eq 0 ]; then
    echo "✅ ElasticNet Polynomial Regression (Degree 3) training completed!"
    if [ "$WORK_DIR" != "." ]; then
        cp *.joblib *.csv *.png "$SLURM_SUBMIT_DIR/" 2>/dev/null
    fi
    echo "📋 NEXT STEPS: Submit comparison job: sbatch job_compare.sh"
else
    echo "❌ Training failed!"
    exit 1
fi

echo "End Time: $(date)" 