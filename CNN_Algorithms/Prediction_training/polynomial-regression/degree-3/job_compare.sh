#!/bin/bash
#SBATCH -J poly_compare_degree3        # Job name
#SBATCH -o poly_compare_degree3.out    # Output file
#SBATCH --time=0:15:00                 # 15 minutes walltime
#SBATCH -N 1                           # 1 Node
#SBATCH -A sxk1942                     # Account/Project ID
#SBATCH -c 2                           # 2 processor cores
#SBATCH --mem=4GB                      # 4GB memory

set -e

echo "🔢 POLYNOMIAL REGRESSION (DEGREE 3) MODEL COMPARISON - SLURM JOB"
echo "================================================================"
echo "Job ID: $SLURM_JOB_ID"
echo "Start Time: $(date)"
echo ""

source ~/.bashrc
conda activate ai3_env

# Check if all required result files exist
echo "🔍 Checking for model result files..."
REQUIRED_FILES=("linear_degree3_training_results.csv" "ridge_degree3_training_results.csv" "lasso_degree3_training_results.csv" "elasticnet_degree3_training_results.csv")
MISSING_FILES=0

for file in "${REQUIRED_FILES[@]}"; do
    if [ ! -f "$file" ]; then
        echo "❌ Missing: $file"
        MISSING_FILES=$((MISSING_FILES + 1))
    else
        echo "✅ Found: $file"
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo ""
    echo "❌ ERROR: $MISSING_FILES result files are missing!"
    echo "Please ensure all individual model training jobs have completed successfully."
    exit 1
fi

echo ""
echo "✅ All model result files found!"
echo ""

echo "🚀 Starting Model Comparison Analysis..."
echo "Expected runtime: 1-5 minutes"
echo ""

python compare_models.py

if [ $? -eq 0 ]; then
    echo ""
    echo "✅ Model comparison completed successfully!"
    echo ""
    echo "📁 Generated comparison files:"
    ls -la *comparison*.csv *comparison*.png 2>/dev/null || echo "No comparison files found"
    echo ""
    echo "🎉 POLYNOMIAL REGRESSION (DEGREE 3) PIPELINE COMPLETE!"
    echo ""
    echo "📊 Check these files for your results:"
    echo "   • polynomial_degree3_models_comparison.png - Visual comparison"
    echo "   • polynomial_degree3_models_comparison.csv - Detailed metrics"
    echo ""
    echo "🏆 The best model ranking is now available!"
else
    echo "❌ Model comparison failed!"
    exit 1
fi

echo "End Time: $(date)" 