#!/bin/bash

# Polynomial Regression Models - Master Job Runner
# This script provides guidance for running all polynomial regression models

echo "🎯 Polynomial Regression Models for CNN Execution Time Prediction"
echo "================================================================="
echo ""
echo "This script guides you through training 4 different polynomial regression models:"
echo "  1. Linear Regression (fastest, baseline)"
echo "  2. Ridge Regression (L2 regularization)"
echo "  3. Lasso Regression (L1 regularization, feature selection)"
echo "  4. ElasticNet Regression (L1+L2 regularization, most robust)"
echo ""

# Check if required files exist
echo "🔍 Checking required files..."
MISSING_FILES=0

if [ ! -f "combined.csv" ]; then
    echo "❌ Missing: combined.csv"
    MISSING_FILES=$((MISSING_FILES + 1))
else
    echo "✅ Found: combined.csv"
fi

SCRIPTS=("polynomial_linear.py" "polynomial_ridge.py" "polynomial_lasso.py" "polynomial_elasticnet.py" "compare_models.py")
JOB_SCRIPTS=("job_linear.sh" "job_ridge.sh" "job_lasso.sh" "job_elasticnet.sh" "job_compare.sh")

for script in "${SCRIPTS[@]}"; do
    if [ ! -f "$script" ]; then
        echo "❌ Missing: $script"
        MISSING_FILES=$((MISSING_FILES + 1))
    else
        echo "✅ Found: $script"
    fi
done

for job in "${JOB_SCRIPTS[@]}"; do
    if [ ! -f "$job" ]; then
        echo "❌ Missing: $job"
        MISSING_FILES=$((MISSING_FILES + 1))
    else
        echo "✅ Found: $job"
    fi
done

if [ $MISSING_FILES -gt 0 ]; then
    echo ""
    echo "❌ ERROR: $MISSING_FILES required files are missing!"
    echo "Please ensure all files are in the current directory."
    exit 1
fi

echo ""
echo "✅ All required files found!"
echo ""

# Show job submission commands
echo "🚀 JOB SUBMISSION WORKFLOW"
echo "=========================="
echo ""
echo "Run these commands in order (wait for each to complete):"
echo ""

echo "1️⃣  Submit Linear Regression (fastest, ~30 min):"
echo "    sbatch job_linear.sh"
echo "    Expected time: 5-30 minutes"
echo "    Resources: 4 CPUs, 8GB RAM"
echo ""

echo "2️⃣  Submit Ridge Regression (L2 regularization, ~1 hour):"
echo "    sbatch job_ridge.sh"
echo "    Expected time: 10-60 minutes"
echo "    Resources: 6 CPUs, 12GB RAM"
echo ""

echo "3️⃣  Submit Lasso Regression (L1 regularization, ~1 hour):"
echo "    sbatch job_lasso.sh"
echo "    Expected time: 10-60 minutes"
echo "    Resources: 6 CPUs, 12GB RAM"
echo ""

echo "4️⃣  Submit ElasticNet Regression (L1+L2, ~1.5 hours):"
echo "    sbatch job_elasticnet.sh"
echo "    Expected time: 15-90 minutes"
echo "    Resources: 8 CPUs, 16GB RAM"
echo ""

echo "5️⃣  Compare All Models (~15 min):"
echo "    sbatch job_compare.sh"
echo "    Expected time: 1-5 minutes"
echo "    Resources: 2 CPUs, 4GB RAM"
echo ""

echo "📊 MONITORING JOBS"
echo "=================="
echo ""
echo "Check job status:"
echo "  squeue -u \$USER"
echo ""
echo "View job output (replace XXXX with job ID):"
echo "  tail -f poly_linear_training.out"
echo "  tail -f poly_ridge_training.out"
echo "  tail -f poly_lasso_training.out"
echo "  tail -f poly_elasticnet_training.out"
echo "  tail -f poly_compare_models.out"
echo ""

echo "⚡ QUICK START (if you want to run them all at once):"
echo "===================================================="
echo ""
echo "Submit all jobs (they will queue automatically):"
echo "  sbatch job_linear.sh && \\"
echo "  sbatch job_ridge.sh && \\"
echo "  sbatch job_lasso.sh && \\"
echo "  sbatch job_elasticnet.sh"
echo ""
echo "Wait for all to complete, then:"
echo "  sbatch job_compare.sh"
echo ""

echo "📈 EXPECTED RESULTS"
echo "=================="
echo ""
echo "Each model job will create a results directory containing:"
echo "  • {model}_training_results.csv - Detailed metrics"
echo "  • best_{model}_model.joblib - Trained model"
echo "  • {model}_polynomial_results.png - Visualization"
echo "  • training_output.log - Training log"
echo ""
echo "The comparison job will create:"
echo "  • polynomial_models_comparison.csv - Model comparison"
echo "  • polynomial_models_comparison.png - Comparison visualization"
echo ""

echo "🎯 MODEL CHARACTERISTICS"
echo "========================"
echo ""
echo "Linear:     Fast baseline, no regularization"
echo "Ridge:      L2 regularization, good for correlated features"
echo "Lasso:      L1 regularization, automatic feature selection"
echo "ElasticNet: L1+L2 combined, most robust and flexible"
echo ""

echo "💡 TIPS"
echo "======="
echo ""
echo "• Start with Linear to test your setup quickly"
echo "• Ridge usually performs well for CNN prediction tasks"
echo "• Use Lasso if you suspect many irrelevant features"
echo "• ElasticNet often gives the most robust results"
echo "• Always run the comparison to see which model is best"
echo ""

echo "🔧 TROUBLESHOOTING"
echo "=================="
echo ""
echo "If jobs are taking too long:"
echo "  • Check the README_POLYNOMIAL.md for optimization tips"
echo "  • Consider reducing hyperparameter grids"
echo "  • Monitor resource usage with 'seff <job_id>'"
echo ""

echo "If you get memory errors:"
echo "  • Check your dataset size in combined.csv"
echo "  • Consider sampling your data if it's very large"
echo ""

echo "If jobs fail:"
echo "  • Check the .out files for error messages"
echo "  • Ensure your virtual environment (ai3_env) is set up"
echo "  • Verify all required packages are installed"
echo ""

echo "🎉 READY TO START!"
echo "=================="
echo ""
echo "Choose your approach:"
echo ""
echo "Option A - Conservative (recommended):"
echo "  Run one job at a time, wait for completion, then run the next"
echo ""
echo "Option B - All at once:"
echo "  Submit all jobs simultaneously (they'll queue)"
echo ""
echo "Start with: sbatch job_linear.sh"
echo ""
echo "Good luck with your polynomial regression training! 🚀" 