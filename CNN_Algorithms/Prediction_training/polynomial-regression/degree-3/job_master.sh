#!/bin/bash
#SBATCH -J poly_master_degree3_controller  # Job name
#SBATCH -o poly_master_degree3_controller.out  # Output file
#SBATCH --time=0:10:00                     # 10 minutes (just for coordination)
#SBATCH -N 1                               # 1 Node
#SBATCH -A sxk1942                         # Account/Project ID
#SBATCH -c 1                               # 1 processor (minimal for coordination)
#SBATCH --mem=2GB                          # 2GB memory (minimal)

# Exit on any error
set -e

echo "🎯 POLYNOMIAL REGRESSION (DEGREE 3) MASTER CONTROLLER"
echo "====================================================="
echo "This job will automatically submit and coordinate all degree 3 polynomial regression training jobs"
echo ""
echo "Job submission started at: $(date)"
echo ""

# Check if all required files exist
echo "🔍 Checking required files..."
MISSING_FILES=0

REQUIRED_FILES=("combined.csv" "polynomial_linear.py" "polynomial_ridge.py" "polynomial_lasso.py" "polynomial_elasticnet.py" "compare_models.py" "job_linear.sh" "job_ridge.sh" "job_lasso.sh" "job_elasticnet.sh" "job_compare.sh")

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
    echo "❌ ERROR: $MISSING_FILES required files are missing!"
    echo "Please ensure all files are in the current directory."
    exit 1
fi

echo ""
echo "✅ All required files found!"
echo ""

# Submit jobs with dependencies
echo "🚀 SUBMITTING POLYNOMIAL REGRESSION (DEGREE 3) JOBS IN SEQUENCE"
echo "================================================================"
echo ""

echo "1️⃣ Submitting Linear Regression (Degree 3) job..."
LINEAR_JOB=$(sbatch --parsable job_linear.sh)
echo "   ✅ Linear job submitted: $LINEAR_JOB"
echo "   Expected time: 5-30 minutes"
echo ""

echo "2️⃣ Submitting Ridge Regression (Degree 3) job (depends on Linear)..."
RIDGE_JOB=$(sbatch --parsable --dependency=afterok:$LINEAR_JOB job_ridge.sh)
echo "   ✅ Ridge job submitted: $RIDGE_JOB"
echo "   Will start after Linear completes"
echo "   Expected time: 10-60 minutes"
echo ""

echo "3️⃣ Submitting Lasso Regression (Degree 3) job (depends on Ridge)..."
LASSO_JOB=$(sbatch --parsable --dependency=afterok:$RIDGE_JOB job_lasso.sh)
echo "   ✅ Lasso job submitted: $LASSO_JOB"
echo "   Will start after Ridge completes"
echo "   Expected time: 10-60 minutes"
echo ""

echo "4️⃣ Submitting ElasticNet Regression (Degree 3) job (depends on Lasso)..."
ELASTICNET_JOB=$(sbatch --parsable --dependency=afterok:$LASSO_JOB job_elasticnet.sh)
echo "   ✅ ElasticNet job submitted: $ELASTICNET_JOB"
echo "   Will start after Lasso completes"
echo "   Expected time: 15-90 minutes"
echo ""

echo "5️⃣ Submitting Model Comparison (Degree 3) job (depends on all models)..."
COMPARE_JOB=$(sbatch --parsable --dependency=afterok:$ELASTICNET_JOB job_compare.sh)
echo "   ✅ Comparison job submitted: $COMPARE_JOB"
echo "   Will start after ElasticNet completes"
echo "   Expected time: 1-5 minutes"
echo ""

echo "🎉 ALL DEGREE 3 POLYNOMIAL REGRESSION JOBS SUBMITTED SUCCESSFULLY!"
echo "=================================================================="
echo ""
echo "📊 Job Schedule:"
echo "  Linear:     $LINEAR_JOB (starts immediately)"
echo "  Ridge:      $RIDGE_JOB (starts after $LINEAR_JOB)"
echo "  Lasso:      $LASSO_JOB (starts after $RIDGE_JOB)"
echo "  ElasticNet: $ELASTICNET_JOB (starts after $LASSO_JOB)"
echo "  Comparison: $COMPARE_JOB (starts after $ELASTICNET_JOB)"
echo ""

echo "⏱️ Total Expected Runtime:"
echo "  Linear:     5-30 minutes"
echo "  Ridge:      10-60 minutes"
echo "  Lasso:      10-60 minutes"
echo "  ElasticNet: 15-90 minutes"
echo "  Comparison: 1-5 minutes"
echo "  ────────────────────────"
echo "  TOTAL:      41-245 minutes (~1-4 hours)"
echo ""

echo "📋 MONITORING COMMANDS:"
echo "======================"
echo ""
echo "Check all job status:"
echo "  squeue -u \$USER"
echo ""
echo "Watch job progress:"
echo "  watch -n 30 'squeue -u \$USER'"
echo ""
echo "View individual job outputs:"
echo "  tail -f poly_linear_degree3.out"
echo "  tail -f poly_ridge_degree3.out"
echo "  tail -f poly_lasso_degree3.out"
echo "  tail -f poly_elasticnet_degree3.out"
echo "  tail -f poly_compare_degree3.out"
echo ""

echo "🔄 JOB DEPENDENCIES:"
echo "==================="
echo "Linear → Ridge → Lasso → ElasticNet → Comparison"
echo ""
echo "Each job will automatically start when the previous one completes successfully."
echo "If any job fails, the subsequent jobs will be cancelled."
echo ""

echo "📈 WHAT TO EXPECT:"
echo "=================="
echo ""
echo "🟡 Phase 1: Linear Regression (Degree 3) - fastest baseline"
echo "🟡 Phase 2: Ridge Regression (Degree 3) - L2 regularization"
echo "🟡 Phase 3: Lasso Regression (Degree 3) - L1 regularization + feature selection"
echo "🟡 Phase 4: ElasticNet Regression (Degree 3) - L1+L2 combined"
echo "🟢 Phase 5: Model Comparison (Degree 3) - final analysis and ranking"
echo ""

echo "📁 RESULTS LOCATION:"
echo "==================="
echo "Each job will create result files in the current directory:"
echo "  • *_degree3_training_results.csv - Individual model results"
echo "  • *_degree3_polynomial_results.png - Individual model visualizations"
echo "  • best_*_degree3_model.joblib - Trained models"
echo "  • polynomial_degree3_models_comparison.* - Final comparison"
echo ""

echo "🎯 SUCCESS CRITERIA:"
echo "==================="
echo "✅ All 5 jobs complete successfully"
echo "✅ Each job generates its result files"
echo "✅ Comparison job produces final model ranking"
echo "✅ Best degree 3 polynomial regression model identified"
echo ""

echo "🚨 TROUBLESHOOTING:"
echo "==================="
echo "If any job fails:"
echo "  1. Check the .out file for error messages"
echo "  2. Use 'scontrol show job <job_id>' for details"
echo "  3. Check resource usage with 'seff <job_id>'"
echo "  4. Subsequent jobs will be automatically cancelled"
echo ""

echo "Master controller completed at: $(date)"
echo ""
echo "🎉 Your degree 3 polynomial regression training pipeline is now running!"
echo "   Just monitor the jobs and wait for all to complete."
echo ""
echo "💡 TIP: The entire pipeline will run unattended. You can log out"
echo "       and check back later. Results will be waiting for you!"
echo ""
echo "🔬 RESEARCH GOAL: Find the best degree 3 polynomial model for"
echo "                  CNN execution time prediction with cubic relationships!" 