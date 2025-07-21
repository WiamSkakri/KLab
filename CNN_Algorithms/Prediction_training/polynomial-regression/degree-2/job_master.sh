#!/bin/bash
#SBATCH -J poly_master_controller      # Job name
#SBATCH -o poly_master_controller.out  # Output file
#SBATCH --time=24:00:00                 # 24 hours (just for coordination)
#SBATCH -N 1                           # 1 Node
#SBATCH -A sxk1942                     # Account/Project ID
#SBATCH -c 1                           # 1 processor (minimal for coordination)
#SBATCH --mem=2GB                      # 2GB memory (minimal)

# Exit on any error
set -e

echo "🎯 POLYNOMIAL REGRESSION MASTER CONTROLLER"
echo "==========================================="
echo "This job will automatically submit and coordinate all polynomial regression training jobs"
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
echo "🚀 SUBMITTING POLYNOMIAL REGRESSION JOBS IN SEQUENCE"
echo "====================================================="
echo ""

echo "1️⃣ Submitting Linear Regression job..."
LINEAR_JOB=$(sbatch --parsable job_linear.sh)
echo "   ✅ Linear job submitted: $LINEAR_JOB"
echo "   Expected time: 5-30 minutes"
echo ""

echo "2️⃣ Submitting Ridge Regression job (depends on Linear)..."
RIDGE_JOB=$(sbatch --parsable --dependency=afterok:$LINEAR_JOB job_ridge.sh)
echo "   ✅ Ridge job submitted: $RIDGE_JOB"
echo "   Will start after Linear completes"
echo "   Expected time: 10-60 minutes"
echo ""

echo "3️⃣ Submitting Lasso Regression job (depends on Ridge)..."
LASSO_JOB=$(sbatch --parsable --dependency=afterok:$RIDGE_JOB job_lasso.sh)
echo "   ✅ Lasso job submitted: $LASSO_JOB"
echo "   Will start after Ridge completes"
echo "   Expected time: 10-60 minutes"
echo ""

echo "4️⃣ Submitting ElasticNet Regression job (depends on Lasso)..."
ELASTICNET_JOB=$(sbatch --parsable --dependency=afterok:$LASSO_JOB job_elasticnet.sh)
echo "   ✅ ElasticNet job submitted: $ELASTICNET_JOB"
echo "   Will start after Lasso completes"
echo "   Expected time: 15-90 minutes"
echo ""

echo "5️⃣ Submitting Model Comparison job (depends on all models)..."
COMPARE_JOB=$(sbatch --parsable --dependency=afterok:$ELASTICNET_JOB job_compare.sh)
echo "   ✅ Comparison job submitted: $COMPARE_JOB"
echo "   Will start after ElasticNet completes"
echo "   Expected time: 1-5 minutes"
echo ""

echo "🎉 ALL JOBS SUBMITTED SUCCESSFULLY!"
echo "==================================="
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
echo "  tail -f poly_linear_training.out"
echo "  tail -f poly_ridge_training.out"
echo "  tail -f poly_lasso_training.out"
echo "  tail -f poly_elasticnet_training.out"
echo "  tail -f poly_compare_models.out"
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
echo "🟡 Phase 1: Linear Regression (fastest baseline)"
echo "🟡 Phase 2: Ridge Regression (L2 regularization)"
echo "🟡 Phase 3: Lasso Regression (L1 regularization + feature selection)"
echo "🟡 Phase 4: ElasticNet Regression (L1+L2 combined)"
echo "🟢 Phase 5: Model Comparison (final analysis)"
echo ""

echo "📁 RESULTS LOCATION:"
echo "==================="
echo "Each job will create its own timestamped results directory:"
echo "  • results_linear_YYYYMMDD_HHMMSS/"
echo "  • results_ridge_YYYYMMDD_HHMMSS/"
echo "  • results_lasso_YYYYMMDD_HHMMSS/"
echo "  • results_elasticnet_YYYYMMDD_HHMMSS/"
echo "  • results_comparison_YYYYMMDD_HHMMSS/"
echo ""

echo "🎯 SUCCESS CRITERIA:"
echo "==================="
echo "✅ All 5 jobs complete successfully"
echo "✅ Each job generates its results directory"
echo "✅ Comparison job produces model ranking"
echo "✅ Best polynomial regression model identified"
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
echo "🎉 Your polynomial regression training pipeline is now running!"
echo "   Just monitor the jobs and wait for all to complete."
echo ""
echo "💡 TIP: The entire pipeline will run unattended. You can log out"
echo "       and check back later. Results will be waiting for you!" 