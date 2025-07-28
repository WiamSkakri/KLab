# Polynomial Regression Models for CNN Execution Time Prediction

This directory contains separate scripts for training different polynomial regression models with degree 2 polynomials only (optimized for faster execution).

## 📁 Available Models

### Individual Model Scripts
- **`polynomial_linear.py`** - Linear Regression (no regularization)
- **`polynomial_ridge.py`** - Ridge Regression (L2 regularization)
- **`polynomial_lasso.py`** - Lasso Regression (L1 regularization)
- **`polynomial_elasticnet.py`** - ElasticNet Regression (L1 + L2 regularization)

### Comparison Script
- **`compare_models.py`** - Compare results from all trained models

## 🚀 Quick Start

### 1. Run Individual Models
Run each model separately to avoid walltime issues:

```bash
# Run each model individually (safer approach)
python polynomial_linear.py      # ~5-15 minutes
python polynomial_ridge.py       # ~10-30 minutes  
python polynomial_lasso.py       # ~10-30 minutes
python polynomial_elasticnet.py  # ~15-45 minutes
```

### 2. Compare Results
After running the individual models:

```bash
python compare_models.py
```

## 📊 What Each Model Does

| Model | Regularization | Feature Selection | Best For |
|-------|---------------|-------------------|----------|
| **Linear** | None | No | Baseline, fastest training |
| **Ridge** | L2 (shrinks coefficients) | No | Correlated features, prevents overfitting |
| **Lasso** | L1 (can zero coefficients) | Yes | Automatic feature selection |
| **ElasticNet** | L1 + L2 combined | Yes | Best of both worlds |

## 📈 Outputs

Each model script generates:
- **`{model}_training_results.csv`** - Detailed fold-by-fold results
- **`best_{model}_model.joblib`** - Best trained model
- **`{model}_polynomial_results.png`** - Visualization
- **`{model}_model_fold_{1-5}.joblib`** - Individual fold models

The comparison script generates:
- **`polynomial_models_comparison.csv`** - Summary comparison
- **`polynomial_models_comparison.png`** - Comparison visualization

## ⚙️ Model Configuration

All models use:
- **Polynomial Degree**: 2 only (for faster training)
- **Cross-Validation**: 5-fold
- **Hyperparameter Search**: Grid search with 3-fold CV
- **Polynomial Features**: No bias term

### Hyperparameter Grids

| Model | Parameters |
|-------|------------|
| Linear | degree=[2] |
| Ridge | degree=[2], alpha=[0.01, 0.1, 1.0, 10.0, 100.0] |
| Lasso | degree=[2], alpha=[0.01, 0.1, 1.0, 10.0, 100.0] |
| ElasticNet | degree=[2], alpha=[0.01, 0.1, 1.0, 10.0], l1_ratio=[0.1, 0.3, 0.5, 0.7, 0.9] |

## 📋 Prerequisites

Ensure you have `combined.csv` in the same directory. The scripts expect these columns:
- Numerical: `Batch_Size`, `Input_Size`, `In_Channels`, `Out_Channels`, `Kernel_Size`, `Stride`, `Padding`
- Categorical: `Algorithm` (will be one-hot encoded)
- Target: `Execution_Time_ms`

## 🕐 Expected Runtime

Based on dataset size, approximate runtimes:

| Model | Small Dataset (<1K samples) | Medium Dataset (1K-10K) | Large Dataset (>10K) |
|-------|---------------------------|------------------------|---------------------|
| Linear | 1-3 minutes | 5-15 minutes | 15-45 minutes |
| Ridge | 2-5 minutes | 10-30 minutes | 30-90 minutes |
| Lasso | 2-5 minutes | 10-30 minutes | 30-90 minutes |
| ElasticNet | 3-8 minutes | 15-45 minutes | 45-120 minutes |

## 🎯 Recommendations

1. **Start with Linear** - Fast baseline to test your setup
2. **Try Ridge next** - Usually performs well for CNN prediction
3. **Use Lasso** if you suspect many irrelevant features
4. **Try ElasticNet** for the most robust results

## 🔧 Troubleshooting

**If training is still too slow:**
1. Reduce alpha grid: `alpha=[0.1, 1.0, 10.0]`
2. Use fewer CV folds: `cv=3` instead of `cv=5`
3. Reduce hyperparameter combinations in the grid

**If you get memory errors:**
1. Check your dataset size
2. Consider sampling your data if it's very large
3. Use `n_jobs=1` instead of `n_jobs=-1` in GridSearchCV

## 📈 Understanding Results

**MAPE (Mean Absolute Percentage Error)**: Lower is better
- < 5%: Excellent
- 5-10%: Good  
- 10-20%: Acceptable
- > 20%: Poor

**R² (Coefficient of Determination)**: Higher is better
- > 0.9: Excellent
- 0.7-0.9: Good
- 0.5-0.7: Acceptable  
- < 0.5: Poor

## 🎉 Next Steps

After finding your best model:
1. Use `best_{model}_model.joblib` for predictions
2. Analyze feature importance (especially for Lasso/ElasticNet)
3. Consider ensemble methods if multiple models perform similarly
4. Validate on completely unseen test data 