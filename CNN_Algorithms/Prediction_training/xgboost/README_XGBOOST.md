# XGBoost CNN Execution Time Prediction

This directory contains the XGBoost implementation for predicting CNN algorithm execution times. The model follows the same comprehensive structure as other prediction models in this project.

## Overview

The XGBoost model uses gradient boosting to predict execution times for different CNN algorithms based on various input parameters. It supports both GPU and CPU acceleration and includes comprehensive hyperparameter tuning.

## Features

- **GPU Acceleration**: Automatic detection and use of GPU when available (`gpu_hist` tree method)
- **CPU Fallback**: Graceful fallback to CPU-based training when GPU is not available
- **K-Fold Cross Validation**: 5-fold cross validation for robust model evaluation
- **Hyperparameter Tuning**: Comprehensive grid search across multiple parameters
- **Comprehensive Metrics**: MAPE, MAE, RMSE, and R² evaluation
- **Advanced Visualizations**: Multiple detailed plots and analysis dashboards
- **Model Persistence**: Automatic saving of trained models and results

## Files Structure

```
xgboost/
├── python.py              # Main XGBoost training script
├── job.sh                  # SLURM job submission script
├── README_XGBOOST.md      # This documentation
└── output/                 # Generated results (created during execution)
    ├── xgb_training_results.csv
    ├── xgb_summary_metrics.csv
    ├── best_xgb_model.joblib
    ├── xgb_evaluation_dashboard.png
    ├── xgb_hyperparameter_analysis.png
    └── xgb_detailed_metrics_analysis.png
```

## Requirements

### Python Packages
- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computing
- `scikit-learn` - Machine learning utilities and metrics
- `xgboost` - XGBoost gradient boosting framework
- `matplotlib` - Plotting and visualization
- `joblib` - Model serialization

### Hardware Requirements
- **CPU**: Multi-core processor (8+ cores recommended)
- **Memory**: 32GB+ RAM recommended for large datasets
- **GPU** (Optional): NVIDIA GPU with CUDA support for acceleration
- **Storage**: Sufficient space for model files and visualizations

## Usage

### Running on HPC (Recommended)

1. **Submit the job**:
   ```bash
   sbatch job.sh
   ```

2. **Monitor progress**:
   ```bash
   squeue -u $USER
   tail -f xgb_training_output.log
   ```

### Running Locally

1. **Ensure data file exists**:
   ```bash
   # combined.csv should be in the current directory or parent directories
   ls combined.csv
   ```

2. **Install dependencies**:
   ```bash
   pip install pandas numpy scikit-learn xgboost matplotlib joblib
   ```

3. **Run the training**:
   ```bash
   python python.py
   ```

## Model Configuration

### Base Parameters
- **Objective**: `reg:squarederror` (regression with squared error)
- **Eval Metric**: `mae` (Mean Absolute Error)
- **Tree Method**: `gpu_hist` (GPU) or `hist` (CPU)
- **Random State**: 42 (for reproducibility)

### Hyperparameter Grid
The model searches through the following hyperparameter space:

| Parameter | Values |
|-----------|--------|
| `n_estimators` | [100, 200, 300, 500] |
| `max_depth` | [3, 6, 10, 15] |
| `learning_rate` | [0.01, 0.1, 0.2, 0.3] |
| `subsample` | [0.8, 0.9, 1.0] |
| `colsample_bytree` | [0.8, 0.9, 1.0] |
| `reg_alpha` | [0, 0.1, 1] |
| `reg_lambda` | [1, 1.5, 2] |

**Total combinations**: 1,296 parameter combinations

### Cross-Validation
- **Method**: K-Fold Cross Validation
- **Folds**: 5
- **Shuffle**: True (random_state=42)
- **Scoring**: Negative Mean Absolute Error

## Output Files

### CSV Results
1. **`xgb_training_results.csv`**: Detailed results for each fold
   - Fold number, training/validation metrics
   - Best hyperparameters for each fold
   - Training times

2. **`xgb_summary_metrics.csv`**: Overall summary statistics
   - Average performance across folds
   - Standard deviations
   - Best fold information

### Model Files
- **`best_xgb_model.joblib`**: Best performing model (lowest validation MAPE)
- **`xgb_model_fold_X.joblib`**: Individual models for each fold

### Visualizations
1. **`xgb_evaluation_dashboard.png`**: Main evaluation dashboard (3x3 grid)
   - Predictions vs Actual scatter plot
   - Residual analysis
   - Cross-validation performance
   - Error distribution
   - Metrics comparison
   - Training time analysis
   - Feature importance (top 10)

2. **`xgb_hyperparameter_analysis.png`**: Hyperparameter analysis (2x3 grid)
   - Parameter vs performance plots
   - Parameter space exploration
   - Best parameters summary

3. **`xgb_detailed_metrics_analysis.png`**: Detailed metrics analysis (2x3 grid)
   - Train vs validation comparisons
   - Metrics distribution
   - Performance vs training time
   - Statistical summary

## Performance Metrics

The model evaluates performance using multiple metrics:

- **MAPE (Mean Absolute Percentage Error)**: Primary evaluation metric (lower is better)
- **MAE (Mean Absolute Error)**: Absolute prediction error
- **RMSE (Root Mean Square Error)**: Square root of mean squared error
- **R² (Coefficient of Determination)**: Proportion of variance explained

## GPU Acceleration

### Enabling GPU Support
1. **Install XGBoost with GPU support**:
   ```bash
   pip install xgboost
   ```

2. **Verify GPU availability**:
   ```python
   import xgboost as xgb
   print(xgb.device.Device().device_type)  # Should show 'cuda'
   ```

### GPU vs CPU Performance
- **Training Speed**: GPU typically 3-10x faster than CPU
- **Memory Usage**: GPU requires sufficient VRAM
- **Compatibility**: Automatically falls back to CPU if GPU unavailable

## Troubleshooting

### Common Issues

1. **GPU Not Detected**:
   ```
   ⚠️ XGBoost GPU not available: [error message]
   ```
   - Check CUDA installation
   - Verify GPU drivers
   - Install XGBoost with GPU support

2. **Memory Issues**:
   ```
   CUDA out of memory
   ```
   - Reduce batch size in hyperparameter grid
   - Use smaller parameter grid
   - Switch to CPU training

3. **Data File Missing**:
   ```
   Error: combined.csv not found!
   ```
   - Ensure `combined.csv` exists in current or parent directory
   - Check file permissions

### Performance Optimization

1. **For Large Datasets**:
   - Use GPU acceleration
   - Reduce hyperparameter grid size
   - Increase memory allocation

2. **For Quick Testing**:
   - Reduce cross-validation folds
   - Use smaller parameter grid
   - Limit n_estimators range

## Model Interpretation

### Feature Importance
The model automatically generates feature importance plots showing:
- Top 10 most important features
- Relative importance scores
- Feature ranking

### Hyperparameter Insights
The analysis provides:
- Best parameter combinations across folds
- Parameter stability analysis
- Performance vs parameter relationships

## Integration with Other Models

This XGBoost model complements other prediction models in the project:
- **Neural Networks** (`../nn/`): Deep learning approach
- **Random Forest** (`../rdf/`): Ensemble tree method
- **SVR** (`../svr/`): Support Vector Regression
- **Polynomial Regression** (`../polynomial-regression/`): Linear approaches

## Future Enhancements

Potential improvements:
1. **Advanced Hyperparameter Tuning**: Bayesian optimization
2. **Feature Engineering**: Automated feature creation
3. **Model Ensembling**: Combine with other algorithms
4. **Online Learning**: Incremental model updates
5. **Multi-objective Optimization**: Balance accuracy vs speed

## References

- [XGBoost Documentation](https://xgboost.readthedocs.io/)
- [GPU Acceleration Guide](https://xgboost.readthedocs.io/en/latest/gpu/index.html)
- [Hyperparameter Tuning Best Practices](https://xgboost.readthedocs.io/en/latest/tutorials/param_tuning.html) 