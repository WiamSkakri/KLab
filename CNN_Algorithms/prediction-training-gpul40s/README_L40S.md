# L40s GPU Training Setup for CNN Algorithm Performance Prediction

This directory contains optimized training scripts for L40s GPU acceleration of machine learning models to predict CNN algorithm execution times.

## Overview

This setup includes:
- **Random Forest** training with cuML GPU acceleration
- **XGBoost** training with CUDA GPU acceleration
- Comprehensive evaluation and visualization
- HPC job scripts optimized for L40s GPU

## Directory Structure

```
prediction-training-gpul40s/
├── rdf/                     # Random Forest training
│   ├── python.py           # cuML-optimized Random Forest script
│   ├── job.sh              # HPC job submission script
│   └── combined_l40s.csv   # L40s experiment data (to be added)
├── xgboost/                # XGBoost training
│   ├── python.py           # CUDA-optimized XGBoost script
│   ├── job.sh              # HPC job submission script
│   └── combined_l40s.csv   # L40s experiment data (to be added)
└── README_L40S.md          # This file
```

## Prerequisites

### Environment Setup
1. **Virtual Environment**: Ensure `ai3_env` is configured with GPU libraries:
   ```bash
   # Install cuML for Random Forest GPU acceleration
   conda install -c rapidsai -c nvidia cuml
   
   # Install XGBoost with GPU support
   pip install xgboost[gpu]
   
   # Install other required packages
   pip install pandas numpy scikit-learn matplotlib joblib
   ```

2. **Data File**: Place your `combined_l40s.csv` file in both `rdf/` and `xgboost/` directories

### L40s GPU Requirements
- NVIDIA L40s GPU with 48GB VRAM
- CUDA-compatible drivers
- HPC environment with SLURM scheduler

## Usage

### Random Forest Training (cuML)

1. **Navigate to RDF directory**:
   ```bash
   cd CNN_Algorithms/prediction-training-gpul40s/rdf/
   ```

2. **Ensure data file is present**:
   ```bash
   ls -la combined_l40s.csv
   ```

3. **Submit HPC job**:
   ```bash
   sbatch job.sh
   ```

4. **Monitor job progress**:
   ```bash
   squeue -u $USER
   tail -f rdf_l40s_training.out
   ```

### XGBoost Training (CUDA)

1. **Navigate to XGBoost directory**:
   ```bash
   cd CNN_Algorithms/prediction-training-gpul40s/xgboost/
   ```

2. **Ensure data file is present**:
   ```bash
   ls -la combined_l40s.csv
   ```

3. **Submit HPC job**:
   ```bash
   sbatch job.sh
   ```

4. **Monitor job progress**:
   ```bash
   squeue -u $USER
   tail -f xgb_l40s_training.out
   ```

## Output Files

### Random Forest Outputs
- `rdf_l40s_training_results.csv` - Detailed fold-by-fold results
- `rdf_l40s_summary_metrics.csv` - Summary statistics
- `rdf_l40s_evaluation_dashboard.png` - Comprehensive visualization
- `best_rf_l40s_model.joblib` - Best performing model
- `rf_l40s_model_fold_*.joblib` - Individual fold models

### XGBoost Outputs
- `xgb_l40s_training_results.csv` - Detailed fold-by-fold results
- `xgb_l40s_summary_metrics.csv` - Summary statistics
- `xgb_l40s_evaluation_dashboard.png` - Comprehensive visualization
- `best_xgb_l40s_model.joblib` - Best performing model
- `xgb_l40s_model_fold_*.joblib` - Individual fold models

## Algorithm Features

### Random Forest (cuML)
- **GPU Acceleration**: cuML Random Forest with L40s optimization
- **Enhanced Parameters**: Higher n_estimators (200-600) for GPU
- **Memory Optimization**: Efficient GPU memory management
- **Parallel Streams**: Leverages L40s parallel processing capabilities

### XGBoost (CUDA)
- **GPU Acceleration**: Native CUDA support with `device='cuda:0'`
- **Tree Method**: Histogram-based algorithm optimized for L40s
- **Enhanced Parameter Space**: Extended hyperparameter grid for GPU
- **Memory Efficient**: Optimized for 48GB VRAM

## Performance Monitoring

### Job Resource Usage
- **CPU**: 8 cores per job
- **Memory**: 64GB system RAM
- **GPU**: 1 L40s GPU (48GB VRAM)
- **Time Limit**: 6 hours (RDF), 4 hours (XGBoost)

### Expected Performance
- **Random Forest**: Typically 5-20x speedup over CPU
- **XGBoost**: Typically 3-10x speedup over CPU
- **Memory Usage**: Efficient utilization of L40s 48GB VRAM

## Troubleshooting

### Common Issues

1. **GPU Not Detected**:
   ```bash
   nvidia-smi  # Check GPU availability
   echo $CUDA_VISIBLE_DEVICES  # Check CUDA device visibility
   ```

2. **Memory Issues**:
   - Reduce batch size or n_estimators if OOM errors occur
   - Monitor GPU memory with `nvidia-smi` during training

3. **cuML Installation**:
   ```bash
   # Ensure RAPIDS cuML is properly installed
   python -c "import cuml; print(cuml.__version__)"
   ```

4. **XGBoost GPU Support**:
   ```bash
   # Test XGBoost GPU functionality
   python -c "
   import xgboost as xgb
   import numpy as np
   data = xgb.DMatrix(np.random.rand(10, 5), label=np.random.rand(10))
   params = {'tree_method': 'hist', 'device': 'cuda:0'}
   xgb.train(params, data, num_boost_round=1)
   print('XGBoost GPU working!')
   "
   ```

### Data Requirements

Your `combined_l40s.csv` should contain:
- **Features**: CNN algorithm parameters (batch size, input dimensions, etc.)
- **Target**: `Execution_Time_ms` column
- **Algorithm**: Algorithm type column (will be one-hot encoded)

Example structure:
```csv
Input_Height,Input_Width,Batch_Size,Algorithm,Execution_Time_ms
224,224,32,GEMM,45.2
224,224,64,Winograd,38.7
...
```

## Advanced Configuration

### Environment Variables
- `CUDA_VISIBLE_DEVICES=0` - Use first GPU
- `RAPIDS_NO_INITIALIZE=1` - Optimize cuML initialization
- `CUML_LOG_LEVEL=INFO` - Enable cuML logging

### Hyperparameter Tuning
- **Random Forest**: Modify `param_grid_hpc` in `rdf/python.py`
- **XGBoost**: Modify `param_space` in `xgboost/python.py`

### Output Customization
- Adjust visualization settings in the dashboard functions
- Modify CSV output columns as needed
- Configure logging levels and output detail

## Performance Comparison

Compare results between:
- L40s GPU vs CPU training times
- Random Forest vs XGBoost accuracy
- Different hyperparameter configurations
- Various data sizes and feature sets

## Contact & Support

For issues specific to:
- **cuML**: Check RAPIDS documentation
- **XGBoost GPU**: Check XGBoost CUDA documentation
- **L40s Optimization**: Review NVIDIA L40s best practices
- **HPC Environment**: Consult your HPC administrator

---

**Note**: Ensure your HPC environment supports L40s GPUs and has the necessary CUDA libraries installed before running these scripts. 