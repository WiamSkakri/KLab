# Neural Network Training on HPC

## Files
- `nn_hpc.py` - HPC-optimized training script
- `job.sh` - SLURM job script
- `combined.csv` - Training data (you need to provide this)

## How to Run

1. **Prepare data**: Place `combined.csv` in this directory
2. **Submit job**: `sbatch job.sh`
3. **Monitor**: `squeue -u $USER` and `tail -f nn_prediction_training.out`
4. **Results**: Check `results_YYYYMMDD_HHMMSS/` folder

## Key Improvements
- GPU acceleration with CUDA
- Better logging and monitoring
- Enhanced model architecture
- Automatic results collection
- Error handling and cleanup

## Output Files
- `training_output.log` - Complete training log
- `training_results.csv` - Cross-validation results
- `best_model.pth` - Trained model weights
- `job_summary.txt` - Job summary

## Resources
- 12 hours runtime
- 1 GPU
- 32GB RAM
- ai3_env environment

## Prerequisites

1. **Data File**: You need to have your `combined.csv` file in this directory
2. **Environment**: The `ai3_env` virtual environment should be set up in your home directory
3. **HPC Access**: Access to the GPU partition with account `sxk1942`

## Key Improvements for HPC

### Code Optimizations (`nn_hpc.py`)
- **GPU Optimization**: Enhanced model with batch normalization and better GPU utilization
- **Memory Efficiency**: Uses `pin_memory=True` and `non_blocking=True` for faster GPU transfers
- **Logging**: Timestamped logging for better monitoring
- **Error Handling**: Better file path handling and error reporting
- **Performance**: Gradient clipping and optimized batch sizes
- **Results**: Saves trained model and detailed metrics

### Job Script Features (`job.sh`)
- **Resource Allocation**: 12 hours, 1 GPU, 32GB RAM
- **Environment Setup**: Automatic ai3_env activation
- **File Management**: Copies files to scratch space for better I/O performance
- **Error Handling**: Comprehensive error checking
- **Results Management**: Automatic results collection and cleanup
- **Monitoring**: GPU usage tracking and detailed logging

## Expected Performance

The HPC version includes several improvements:
- **Faster Training**: GPU acceleration with optimized data loading
- **Better Monitoring**: Real-time progress tracking
- **Enhanced Model**: Larger network with batch normalization
- **Robust Training**: Early stopping and learning rate scheduling

## Troubleshooting

1. **Job fails immediately**: Check if `combined.csv` exists and ai3_env is available
2. **CUDA not available**: Verify GPU partition access and environment setup
3. **Memory issues**: Reduce batch size in `nn_hpc.py` if needed
4. **Long queue times**: GPU resources might be busy, check with `squeue -p gpu`

## Customization

You can modify training parameters in `nn_hpc.py`:
- `epochs = 150` - Maximum training epochs
- `patience = 20` - Early stopping patience
- `learning_rate = 0.001` - Learning rate
- `batch_size = 64` - Batch size for training

## Next Steps

After training completes successfully:
1. Review results in `training_results.csv`
2. Load the best model with: `torch.load('best_model.pth')`
3. Use the model for inference on new data 