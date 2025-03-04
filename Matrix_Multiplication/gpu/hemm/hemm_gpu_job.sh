
#!/bin/bash
  
#SBATCH -J hemm_gpu_mult    # Job name
#SBATCH -o hemm_gpu_results.out
#SBATCH --time=10:00:00     # 10 hour of wall time
#SBATCH -N 1                # 1 Node
#SBATCH -c 4                # 4 processors
#SBATCH --mem=16gb          # Increased memory
#SBATCH --gres=gpu:1        # Request 1 GPU
#SBATCH --partition=gpu     # Specify gpu partition
# Load required modules
module load Python/3.8.6-GCCcore-10.2.0
module load cuda  # Adjust if cuda module name is different
# Install pandas to user's home directory
pip install --user pandas
# Create a directory in scratch and copy necessary files
SCRATCH_DIR=$PFSDIR/hemm_gpu_mult_${SLURM_JOB_ID}
mkdir -p $SCRATCH_DIR
# Copy the Python script and the GPU HEMM executable to the scratch directory
cp MM_gpu_hemm_auto.py $SCRATCH_DIR/
cp hemm_gpu $SCRATCH_DIR/
# Change to the scratch directory
cd $SCRATCH_DIR
# Make the GPU executable has execute permissions
chmod +x hemm_gpu
# Execute the Python script  
python MM_gpu_hemm_auto.py
# Copy all output files back to the submit directory
cp -ru * $SLURM_SUBMIT_DIR
# Clean up scratch directory
rm -rf $SCRATCH_DIR