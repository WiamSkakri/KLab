
#!/bin/bash
  
#SBATCH -J matrix_mult      # Job name
#SBATCH -o matrix_multiply_hemm_results.out
#SBATCH --time=02:00:00    # 2 hour of wall time
#SBATCH -N 1               # 1 Node
#SBATCH -c 4               # 4 processors
#SBATCH --mem=4gb          # Assign 4gb memory
# Load required modules
module load Python/3.8.6-GCCcore-10.2.0
# Install pandas to user's home directory
pip install --user pandas
# Create a directory in scratch and copy necessary files
SCRATCH_DIR=$PFSDIR/matrix_mult_${SLURM_JOB_ID}
mkdir -p $SCRATCH_DIR
# Copy the Python script and the matrix multiplication code to the scratch directory
cp MM_hemm_automation.py $SCRATCH_DIR/
cp Matrix_multiply_hemm $SCRATCH_DIR/
# Change to the scratch directory
cd $SCRATCH_DIR
# Execute the Python script
python MM_hemm_automation.py
# Copy all output files back to the submit directory
cp -ru * $SLURM_SUBMIT_DIR
# Clean up scratch directory
rm -rf $SCRATCH_DIR