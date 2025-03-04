
#!/bin/bash
  
#SBATCH -o matrix_multiply_results.out
#SBATCH --time=01:00:00 # 1 hour of wall time
#SBATCH -N 1 # 1 Node
#SBATCH -c 4 # 4 processor
#SBATCH --mem=4gb # Assign 4gb memory
module load Python/3.8.6-GCCcore-10.2.0 # Loading python
# Copy the Python script and the matrix multiplication code to the scratch directory


# These are incorrect file name!!
cp -r MM_automation.py MM_run.cpp matrix_multiply  $PFSDIR
cd $PFSDIR # Change to the scratch directory
# Execute the Python script
python MM_automation.py
# Copy all output files back to the working directory
cp -ru * $SLURM_SUBMIT_DIR