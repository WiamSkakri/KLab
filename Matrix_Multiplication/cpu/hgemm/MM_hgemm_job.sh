#!/bin/bash
#SBATCH -o matrix_multiply_hgemm_results.out  # Output file
#SBATCH --time=02:00:00                 # 2 hour of wall time
#SBATCH -N 1                            # 1 Node
#SBATCH -c 4                            # 4 processors
#SBATCH --mem=4gb                       # 4GB memory

module load Python/3.8.6-GCCcore-10.2.0 # Load Python module

# Copy the Python script and the matrix multiplication code to the scratch directory
cp -r MM_hgemm_automation.py MM_hgemm.cpp Matrix_multiply_hgemm $PFSDIR
cd $PFSDIR  # Change to the scratch directory

# Execute the Python script
python MM_hgemm_automation.py

# Copy all output files back to the working directory
cp -ru * $SLURM_SUBMIT_DIR
