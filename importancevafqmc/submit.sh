#!/bin/bash

#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=sapphire
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sheabritt@college.harvard.edu

module load python/3.10.9-fasrc01
mamba activate afqmc_env

# Ensure pip inside the conda environment is used
which pip  # This will show you which pip is being used, and confirm it's from the 'afqmc_env'

# Install necessary packages using pip
pip install --upgrade pip  # Make sure pip is the latest version
pip install pyscf jax scipy numpy matplotlib

mamba list  # Check the environment's installed packages

python pesplot.py
