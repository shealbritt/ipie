#!/bin/bash

#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --time=36:00:00
#SBATCH --ntasks=1                      # Reduced for efficiency, adjust based on your needs
#SBATCH --cpus-per-task=4               # Increase if your code supports threading
#SBATCH --mem=96G                       # Increase memory if necessary
#SBATCH --partition=joonholee
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sheabritt@college.harvard.edu

module load python/3.10.9-fasrc01
source activate afqmc_env  

echo "Using Python from: $(which python)"
echo "Using Pip from: $(which pip)"

# Consider removing the package installation if environment is set before
#pip install --upgrade pip
#pip install pyscf jax scipy numpy matplotlib numpyro optax

# Ensure script is executable and in the correct location
ls -l lbfgs_fast.py

# Run the script and log errors
python lbfgs_fast.py
