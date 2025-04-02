#!/bin/bash

#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --time=36:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=joonholee
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sheabritt@college.harvard.edu

module load python/3.10.9-fasrc01
mamba activate afqmc_env

# Ensure pip inside the conda environment is used
which pip  # This will show you which pip is being used, and confirm it's from the 'afqmc_env'

# Install necessary packages using pip
pip install --upgrade pip  # Make sure pip is the latest version
pip install pyscf jax scipy numpy matplotlib numpyro

mamba list  # Check the environment's installed packages
distances=(0.1 0.25 0.4 0.55 0.7 0.85 1.0 1.2 1.4 1.6 1.8 2.0 2.2 2.4 2.6 2.8 3.0)

for d in "${distances[@]}"; do
    echo "Running pes.py for distance $d Å"
    python pes.py "$d"
done

echo "PES scan complete."
