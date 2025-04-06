#!/bin/bash

#SBATCH --output=out/%x_%j.out
#SBATCH --error=err/%x_%j.err
#SBATCH --time=36:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=100G
#SBATCH --partition=joonholee
#SBATCH --array=0-29
#SBATCH --mail-type=FAIL
#SBATCH --mail-user=sheabritt@college.harvard.edu

module load python/3.10.9-fasrc01
mamba activate afqmc_env

# Ensure pip inside the conda environment is used
which pip  # This will show you which pip is being used, and confirm it's from the 'afqmc_env'

# Install necessary packages using pip
pip install --upgrade pip  # Make sure pip is the latest version
pip install pyscf jax scipy numpy matplotlib numpyro optax arviz

mamba list  # Check the environment's installed packages
distances=($(python -c "import numpy as np; print(' '.join([f'{x:.5f}' for x in [0.4, 2.4]]))"))
d=${distances[$SLURM_ARRAY_TASK_ID]}


echo "Running pes.py for distance $d Å"
python pes.py "$d"
echo "PES scan complete."
