#!/bin/bash
#SBATCH --job-name=h2_array
#SBATCH --output=plots/h2_%a.out
#SBATCH --error=plots/h2_%a.err
#SBATCH --time=32:00:00
#SBATCH --partition=sapphire
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=128G
#SBATCH --array=1-30


module load python/3.10.9-fasrc01
mamba activate afqmc_env

# Ensure pip inside the conda environment is used
which pip  # This will show you which pip is being used, and confirm it's from the 'afqmc_env'

# Install necessary packages using pip
pip install --upgrade pip  # Make sure pip is the latest version
pip install pyscf jax scipy numpy

mamba list  # Check the environment's installed packages


# Set PYTHONPATH to find afqmc module
export PYTHONPATH=/n/home06/mbritt/VAFQMC:$PYTHONPATH

task_id=$SLURM_ARRAY_TASK_ID
workdir="job_${task_id}"
mkdir -p $workdir
cd $workdir

distance_values=$(python -c "import numpy as np; print(' '.join(map(str, np.linspace(0.1, 3, 30))))")
dist=$(echo $distance_values | cut -d' ' -f $task_id)

python ../pes.py $dist
