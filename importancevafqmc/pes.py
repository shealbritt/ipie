import numpy as np
from pyscf import gto
import sys
import os
import csv
from hmc_vafqmc_lbfgs import Propagator
from itertools import zip_longest
sys.path.append('../')
from afqmc.ipieafqmc import ipierun

def run_simulation(dist):
    os.makedirs('./params-3', exist_ok=True) 
    natoms = 2
    nsteps = 5
    mol = gto.M(
    atom=[("H", 0, 0, i * dist) for i in range(natoms)],
    basis='sto-6g',
    unit='Bohr',
    verbose=5  
    )
    #ipie_energy = ipierun(mol)
    prop = Propagator(mol, dt=0.5, nsteps = nsteps, nwalkers=1000)

    # Run the simulation to get time and energy lists
    opt_params = prop.run()
    np.save(f'./params-3/optimal_param-{dist}s.npy', opt_params)

if __name__ == "__main__":
    dist = float(sys.argv[1])
    run_simulation(dist)



    
