import numpy as np
from pyscf import gto
import sys
import os
import csv
from lbfgs_fast import Propagator
from itertools import zip_longest
sys.path.append(os.path.abspath("../"))
#from afqmc.ipieafqmc import ipierun

def run_simulation(dist):
    os.makedirs('./h2pes', exist_ok=True) 
    natoms = 2
    nsteps = 3
    mol = gto.M(
    atom=[("H", 0, 0, i * dist) for i in range(natoms)],
    basis='sto-6g',
    unit='Bohr',
    verbose=5  
    )
    #ipie_energy = ipierun(mol)
    prop = Propagator(mol, dt=0.1, nsteps = nsteps, nwalkers=100000, num_chains=50, num_warmup=200)

    # Run the simulation to get time and energy lists
    opt_params = prop.run()
    np.save(f'./h2pes/optimal_param-{dist}s.npy', opt_params)

if __name__ == "__main__":
    dist = float(sys.argv[1])
    run_simulation(dist)



    
