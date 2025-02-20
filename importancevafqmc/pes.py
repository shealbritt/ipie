import numpy as np
from pyscf import gto
import sys
import os
import csv
from importance_vafqmc import Propagator
from itertools import zip_longest
sys.path.append('../')
from afqmc.ipieafqmc import ipierun

def run_simulation(dist):
    os.makedirs('./plots', exist_ok=True) 
    natoms = 2
    time = 50
    mol = gto.M(
    atom=[("H", 0, 0, i * dist) for i in range(natoms)],
    basis='sto-6g',
    unit='Bohr',
    verbose=5  
    )
    ipie_energy = ipierun(mol)
    prop = Propagator(mol, dt=0.01, total_t=time, nwalkers=100)

    # Run the simulation to get time and energy lists
    time_list, vafqmc_energy = prop.run()
    rows = zip_longest(ipie_energy, vafqmc_energy, fillvalue="")
    file_path = f"./plots/h2-{dist}.csv"  # Using f-string to insert the value of `dist`
    with open(file_path, "w", newline="") as file:  
        writer = csv.writer(file)
        writer.writerow(["ipie-ETotal", "vafqmc-ETotal"])
        writer. writerows(rows)


if __name__ == "__main__":
    dist = float(sys.argv[1])
    run_simulation(dist)



    
