import pandas as pd
from pyscf import gto, fci, scf
import glob
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt

ipieenergy = []
ipieerror = []
vafqmcenergy = []
vafqmcerror = []
distances = []
fcienergies = []

nan_distances = []
for csv_path in sorted(glob.glob("h2-*.csv")):  # Now looking directly in 'csv'
    match = re.search(r"h2-(\d+\.?\d*).csv", csv_path)
    if match:
            dist = float(match.group(1))
            df = pd.read_csv(csv_path)
            # Check for NaNs
   #         if df['ipie-ETotal'].isna().any() or df['vafqmc-ETotal'].isna().any():
    #            nan_distances.append(dist)
     #           continue  # Skip this distance but keep going

            try:
                reblocked_ipie = df['ipie-ETotal'].dropna()
                reblocked_vafqmc = df['vafqmc-ETotal']

            
                natoms = 2
                time = 50
                mol = gto.M(
                atom=[("H", 0, 0, i * dist) for i in range(natoms)],
                basis='sto-6g',
                unit='Bohr',
                verbose=5  
                )
                mf = scf.RHF(mol)
                hf_energy = mf.kernel()
                cisolver = fci.FCI(mf)
                fci_energy = cisolver.kernel()[0]

                tipie = 0.01 * np.array(range(len(reblocked_ipie)))
                tvafqmc = 0.01 * np.array(range(len(reblocked_vafqmc)))
                fig, axes = plt.subplots()
              
                axes.plot(tvafqmc, reblocked_vafqmc, '--', label ='vafqmc' )
                axes.plot(tvafqmc, [fci_energy] * len(tvafqmc), '--')
                axes.plot(tvafqmc, [hf_energy] * len(tvafqmc), '--')
                axes.set_ylabel("ground state energy")
                axes.set_xlabel("imaginary time")
                axes.plot(tipie, reblocked_ipie, '--', label='ipie')
                axes.legend()
                axes.set_ylim(fci_energy - 0.01, hf_energy+ 0.001)
                axes.set_xlim(0, 3)
                plt.savefig(f"energy_{dist}.png", dpi=300, bbox_inches="tight")
 
            except Exception as e:
                print(f"Error processing distance {dist}: {e}")
                nan_distances.append(dist)
                continue  # Skip this iteration safely

