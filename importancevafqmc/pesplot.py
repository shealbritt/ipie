import pandas as pd
from pyscf import gto, fci, scf
import glob
import os
import re
import sys
import numpy as np
import matplotlib.pyplot as plt
from lbfgs_fast import Propagator
import jax.numpy as jnp
import jax
sys.path.append('../afqmc')
from utils import reblock
from trial import Trial
from ipieafqmc import ipierun

ipieenergy = []
ipieerror = []
vafqmcenergy = []
vafqmcerror = []
hf_energies = []
distances = []
fcienergies = []
nan_distances = []
dist_for_param = []
warmup_steps = 200
for npy_path in sorted(glob.glob("h2pes/optimal_param-*.npy")):
    match = re.search(r"optimal_param-(\d+\.?\d*)s.npy", npy_path)
    dist = float(match.group(1))
    if match:
        mol = gto.M(atom=f'H 0 0 0; H 0 0 {dist}', basis='sto-3g', unit='bohr')
        nsteps = 5
        dt = 0.1
        #reblocked_ipie= reblock(ipierun(mol))
        #print(reblocked_ipie)
        #ipieenergy.append(reblocked_ipie['ETotal_ac'].values[0])
        #vafqmcenergy.append(reblocked_vafqmc['ETotal_ac'].values[0])
       # ipieerror.append(reblocked_ipie['ETotal_error_ac'].values[0])

        prop = Propagator(mol, dt=dt, nsteps=nsteps, nwalkers=100000, num_chains=50, num_warmup=200) # Example parameters
        prop.trial = Trial(prop.mol)
        prop.trial.get_trial()
        prop.trial.tensora = jnp.array(prop.trial.tensora, dtype=jnp.complex128)
        prop.trial.tensorb = jnp.array(prop.trial.tensorb, dtype=jnp.complex128)
        prop.input = prop.trial.input
        h1e, v2e, nuc, l_tensor = prop.hamiltonian_integral()
        prop.h1e = jnp.array(h1e)
        prop.v2e = jnp.array(v2e)
        prop.nuc = nuc
        prop.l_tensor = jnp.array(l_tensor)
        h1e_repeated = jnp.tile(h1e, (prop.nsteps, 1, 1))  # Repeat h1e nsteps times
        t = jnp.array([prop.dt] * prop.nsteps)
        s = t.copy()
        param_value = float(match.group(1))  # Extract numerical parameter
        params = np.load(npy_path)  # Load NumPy array
        #prop.h1e_params, prop.l_tensor_params, prop.tensora_params, prop.tensorb_params, prop.t_params, prop.s_params = prop.unpack_params(params)
        #samples, acceptance_rate = prop.sampler()
        #vectorized_variational_energy_func = jax.vmap(prop.variational_energy, in_axes=0)
        
        #energies_phases = vectorized_variational_energy_func(samples)
        #energy, phase = energies_phases
        #energy = jnp.real((energy) / jnp.sum(phase))
        vafqmcenergy.append(params)
        #vafqmcerror.append(jnp.std(energy)) #This might not be the way to cALCULATE
        dist_for_param.append(dist)


for csv_path in sorted(glob.glob("csv/h2-*.csv")):  # Now looking directly in 'csv'
    match = re.search(r"h2-(\d+\.?\d*).csv", csv_path)
    if match:
            dist = float(match.group(1))
            df = pd.read_csv(csv_path)
            # Check for NaNs
   #         if df['ipie-ETotal'].isna().any() or df['vafqmc-ETotal'].isna().any():
    #            nan_distances.append(dist)
     #           continue  # Skip this distance but keep going

            try:
                reblocked_ipie = reblock(df['ipie-ETotal'].dropna())
               # reblocked_vafqmc = reblock(df['vafqmc-ETotal'])

                distances.append(dist)
                ipieenergy.append(reblocked_ipie['ETotal_ac'].values[0])
                #vafqmcenergy.append(reblocked_vafqmc['ETotal_ac'].values[0])
                ipieerror.append(reblocked_ipie['ETotal_error_ac'].values[0])
                #vafqmcerror.append(reblocked_vafqmc['ETotal_error_ac'].values[0])
            
            
            except Exception as e:
                print(f"Error processing distance {dist}: {e}")
                nan_distances.append(dist)
                continue  # Skip this iteration safely

# Print distances that were skipped due to NaNs
if nan_distances:
    print(f"Skipped distances due to NaNs: {sorted(nan_distances)}")
sorted_indices = np.argsort(distances)
sorted_distances = np.array(distances)[sorted_indices]
sorted_ipieenergy = np.array(ipieenergy)[sorted_indices]
#sorted_vafqmcenergy = np.array(vafqmcenergy)[sorted_indices]
sorted_ipieerror = np.array(ipieerror)[sorted_indices]
#sorted_vafqmcerror = np.array(vafqmcerror)[sorted_indices]

for dist in sorted_distances:
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
    hf_energies.append(hf_energy)
    fci_energy = cisolver.kernel()[0]
    fcienergies.append(fci_energy)
# Create the plot
plt.figure(figsize=(8, 6))

# Plot IPIE data with error bars
plt.errorbar(sorted_distances, sorted_ipieenergy, yerr=sorted_ipieerror, fmt='o-',
             label="IPIE", capsize=3, markersize=5, color='blue')

# Plot VAFQMC data with error bars
plt.plot(dist_for_param, vafqmcenergy,
             label="VAFQMC", markersize=5, color='red', linestyle = "-", marker = "o")
plt.plot(np.linspace(0.1,4,40), fcienergies, label="fci", linestyle = '--', color = 'green')
plt.plot(np.linspace(0.1,4,40), hf_energies, label="hf", linestyle = '--', color = 'orange')
# Labels and legend
plt.xlabel("H2 Bond Distance (Bohr)", fontsize=12)
plt.ylabel("Total Energy (Ha)", fontsize=12)
plt.title("Energy Comparison: IPIE vs. VAFQMC", fontsize=14)
plt.ylim(-1.15,-0.8)
plt.legend()
plt.grid(True)

# Save the figure
plt.show()
plt.savefig("energy_comparison.png", dpi=300, bbox_inches="tight")

