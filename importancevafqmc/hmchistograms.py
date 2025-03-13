import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci  # Assuming you use PySCF for molecule definition
from hmc_vafqmc_lbfgs import Propagator  # Replace with the actual module where your Propagator class is
import sys
sys.path.append('../afqmc/')
from trial import Trial 
from keymanager import KeyManager
if __name__ == "__main__":
    # Define your molecule and propagator parameters (replace with your setup)
    params = np.load("optimal_params.npy", allow_pickle=True)
    
    mol = gto.M(atom='H 0 0 0; H 0 0 1.6', basis='sto-3g', unit='bohr')
    mf = scf.RHF(mol)
    hf_energy = mf.kernel()
    cisolver = fci.FCI(mf)
    fci_energy = cisolver.kernel()[0]
    print(fci_energy)
    exit()
    nsteps = 10
    dt = 0.1
    prop = Propagator(mol, dt=dt, nsteps=nsteps, nwalkers=100000) # Example parameters
    prop.trial = Trial(prop.mol)
    prop.trial.get_trial()
    prop.trial.tensora = jnp.array(prop.trial.tensora, dtype=jnp.complex128)
    prop.trial.tensorb = jnp.array(prop.trial.tensorb, dtype=jnp.complex128)
    h1e, v2e, nuc, l_tensor = prop.hamiltonian_integral()
    prop.h1e = jnp.array(h1e)
    prop.v2e = jnp.array(v2e)
    prop.nuc = nuc
    prop.l_tensor = jnp.array(l_tensor)
    h1e_repeated = jnp.tile(h1e, (prop.nsteps, 1, 1))  # Repeat h1e nsteps times
    t = jnp.array([prop.dt] * prop.nsteps)
    s = t.copy()
    params = np.load("optimal_params.npy", allow_pickle=True)    
        

    num_hmc_runs = 1000
    warmup_steps = 300 # Example warmup steps for HMC

    # Vectorize simulate_afqmc to run multiple chains in parallel
    vmap_simulate_afqmc = jax.vmap(prop.sampler, in_axes=(None, None), out_axes=(0, 0, 0))

    def run_hmc_chain(seed):
        """Runs a single HMC chain and returns energy and acceptance rate."""
        

    # Vectorize run_hmc_chain to run multiple chains in parallel using vmap
    vmap_run_hmc_chain = jax.vmap(run_hmc_chain, in_axes=(0), out_axes=(0, 0))

    # Generate seeds for each HMC run
    seeds = jnp.arange(num_hmc_runs)

    # Run vectorized HMC to get energies and acceptance rates
    energies = []
    acceptance_rates = []
    for seed in seeds:
        prop.key_manager = KeyManager(seed)
        samples, acceptance_rate = prop.sampler(params, warmup_steps)
        vectorized_variational_energy_func = jax.vmap(prop.variational_energy, in_axes=0)
        
        energies_phases = vectorized_variational_energy_func(samples)
        energy, phase = energies_phases
        energy_estimate = jnp.real(jnp.sum(energy) / jnp.sum(phase))
        print(jnp.sum(phase))
        plt.hist(energy/jnp.mean(phase), bins=10000)
        plt.show()
        print(energy_estimate)
        energies.append(energy_estimate)
        acceptance_rates.append(acceptance_rate)
        
    
    # Convert energies and acceptance rates to numpy arrays for plotting
    energies_np = np.array(energies)
    acceptance_rates_np = np.array(acceptance_rates)
    # --- Histogram Plotting with Color Coding ---
    plt.figure(figsize=(8, 6))
    n_bins = 100  # Adjust the number of bins as needed
    n, bins, patches = plt.hist(energies_np, bins=n_bins, edgecolor='black')

    # Normalize acceptance rates to the range [0, 1] for color mapping

    # Use a colormap to color the histogram bars based on acceptance rate
    cmap = plt.cm.viridis  # You can choose other colormaps like 'viridis', 'plasma', 'magma', 'inferno', etc.
    ax = plt.gca()
    for i in range(n_bins):
        # Find the acceptance rate corresponding to the current bins
        if  i < n_bins - 1:
            bin_mask = (energies_np >= bins[i]) & (energies_np < bins[i+1])
        else:
            bin_mask = (energies_np >= bins[i])
        bin_acceptance_rate = np.mean(acceptance_rates_np[bin_mask])  # Take the mean acceptance rate for the bin
        color_val = cmap(bin_acceptance_rate)  # Map to color
        # Set the color for the corresponding patch
        patches[i].set_facecolor(color_val)

    # Add colorbar as legend for acceptance rate
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])  # Dummy array for matplotlib versions < 3.1
    cbar = plt.colorbar(sm, ax=ax)  # Explicitly pass the Axes object 'ax' to colorbar
    cbar.set_label('Acceptance Rate')
    
    plt.axvline(hf_energy, color='red', linestyle='--', linewidth=2, label='HF Energy')
    plt.axvline(fci_energy, color='blue', linestyle='--', linewidth=2, label='FCI Energy')

    # Add labels and title
    plt.xlabel('Energy (Hartree)')
    plt.ylabel('Frequency')
    plt.title('Histogram of HMC Energies Color-Coded by Acceptance Rate')
    plt.grid(axis='y', alpha=0.75)

    # Save and show plot
    plt.savefig("hmc_energy_histogram_colored.png")
    plt.show()

    print("Histogram plot saved to hmc_energy_histogram_colored.png")