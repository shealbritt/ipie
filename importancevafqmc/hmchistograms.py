import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci  # Assuming you use PySCF for molecule definition
from profiler import Propagator  # Replace with the actual module where your Propagator class is
import sys
import time
import numpy as np
from scipy.stats import gaussian_kde, entropy
from sklearn.neighbors import KernelDensity

sys.path.append('../afqmc/')
from trial import Trial 
from keymanager import KeyManager
if __name__ == "__main__":
    # Define your molecule and propagator parameters (replace with your setup)
    params = np.load("optimal_params.npy", allow_pickle=True)
    print(params)
    
    mol = gto.M(atom='H 0 0 0; H 0 0 1.6', basis='sto-3g', unit='bohr')
    mf = scf.RHF(mol)
    hf_energy = mf.kernel()
    cisolver = fci.FCI(mf)
    fci_energy = cisolver.kernel()[0]
    print(fci_energy)
    nsteps = 3
    dt = 0.01
    prop = Propagator(mol, dt=dt, nsteps=nsteps, nwalkers=10000) # Example parameters
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
    params = np.load("optimal_params.npy")
    params = prop.unpack_params(params)
    prop.h1e_params, prop.l_tensor_params, prop.tensora_params, prop.tensorb_params, prop.t_params, prop.s_params = params

    num_hmc_runs = 20
    warmup_steps = 20 # Example warmup steps for HMC

    # Vectorize simulate_afqmc to run multiple chains in parallel
    vmap_simulate_afqmc = jax.vmap(prop.sampler, in_axes=(None, None), out_axes=(0, 0, 0))
    # Generate seeds for each HMC run
    seeds = jnp.arange(num_hmc_runs)

    # Run vectorized HMC to get energies and acceptance rates
    energies = []
    def kl_divergence_kde(samples_p, samples_q, bandwidth=0.1, n_samples=10000):
        # Fit KDE for both distributions
        kde_p = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(samples_p)
        kde_q = KernelDensity(kernel='gaussian', bandwidth=bandwidth).fit(samples_q)

        # Draw random sample points for evaluation
        combined_samples = np.vstack((samples_p, samples_q))
        min_vals = combined_samples.min(axis=0)
        max_vals = combined_samples.max(axis=0)
        rng = np.random.default_rng()
        eval_points = rng.uniform(low=min_vals, high=max_vals, size=(n_samples, samples_p.shape[1]))

        # Evaluate the log densities
        log_dens_p = kde_p.score_samples(eval_points)
        log_dens_q = kde_q.score_samples(eval_points)

        # Calculate densities
        dens_p = np.exp(log_dens_p)
        dens_q = np.exp(log_dens_q)

        # Compute the KL divergence using Monte Carlo integration
        kl_div = np.mean(dens_p * (log_dens_p - log_dens_q))
        return kl_div
    
    for seed in seeds:
        #print(seed)
        key = jax.random.PRNGKey(seed)
        warmup_steps = 20
        prop.num_chains = 1
        start = time.time()
        samples = prop.sampler(warmup_steps, key)
        newkey, subkey = jax.random.split(key)
        vectorized_variational_energy_func = jax.vmap(prop.variational_energy, in_axes=0)
        energies_phases = vectorized_variational_energy_func(samples)
        energy, phase = energies_phases
        energy_estimate = jnp.real(jnp.sum(energy) / jnp.sum(phase))
        #plt.hist((energy/phase).real, bins=100)
        #plt.show()
        print("energy samples", energy_estimate)
        energies.append(energy_estimate)
        
    
    # Convert energies and acceptance rates to numpy arrays for plotting
    energies_np = np.array(energies)
    # --- Histogram Plotting with Color Coding ---
    plt.figure(figsize=(8, 6))
    n_bins = 100  # Adjust the number of bins as needed
    n, bins, patches = plt.hist(energies_np, bins=n_bins, edgecolor='black')
    
    plt.axvline(hf_energy, color='red', linestyle='--', linewidth=2, label='HF Energy')
    plt.axvline(fci_energy, color='blue', linestyle='--', linewidth=2, label='FCI Energy')

    # Add labels and title
    plt.xlabel('Energy (Hartree)')
    plt.ylabel('Frequency')
    plt.title('Histogram of HMC Energies Color-Coded by Acceptance Rate')
    plt.grid(axis='y', alpha=0.75)

    # Save and show plot
    plt.savefig("hmc_energy_histogram_colored_adams.png")
    plt.show()

    print("Histogram plot saved to hmc_energy_histogram_colored.png")