
from pyscf import tools, lo, scf, fci, gto, cc
import numpy as np
import scipy
import logging
from functools import partial
import h5py
import jax.numpy as jnp
import jax.scipy as jsp
import jax
import arviz as az
from scipy.stats import ttest_ind, ttest_1samp

import time
import os
from jaxlib import xla_extension
from jax import jit
import optax
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import sys
import numpyro
#Interestingly numpyro doesn't like being doubl
numpyro.set_host_device_count(jax.device_count()) 
numpyro.enable_x64()
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
sys.path.append('../afqmc/')
from trial import Trial
from walkers import Walkers
from utils import read_fcidump, get_fci
from keymanager import KeyManager
from jax_afqmc import JaxPropagator
jax.config.update('jax_enable_x64', True)


from pyscf import tools, lo, scf, fci
import numpy as np
import scipy
import itertools
import logging
import h5py

logger = logging.getLogger(__name__)
def heidelberger_welch(chain_samples, alpha=0.05):
    """Perform Heidelberger-Welch stationarity test."""
    n = len(chain_samples)
    half = n // 2
    if half < 10: 
        return False
    first_half, second_half = chain_samples[:half], chain_samples[half:]
    stat, p = ttest_ind(first_half, second_half, equal_var=False)
    return p > alpha

def geweke(chain_samples, first_frac=0.1, last_frac=0.5):
    """
    Compute Geweke's convergence diagnostic.
    
    Parameters:
    - chain_samples: 1D np.ndarray of MCMC samples from a single chain
    - first_frac: Fraction of early samples to compare (default: 10%)
    - last_frac: Fraction of late samples to compare (default: 50%)

    Returns:
    - float: z-score (should be close to 0 for convergence)
    """
    n = len(chain_samples)
    first_n = int(n * first_frac)
    last_n = int(n * last_frac)

    if first_n < 10 or last_n < 10:
        return np.nan  # Not enough samples

    first_part = chain_samples[:first_n]
    last_part = chain_samples[-last_n:]
    
    # Perform a t-test: if p > 0.05, no significant difference
    stat, p = ttest_1samp(first_part, np.mean(last_part))

    return stat  # Should be close to 0 if converged

def plot_paginated_trace_autocorr(data, parameter_names, params_per_page=6):
    """Plots trace and autocorrelation plots in multiple pages."""
    num_params = len(parameter_names)
    num_pages = int(np.ceil(num_params / params_per_page))

    for page in range(num_pages):
        start = page * params_per_page
        end = min(start + params_per_page, num_params)
        subset = parameter_names[start:end]

        # Trace Plots
        fig, axes = plt.subplots(len(subset), 2, figsize=(10, 3 * len(subset)))
        az.plot_trace(data, var_names=subset, axes=axes)
        plt.suptitle(f"Trace Plots (Page {page + 1}/{num_pages})", fontsize=14)
        plt.show()
        plt.close(fig)

        # Autocorrelation Plots
        fig, axes = plt.subplots(len(subset), 1, figsize=(10, 3 * len(subset)))
        if len(subset) == 1:
            axes = [axes]  # Ensure it's iterable for a single param
        for ax, param in zip(axes, subset):
            az.plot_autocorr(data, var_names=[param], ax=ax)
            ax.set_title(f"Autocorrelation - {param}")
        plt.suptitle(f"Autocorrelation Plots (Page {page + 1}/{num_pages})", fontsize=14)
        plt.show()
        plt.close(fig)

def check_mcmc_convergence(samples, parameter_names=None, verbose=False):
    """
    Run convergence diagnostics on MCMC samples with shape (n_chains, n_samples, n_parameters).

    Parameters:
    - samples: ndarray of shape (n_chains, n_samples, n_parameters)
    - parameter_names: list of parameter names (length = n_parameters)
    - verbose: whether to print diagnostic results

    Returns:
    - summary: ArviZ summary (ESS, R_hat, etc.)
    - warnings: dict of parameter_name -> list of convergence issues
    """
    if samples.ndim != 3:
        raise ValueError("Expected shape (n_chains, n_samples, n_parameters)")

    n_chains, n_samples, n_params = samples.shape
    if parameter_names is None:
        parameter_names = [f"param_{i}" for i in range(n_params)]
    elif len(parameter_names) != n_params:
        raise ValueError("Length of parameter_names must match number of parameters")

    # Package into arviz-compatible dict
    posterior = {
        name: samples[:, :, i] for i, name in enumerate(parameter_names)
    }
    data = az.from_dict(posterior=posterior)

    # Get summary with R_hat and ESS
    summary = az.summary(data, round_to=2)
    if verbose:
        print("\n=== Summary Statistics ===")
        print(summary)

    # Plot diagnostics
    #plot_paginated_trace_autocorr(data, parameter_names, 6)

    # Initialize warnings
    warnings = {name: [] for name in parameter_names}

    # --- Check R-hat and ESS ---
    for name in parameter_names:
        rhat = summary.loc[name, 'r_hat']
        ess_bulk = summary.loc[name, 'ess_bulk']
        if rhat > 1.1:
            warnings[name].append(f"High R-hat: {rhat:.2f}")
        if ess_bulk < 200:
            warnings[name].append(f"Low ESS (bulk): {ess_bulk:.1f}")

    # --- Geweke diagnostic ---
    if verbose:
        print("\n=== Geweke Diagnostic ===")
    for i, name in enumerate(parameter_names):
        for chain in range(n_chains):
            z = geweke(samples[chain, :, i])
            if abs(z) > 2:
                warnings[name].append(f"Geweke z={z:.2f} (chain {chain})")
            if verbose:
                print(f"{name}, chain {chain}: z = {z:.2f}")

    if verbose:
        print("\n=== Heidelberger-Welch Diagnostic ===")
    for i, name in enumerate(parameter_names):
        for chain in range(n_chains):
            passed = heidelberger_welch(samples[chain, :, i])
            if not passed:
                warnings[name].append(f"Heidelberger-Welch failed (chain {chain})")
            if verbose:
                print(f"{name}, chain {chain}: {'Passed' if passed else 'Failed'}")


    # Print summary of warnings
    if verbose:
        print("\n=== Convergence Warnings ===")
        for name, issues in warnings.items():
            if issues:
                print(f"{name}:")
                for issue in issues:
                    print(f"  - {issue}")
            else:
                print(f"{name}: ✓ No issues")
    
    warning_counts = {name: len(issues) for name, issues in warnings.items()}
    print("\n=== Warning Counts ===")
    total = 0
    for name, count in warning_counts.items():
        total += count
        print(f"{name}: {count} warning(s)")
    print(f"{total} total warning(s)")

    return summary, warnings

@jit
def _normal_pdf(x):
    return  jnp.exp(-0.5 * jnp.sum(x**2)) / (jnp.sqrt(2 * jnp.pi) ** len(x))

@jit
def _get_overlap(l_tensora, l_tensorb, r_tensora, r_tensorb):
    ovlpa = jnp.einsum('pr, pq->rq', l_tensora.conj(), r_tensora)
    ovlpb = jnp.einsum('pr, pq->rq', l_tensorb.conj(), r_tensorb)
    return ovlpa, ovlpb

@jit
def _green_func(la, lb, ra, rb, ovlpa, ovlpb):
    ovlp_inva = jnp.linalg.inv(ovlpa)
    ovlp_invb = jnp.linalg.inv(ovlpb)
    thetaa = jnp.einsum("qp, pr->qr", ra, ovlp_inva)
    thetab = jnp.einsum("qp, pr->qr", rb, ovlp_invb)
    green_funca =jnp.einsum("qr, pr->pq", thetaa, la.conj())
    green_funcb = jnp.einsum("qr, pr->pq", thetab, lb.conj())
    return green_funca, green_funcb

@jit
def _propagate_one_step(h1e_mod, xi, l_tensor, tensora, tensorb, t, s):
    one_body_op_power = jsp.linalg.expm(-t/2 * h1e_mod)
    tensora = jnp.einsum('pq, qr->pr', one_body_op_power, tensora)
    tensorb = jnp.einsum('pq, qr->pr', one_body_op_power, tensorb)
            # 2-body propagator propagation
    two_body_op_power = jsp.linalg.expm(1j * jnp.sqrt(jnp.abs(s)) * ((-1j)**((1-jnp.sign(s))/2)) * jnp.einsum('n, npq->pq', xi, l_tensor))
    '''Tempa = tensora.copy()
        Tempb = tensorb.copy()
        for order_i in range(1, 1+self.taylor_order):
        Tempa = jnp.einsum('pq, qr->pr', two_body_op_power, Tempa) / order_i
        Tempb = jnp.einsum('pq, qr->pr', two_body_op_power, Tempb) / order_i
        self.walkers.tensora += Tempa
        self.walkers.tensorb += Tempb'''
    tensora = jnp.einsum('pq, qr->pr', two_body_op_power, tensora)
    tensorb = jnp.einsum('pq, qr->pr', two_body_op_power, tensorb)
            # 1-body propagator propagation
    tensora = jnp.einsum('pq, qr->pr', one_body_op_power, tensora)
    tensorb = jnp.einsum('pq, qr->pr', one_body_op_power, tensorb)
    return tensora, tensorb

@partial(jit, static_argnums=(7,))
def _propagate(x, tensora_params,tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps):
    tensora = tensora_params.copy()
    tensorb = tensorb_params.copy()
    for i in range(nsteps):
        tensora, tensorb = _propagate_one_step(h1e_params[i], x[i], l_tensor_params, tensora, tensorb, t_params[i], s_params[i])
    prob = _normal_pdf(x.flatten())   
    return prob*tensora, prob*tensorb

@jit
def _local_energy(green_funca, green_funcb, v2e, h1e, nuc):
    local_e2 = jnp.einsum("prqs, pr, qs->", v2e, green_funca, green_funca)
    local_e2 += jnp.einsum("prqs, pr, qs->", v2e, green_funca, green_funcb)
    local_e2 += jnp.einsum("prqs, pr, qs->", v2e, green_funcb, green_funca)
    local_e2 += jnp.einsum("prqs, pr, qs->", v2e, green_funcb, green_funcb)
    local_e2 -= jnp.einsum("prqs, ps, qr->", v2e, green_funca, green_funca)
    local_e2 -= jnp.einsum("prqs, ps, qr->", v2e, green_funcb, green_funcb)
    local_e1 = jnp.einsum("pq, pq->", green_funca, h1e)
    local_e1 += jnp.einsum("pq, pq->", green_funcb, h1e)
    local_e = (local_e1 + 0.5 * local_e2 + nuc)
    return local_e

@partial(jit,static_argnums=(7, 8))   
def _variational_energy(x, tensora_params,tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps, nfields, v2e, h1e, nuc): # Removed method arguments

    x = x.reshape(2, nsteps, nfields)
    x_l = x[0]
    x_r = x[1]
    l_tensora, l_tensorb = _propagate(x_l, tensora_params,tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps) 
    r_tensora, r_tensorb = _propagate(x_r, tensora_params,tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps) 
    wfs = l_tensora, l_tensorb, r_tensora, r_tensorb
    overlapa, overlapb = _get_overlap(*wfs)
    overlap = jnp.linalg.det(overlapa) * jnp.linalg.det(overlapb)
    magnitude = jnp.abs(overlap)
    green_funca, green_funcb = _green_func(*wfs, overlapa, overlapb)
    energy = _local_energy(green_funca, green_funcb, v2e, h1e, nuc) 
    phase = overlap / magnitude
    return energy*phase, phase

@partial(jit, static_argnums=(7,8))
def _target(x, tensora_params, tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps, nfields):
    x = x.reshape(2, nsteps, nfields)
    x_l = x[0]
    x_r = x[1]
    l_tensora, l_tensorb = _propagate(x_l, tensora_params, tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps)
    r_tensora, r_tensorb = _propagate(x_r, tensora_params,tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps)   
    wfs = l_tensora, l_tensorb, r_tensora, r_tensorb   
    overlapa, overlapb = _get_overlap(*wfs)
    overlap = jnp.linalg.det(overlapa) * jnp.linalg.det(overlapb)
    return overlap

@partial(jit, static_argnums=(7,8))
def _potential_fn(x_flat, tensora_params, tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps, nfield):
    x_flat = x_flat['x_flat']
    overlap = _target(x_flat, tensora_params, tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps, nfield)
    log_overlap_magnitude = jnp.log(jnp.abs(overlap))
    potential_energy = -log_overlap_magnitude
    return potential_energy

@partial(jit,static_argnums=(2, 3))   
def _objective_func(samples, params, nsteps, nfields, v2e, h1e, nuc):   
    lam = 1
    B = 0.7
    vectorized_variational_energy_func = jax.vmap(_variational_energy, in_axes=(0, None, None, None, None, None, None, None, None, None, None, None)) # Vectorize over the first argument (x)
    h1e_params, l_tensor_params, tensora_params, tensorb_params,t_params, s_params = params
    # Vectorizsd calculation of energies and phases for all samples
    energies_phases = vectorized_variational_energy_func(samples, tensora_params,tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps, nfields, v2e, h1e, nuc)  # Call the vectorized function
    
    energies, phases = energies_phases # Unpack the tuple of array)
    num = jnp.sum(energies)
    denom = jnp.sum(phases)
    return jnp.real(num / denom) + lam *(jnp.maximum(B-jnp.real(jnp.mean(phases)), 0))**2
    
@partial(jit,static_argnums=(2, 3))   
def _gradient(samples, params, nsteps, nfields, v2e, h1e, nuc):
            gradient_fn = lambda p: _objective_func(samples, p, nsteps, nfields, v2e, h1e, nuc)
            grad_params = jax.grad(gradient_fn)(params)
            return grad_params


def read_fcidump(fname, norb):
    """

    :param fname: electron integrals dumped by pyscf
    :param norb: number of orbitals
    :return: electron integrals for 2nd quantization with chemist's notation
    """
    v2e = np.zeros((norb, norb, norb, norb))
    h1e = np.zeros((norb, norb))

    with open(fname, "r") as f:
        lines = f.readlines()
        for line, info in enumerate(lines):
            if line < 4:
                continue
            line_content = info.split()
            integral = float(line_content[0])
            p, q, r, s = [int(i_index) for i_index in line_content[1:5]]
            if r != 0:
                # v2e[p,q,r,s] is with chemist notation (pq|rs)=(qp|rs)=(pq|sr)=(qp|sr)
                v2e[p-1, q-1, r-1, s-1] = integral
                v2e[q-1, p-1, r-1, s-1] = integral
                v2e[p-1, q-1, s-1, r-1] = integral
                v2e[q-1, p-1, s-1, r-1] = integral
            elif p != 0:
                h1e[p-1, q-1] = integral
                h1e[q-1, p-1] = integral
            else:
                nuc = integral
    return h1e, v2e, nuc




class Propagator(object):
    def __init__(self, mol, dt, nsteps, nwalkers=10000, num_chains=50, num_warmup=200,
                 taylor_order=6, scheme='local energy',
                 stab_freq=5, seed = 47193717):
        self.mol = mol
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.dt = dt
        self.num_chains = num_chains
        self.key_manager = KeyManager(seed)
        self.nfields = None
        self.trial = None
        self.walkers = None
        self.input = None
        self.num_warmup = num_warmup
        self.precomputed_l_tensora = None
        self.precomputed_l_tensorb = None
        self.taylor_order = taylor_order
        self.hybrid_energy = None
        self.mf_shift = None
        self.scheme = scheme
        self.stab_freq = stab_freq
        self.seed = seed
        self.key = jax.random.PRNGKey(self.seed)

    def hamiltonian_integral(self):
        # 1e & 2e integrals
        if self.input == None:
            print("No Trial")
        with h5py.File(self.input) as fa:
            ao_coeff = fa["ao_coeff"][()]
        norb = ao_coeff.shape[0]
        import tempfile
        ftmp = tempfile.NamedTemporaryFile()
        tools.fcidump.from_mo(self.mol, ftmp.name, ao_coeff)
        h1e, eri, nuc = read_fcidump(ftmp.name, norb)
        # Cholesky decomposition
        v2e = eri.reshape((norb**2, -1))
        u, s, v = scipy.linalg.svd(v2e)
        l_tensor = u * np.sqrt(s)
        l_tensor = l_tensor.T
        l_tensor = l_tensor.reshape(l_tensor.shape[0], norb, norb)
        self.nfields = l_tensor.shape[0]
        #originally was r+ not w
        with h5py.File(self.input, "r+") as fa:
            fa["h1e"] = h1e
            fa["nuc"] = nuc
            fa["chol"] = l_tensor

        return jnp.array(h1e, dtype=jnp.complex128), jnp.array(eri,  dtype=jnp.complex128), jnp.array(nuc,  dtype=jnp.complex128), jnp.array(l_tensor,  dtype=jnp.complex128)
    
    def target(self, x):
        return _target(x, self.tensora_params, self.tensorb_params, self.h1e_params, self.l_tensor_params, self.t_params, self.s_params, self.nsteps, self.nfields)
    
    def potential_fn(self, x_flat): 
        return _potential_fn(x_flat, self.tensora_params, self.tensorb_params, self.h1e_params, self.l_tensor_params, self.t_params, self.s_params, self.nsteps, self.nfields)

              
    def sampler(self): 
        keys = jax.random.split(self.key, self.num_chains + 2)
        self.key = keys[-1]
        # Create NUTS kernel with the potential_fn
        nuts_kernel = NUTS(potential_fn=self.potential_fn, target_accept_prob=0.6)
        # Run MCMC
        # Split key for MCM
        num_samples = int(self.nwalkers / self.num_chains) # Sample as many as your current walkers
        mcmc = MCMC(nuts_kernel, num_warmup=self.num_warmup, num_samples=num_samples, 
                    num_chains=1, chain_method='sequential', progress_bar=False)
        
        @jit
        def run_single_chain(single_key, init_param):
        # Run the MCMC sampler for a single chain
            mcmc.run(single_key, init_params={"x_flat": init_param})
            samples = jax.lax.stop_gradient(mcmc.get_samples())
            x_samples_flat = samples["x_flat"]
            return x_samples_flat.reshape(-1, x_samples_flat.shape[-1])
        @partial(jit, static_argnums=(1,2,3))
        def _run_multiple_chains(keys, num_chains, nsteps, nfields):
            init_params = jax.random.normal(keys[0], (num_chains, 2 * nsteps * nfields))
            vectorized_run = jax.vmap(run_single_chain, in_axes=(0, 0))
            samples = vectorized_run(keys[1:-1], init_params)
            return samples
        samples = _run_multiple_chains(keys, self.num_chains, self.nsteps, self.nfields)
        total_samples_shape = samples.shape[0] * samples.shape[1]
        final_samples = samples.reshape(total_samples_shape, samples.shape[-1])
        return final_samples

    
    def get_overlap(self, l_tensora, l_tensorb, r_tensora, r_tensorb):
        return _get_overlap(l_tensora, l_tensorb, r_tensora, r_tensorb)
    
    def propagate_one_step(self, h1e_mod, xi, l_tensor, tensora, tensorb, t, s):
        return _propagate_one_step(h1e_mod, xi, l_tensor, tensora, tensorb, t, s)
    
    def normal_pdf(self, x):
        return _normal_pdf(x)
 
    def propagate(self, x):
        return _propagate(x, self.tensora_params, self.tensorb_params, self.h1e_params, self.l_tensor_params, self.t_params, self.s_params, self.nsteps)
  

    def green_func(self, la, lb, ra, rb):
        ovlpa, ovlpb = self.get_overlap(la, lb, ra, rb)
        return _green_func(la, lb, ra, rb, ovlpa, ovlpb)
    
    def local_energy(self, la, lb, ra, rb):
        green_funca, green_funcb = self.green_func(la, lb, ra, rb) 
        return _local_energy(green_funca, green_funcb, self.v2e, self.h1e, self.nuc)

    
    def unpack_params(self, params):
        """
        Unpacks the parameter vector into the original h1e, l_tensor, tensora, tensorb, t, and s structures.
        """
        # Assuming the shapes are known
        h1e_shape = self.h1e.shape
        l_tensor_shape = self.l_tensor.shape  # Ensure this is correct, e.g., (4, 2, 2)
        tensora_shape = self.trial.tensora.shape
        tensorb_shape = self.trial.tensorb.shape
        
        # Modify h1e shape t o incorporate nsteps as the first dimension
        h1e_shape_with_time = (self.nsteps, *h1e_shape)
        
        # Calculate the lengths
        h1e_length = self.nsteps * h1e_shape[0] * h1e_shape[1]
        l_tensor_length = l_tensor_shape[0] * l_tensor_shape[1] * l_tensor_shape[2]
        tensora_length = tensora_shape[0] * tensora_shape[1]
        tensorb_length = tensorb_shape[0] * tensorb_shape[1]
        
        # Unpack parameters
        h1e_unpacked = params[:h1e_length].reshape(h1e_shape_with_time)
        l_tensor_unpacked = params[h1e_length:h1e_length + l_tensor_length].reshape(l_tensor_shape)
        tensora_unpacked = params[h1e_length + l_tensor_length:h1e_length + l_tensor_length + tensora_length].reshape(tensora_shape)
        tensorb_unpacked = params[h1e_length + l_tensor_length + tensora_length:h1e_length + l_tensor_length + tensora_length + tensorb_length].reshape(tensorb_shape)
        
        # Extract t and s from the remaining params
        t_start = h1e_length + l_tensor_length + tensora_length + tensorb_length
        t = params[t_start:t_start + self.nsteps]
        s = params[t_start + self.nsteps:]
        
        return h1e_unpacked, l_tensor_unpacked, tensora_unpacked, tensorb_unpacked, t, s



    def variational_energy(self, x):
        return _variational_energy(x,  
                                   self.tensora_params, 
                                   self.tensorb_params, 
                                   self.h1e_params, 
                                   self.l_tensor_params, 
                                   self.t_params, 
                                   self.s_params, 
                                   self.nsteps, 
                                   self.nfields,
                                   self.v2e,
                                   self.h1e, 
                                   self.nuc)

   
    def objective_func(self, params):
        params = self.unpack_params(params)
        self.h1e_params, self.l_tensor_params, self.tensora_params, self.tensorb_params, self.t_params, self.s_params = params
        samples = self.sampler()
        #check_mcmc_convergence(samples.reshape(self.num_chains, int(self.nwalkers/self.num_chains), 24), verbose=False)
        return _objective_func(samples, params, self.nsteps, self.nfields, self.v2e, self.h1e, self.nuc)

    def gradient(self, params):
        start = time.time()
        params = self.unpack_params(params)
        self.h1e_params, self.l_tensor_params, self.tensora_params, self.tensorb_params, self.t_params, self.s_params = params 
        samples = self.sampler()
        #check_mcmc_convergence(samples.reshape(self.num_chains, int(self.nwalkers/self.num_chains), 24), verbose=False)
        grads = _gradient(samples, params, self.nsteps, self.nfields, self.v2e, self.h1e, self.nuc)
        end = time.time()
        print("grad", end - start)
        grads = jnp.concatenate([x.ravel() for x in grads])
        return grads
        


    def run(self, max_iter=1000, tol=1e-12, disp=True):
        self.trial = Trial(self.mol)
        self.trial.get_trial()
        self.input = self.trial.input
        self.trial.tensora = jnp.array(self.trial.tensora, dtype=jnp.complex128)
        self.trial.tensorb = jnp.array(self.trial.tensorb, dtype=jnp.complex128)
        h1e, v2e, nuc, l_tensor = self.hamiltonian_integral()
        self.h1e = jnp.array(h1e)
        self.v2e = jnp.array(v2e)
        self.nuc = nuc
        self.l_tensor = jnp.array(l_tensor)
        h1e_repeated = jnp.tile(h1e, (self.nsteps, 1, 1))  # Repeat h1e nsteps times
        t = jnp.array([self.dt] * self.nsteps)
        s = t.copy()
        params = jnp.concatenate([h1e_repeated.flatten(), 
                                  l_tensor.flatten(),  
                                  self.trial.tensora.flatten(), 
                                  self.trial.tensorb.flatten(),
                                  t, s])
        perturbation = 1.0 + 0.1 * np.random.normal(size=params.shape)
        params = params * perturbation
        
        energy_history = []
        grad_norm_history = []
        param_update_norm_history = []
        prev_params = jnp.real(params)

        def callback(xk): # Callback function for scipy.optimize.minimize
            nonlocal prev_params # Allow modification of outer scope variables
            current_energy = self.objective_func(xk)
            current_grad = self.gradient(xk) # Calculate gradient here to get norm
            grad_norm = jnp.linalg.norm(current_grad)
            energy_history.append(jnp.real(current_energy)) # Store real part of energy
            grad_norm_history.append(grad_norm)

            param_update = xk - prev_params
            param_update_norm = jnp.linalg.norm(param_update)
            param_update_norm_history.append(param_update_norm)
            prev_params = xk.copy() # Update previous params for next iteration

            if disp: # Print information during optimization
                print(f"Iteration {len(energy_history)}: Energy = {current_energy:.8f}, Grad Norm = {grad_norm:.8f}, Param Update Norm = {param_update_norm:.8f}")
       

        res = scipy.optimize.minimize(
                self.objective_func,
                jnp.real(params),
                args=(),
                jac=self.gradient,
                tol=tol,
                method="L-BFGS-B",
                options={
                    "maxls": 100,
                    "gtol": 1e-12,
                    "eps": 1e-12,
                    "maxiter": max_iter,
                    "ftol": 1e-12,
                    "maxcor": 1000,
                    "maxfun": max_iter,
                    "disp": disp,
                },
                #callback=callback
            )
        
        print(res)
        opt_params = res.x
        np.save('optimal_params.npy', opt_params)
        # **Plot Results**
        '''iterations = range(1, len(energy_history) + 1)
            
        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(iterations, energy_history, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Variational Energy")
        plt.title("Energy Convergence")

        plt.subplot(1, 3, 2)
        plt.plot(iterations, grad_norm_history, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Gradient Norm")
        plt.yscale('log')  # Log scale to show small values
        plt.title("Gradient Norm Convergence")

        plt.subplot(1, 3, 3)
        plt.plot(iterations, param_update_norm_history, marker='o')
        plt.xlabel("Iteration")
        plt.ylabel("Parameter Update Norm")
        plt.yscale('log')  # Log scale to show small changes
        plt.title("Parameter Update Convergence")

        plt.tight_layout()
        plt.show()'''

        warmup = 300
        key = jax.random.PRNGKey(self.seed)
        opt_params = self.unpack_params(opt_params)
        self.h1e_params, self.l_tensor_params, self.tensora_params, self.tensorb_params, self.t_params, self.s_params = opt_params
        samples = self.sampler()
        check_mcmc_convergence(samples.reshape(self.num_chains, int(self.nwalkers/self.num_chains), 24), verbose=False)
        #while (self.acceptance(samples) < 0.3): 
         #   samples = jax.lax.stop_gradient(self.sampler(opt_params, warmup))
        vectorized_variational_energy_func = jax.vmap(self.variational_energy, in_axes=0) # Vectorize over the first argument (x)
        energies_phases = vectorized_variational_energy_func(samples)  # Call the vectorized function
        energies, phases = energies_phases # Unpack the tuple of arrays
        num = jnp.sum(energies)
        denom = jnp.sum(phases)

        return num/denom
    

if __name__ == "__main__":
    # Define the H2 molecule with PySCF
    mol1 = gto.M(atom='H 0 0 0; H 0 0 1.6', basis='sto-3g', unit='bohr')
    mol = gto.M(
    atom='''
        C       0.000000     1.208000     0.000000
        C       1.208000     0.000000     0.000000
        C       0.000000    -1.208000     0.000000
        C      -1.208000     0.000000     0.000000
        H       0.000000     2.147000     0.000000
        H       2.147000     0.000000     0.000000
        H       0.000000    -2.147000     0.000000
        H      -2.147000     0.000000     0.000000
    ''',
    unit='angstrom',
    basis='cc-pvdz'

    )
    mol = mol1
    # Instantiate the Propagator class for the H2 molecule
    # Example parameters: time step dt=0.01, total simulation time total_t=1.0
    print("Running C4H4")
    t = 1
    times = np.linspace(0, t, int(t/0.005) + 1)
    times = [3]
    energy_list = []
    error_list = []
    
    for _ in times:
        
       # prop = JaxPropagator(mol, dt=0.01, total_t=time, nwalkers=100)  
        #time_list, energy = prop.sampler()
        #plt.plot(time_list, energy, label="jax_afqmc")
        prop = Propagator(mol, dt=0.01, nsteps=3, nwalkers=100000, num_chains=50)
    # Run the simulation to get time and energy lists
        start= time.time()
        energy = prop.run()
        end = time.time()
        print("Total time", end - start)




    #plt.errorbar(time_list, np.real(energy_list), np.real(error_list), label='Importance Propagator', color='r', marker='x')
    mf = scf.RHF(mol)
    hf_energy = mf.kernel()
    cisolver = fci.FCI(mf)
    fci_energy = cisolver.kernel()[0]
    print("vafqmc-energy", energy)
    # Optionally, plot a reference energy line if available
    plt.hlines(fci_energy, xmin=0, xmax=10, color='k', linestyle='--', label='Reference Energy')
    plt.hlines(hf_energy, xmin=0, xmax=10, color='k', linestyle='--', label='HF Energy')
    
    plt.hlines(energy,xmin=0, xmax=10, linestyle=':', label="vafqmc")

    # Add labels, title, and legend
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Energy (Hartree)')
    plt.title('Comparison of AFQMC Propagators: JAX vs VAFQMC for H2')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.savefig("h2-lbfgs.png")
