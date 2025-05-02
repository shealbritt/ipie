
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
from scipy.optimize import fmin_l_bfgs_b as lbfgs
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
        # optional but helps eliminate nondeterminism
#numpyro.set_rng_key(0)

from pyscf import tools, lo, scf, fci
import numpy as np
import scipy
import itertools
import logging
import h5py

logger = logging.getLogger(__name__)

def finite_difference_gradient_subset(f, theta, start=12, end=23, eps=1e-5):
    """
    Finite difference gradient only for parameters in [start, end] inclusive.
    Other entries will be zero.
    """
    theta = np.array(theta)
    grad = np.zeros_like(theta)
    
    for i in range(start, end+1):  # Inclusive
        theta_eps = theta.copy()
        
        theta_eps[i] += eps
        f_plus = f(theta_eps)
        
        theta_eps[i] -= 2 * eps
        f_minus = f(theta_eps)
        
        grad[i] = (f_plus - f_minus) / (2 * eps)
        
    return grad

def check_gradients_finite_difference(obj, params, epsilon=1e-3, verbose=True, tol=1e-3):
    """
    Compare analytical gradients from obj.gradient(params)
    with numerical gradients using finite differences on params 12–23.
    """
    print("Checking gradients...")
    # Define loss function wrapper
    def loss_func(p):
        #prop.samples = None  # Reset any cached samples if needed
        return obj.objective_func(p)
    
    print("Computing analytic gradients...")
    #prop.samples = None
    analytic_grads = obj.gradient(params)
    
    print("Key:", prop.key)

    print("Computing numerical finite-difference gradients (12–23 only)...")
    numerical_grads = finite_difference_gradient_subset(loss_func, params, start=0, end=len(params) - 1, eps=epsilon)
   
    print("Numerical Gradients (12–23):")
    print(numerical_grads)
    
    print("Analytical Gradients (12–23):")
    print(analytic_grads)

    diff = jnp.abs(analytic_grads - numerical_grads)
    max_diff = jnp.max(diff)
    mean_diff = jnp.mean(diff)

    if verbose:
        print(f"\nMax diff (12–23): {max_diff}")
        print(f"Mean diff (12–23): {mean_diff}")
        print(f"Gradients match? {max_diff < tol}")

    return analytic_grads, numerical_grads, diff

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
       # print(f"{name}: {count} warning(s)")
    print(f"{total} total warning(s)")

    return summary, warnings
@jit
def _normal_logpdf(x):
    return -0.5 * jnp.sum(x**2) - 0.5 * len(x) * jnp.log(2 * jnp.pi)

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
    two_body_op_power = jsp.linalg.expm(1j * jnp.sqrt(jnp.abs(s)) * jnp.einsum('n, npq->pq', xi, l_tensor))
    '''Tempa = tensora.copy()
        Tempb = tensorb.copy()
        for order_i in range(1, 1+self.taylor_order):
        Tempa = jnp.einsum('pq, qr->pr', two_body_op_power, Tempa) / order_i
        Tempb = jnp.einsum('pq, qr->pr', two_body_op_power, Tempb) / order_i
        tensora += Tempa
        tensorb += Tempb'''
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
    prob = _normal_logpdf(x.flatten()) 
    return prob, tensora, tensorb

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

@partial(jit,static_argnums=(7,))   
def _variational_energy(x, tensora_params,tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps, v2e, h1e, nuc): # Removed method arguments
    x_l = x[0]
    x_r = x[1]
    #probabilities unnecessar
    # y
    probl, l_tensora, l_tensorb = _propagate(x_l, tensora_params,tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps) 
    probr, r_tensora, r_tensorb = _propagate(x_r, tensora_params,tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps) 
    wfs = jnp.exp(probl) * l_tensora, jnp.exp(probl) * l_tensorb, jnp.exp(probr) * r_tensora, jnp.exp(probr) * r_tensorb
    overlapa, overlapb = _get_overlap(*wfs)
    overlap = jnp.linalg.det(overlapa) * jnp.linalg.det(overlapb)
    magnitude = jnp.abs(overlap)
    green_funca, green_funcb = _green_func(*wfs, overlapa, overlapb)
    energy = _local_energy(green_funca, green_funcb, v2e, h1e, nuc) 
    #Trying angle instead 
    phase = overlap / magnitude
    return energy*phase, phase

@partial(jit, static_argnums=(7,))
def _target(x, tensora_params, tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps):
    x_l = x[0]
    x_r = x[1]
    probl, l_tensora, l_tensorb = _propagate(x_l, tensora_params, tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps)
    probr, r_tensora, r_tensorb = _propagate(x_r, tensora_params,tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps)   
    wfs = l_tensora, l_tensorb, r_tensora, r_tensorb   
    overlapa, overlapb = _get_overlap(*wfs)
    overlap = jnp.linalg.det(overlapa) * jnp.linalg.det(overlapb)
    return probl + probr , overlap

@partial(jit, static_argnums=(7,))
def _potential_fn(x, tensora_params, tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps):
    prob, overlap = _target(x, tensora_params, tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps)
    log_overlap_magnitude = jnp.log(jnp.abs(overlap))
    potential_energy = -(2*prob + log_overlap_magnitude)
    return potential_energy

@partial(jit,static_argnums=(2,))   
def _objective_func(samples, params, nsteps, v2e, h1e, nuc):   
    lam = 0.1
    B = 0.7
    vectorized_variational_energy_func = jax.vmap(_variational_energy, in_axes=(0, None, None, None, None, None, None, None, None, None, None)) # Vectorize over the first argument (x)
    h1e_params, l_tensor_params, tensora_params, tensorb_params,t_params, s_params = params
    # Vectorizsd calculation of energies and phases for all samples
    energies_phases = vectorized_variational_energy_func(samples, tensora_params,tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps, v2e, h1e, nuc)  # Call the vectorized function
    energies, phases = energies_phases # Unpack the tuple of array)
    num = jnp.sum(energies)
    denom = jnp.sum(phases)
    return jnp.real(num / denom), lam *(jnp.maximum(B-jnp.real(jnp.mean(phases)), 0))**2

@partial(jit,static_argnums=(7,))     
def _gradient_helper(x, tensora_params, tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps, v2e, h1e, nuc): # Removed method arguments
    x_l = x[0]
    x_r = x[1]

    #probabilities unnecessary
    probl, l_tensora, l_tensorb = _propagate(x_l, tensora_params,tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps) 
    probr, r_tensora, r_tensorb = _propagate(x_r, tensora_params,tensorb_params, h1e_params, l_tensor_params, t_params, s_params, nsteps) 
    wfs = l_tensora, l_tensorb, r_tensora, r_tensorb
    overlapa, overlapb = _get_overlap(*wfs)
    log_overlap = 2*(probl + probr) + jnp.log(jnp.abs(jnp.linalg.det(overlapa))) + jnp.log(jnp.abs(jnp.linalg.det(overlapb)))
    green_funca, green_funcb = _green_func(*wfs, overlapa, overlapb)
    energy = _local_energy(green_funca, green_funcb, v2e, h1e, nuc) 
    overlap = jnp.linalg.det(overlapa) * jnp.linalg.det(overlapb)
    phase = overlap / jnp.abs(overlap)
    return energy, log_overlap, phase


@partial(jit, static_argnums=(2,))
def _gradient(samples, params, nsteps, v2e, h1e, nuc):
    lam = 0.1
    B = 0.7
    def single_sample_grad(x):
        def energy_overlap_fn(packed_params):
            return _gradient_helper(
                x, packed_params[2], packed_params[3],
                packed_params[0], packed_params[1],
                packed_params[4], packed_params[5],
                nsteps, v2e, h1e, nuc
            )

        energy, logoverlap, phase = energy_overlap_fn(params)

        grad_energy_fn = jax.grad(lambda p: jnp.real(energy_overlap_fn(p)[0]))
        grad_logw_fn = jax.grad(lambda p: jnp.real(jnp.log(energy_overlap_fn(p)[1])))
        grad_phase_fn = jax.grad(lambda p: jnp.real(jnp.log(energy_overlap_fn(p)[2])))

        grad_e = grad_energy_fn(params)
        grad_logw = grad_logw_fn(params)
        grad_phase = grad_phase_fn(params)

        return energy, phase, grad_e, grad_logw, grad_phase

    vectorized = jax.vmap(single_sample_grad, in_axes=(0,))
    e_vals, s_vals, de_vals, dlogw_vals, ds_vals = vectorized(samples)
    de_vals = [x.reshape(x.shape[0], -1) for x in de_vals]
    de_vals = jnp.concatenate(de_vals, axis=1)
    dlogw_vals = [x.reshape(x.shape[0], -1) for x in dlogw_vals]
    dlogw_vals = jnp.concatenate(dlogw_vals, axis=1)
    ds_vals = [x.reshape(x.shape[0], -1) for x in ds_vals]
    ds_vals = jnp.concatenate(ds_vals, axis=1)
    s_vals = s_vals[:, None] 
    e_vals = e_vals[:, None]
    num_grad = jnp.sum(e_vals*s_vals*dlogw_vals, axis=0) + jnp.sum(e_vals * ds_vals + s_vals * de_vals, axis=0)
    denom_grad = jnp.sum(s_vals*dlogw_vals,axis=0) + jnp.sum(ds_vals, axis = 0)
    num = jnp.sum(e_vals*s_vals)
    denom = jnp.sum(s_vals)
    penalty = 2*lam *(B - jnp.mean(s_vals)) * (jnp.mean(s_vals * dlogw_vals, axis=0) + jnp.mean(ds_vals, axis=0))
    return  jnp.real((num_grad*denom - num * denom_grad)/(denom**2)) #- jnp.where(B - jnp.mean(s_vals) > 0, penalty, 0.))

'''
@partial(jit,static_argnums=(2,))   
def _gradient(samples, params, nsteps, v2e, h1e, nuc):
    gradient_fn = lambda p: _objective_func(samples, p, nsteps, v2e, h1e, nuc)
    grad_params = jax.grad(gradient_fn)(params)
    return grad_params'''

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
        self.samples = None
        self.key = jax.random.PRNGKey(self.seed)
        np.random.seed(self.seed)  

    def hamiltonian_integral(self):
        # 1e & 2e integrals
        if self.trial == None:
            print("No Trial")
            exit()
        self.input = self.trial.input
        with h5py.File(self.input) as fa:
            ao_coeff = fa["ao_coeff"][()]
        norb = ao_coeff.shape[0]
        import tempfile
        ftmp = tempfile.NamedTemporaryFile()
        tools.fcidump.from_mo(self.mol, ftmp.name, ao_coeff)
        h1e, eri, nuc = read_fcidump(ftmp.name, norb)
        # Cholesky decomposition
        #v2e = eri.reshape((norb**2, -1))
        #u, s, v = scipy.linalg.svd(v2e)
        #l_tensor = u * np.sqrt(s)
        #l_tensor = l_tensor.T
        #l_tensor = l_tensor.reshape(l_tensor.shape[0], norb, norb)
        #self.nfields = l_tensor.shape[0]
        l_tensor, self.nfields = self.modified_cholesky_decomposition(eri)
        #originally was r+ not w
        with h5py.File(self.input, "r+") as fa:
            fa["h1e"] = h1e
            fa["nuc"] = nuc
            fa["chol"] = l_tensor

        return jnp.array(h1e, dtype=jnp.complex128), jnp.array(eri,  dtype=jnp.complex128), jnp.array(nuc,  dtype=jnp.complex128), jnp.array(l_tensor,  dtype=jnp.complex128)
    
    def target(self, x):
        x = x.reshape(2, self.nsteps, self.nfields)
        return _target(x, self.tensora_params, self.tensorb_params, self.h1e_params, self.l_tensor_params, self.t_params, self.s_params, self.nsteps)
    
    def potential_fn(self, x_flat): 
        x = x_flat['x_flat'].reshape(2, self.nsteps, self.nfields)
        return _potential_fn(x, self.tensora_params, self.tensorb_params, self.h1e_params, self.l_tensor_params, self.t_params, self.s_params, self.nsteps)


    def modified_cholesky_decomposition(self, eri):
       # print("Starting modified Cholesky decomposition (NumPy version).")
        
        eri = np.array(eri)  # Ensure eri is a numpy array
        norb = eri.shape[0]
        v2e = eri.reshape((norb**2, -1))
        thresh = 10**(-9)
        residual = v2e.copy()
        num_fields = 0
        tensor_list = []
        error = np.linalg.norm(residual)
      #  print(f"Initial error: {error}")
        
        while error > thresh:
            max_res, max_i = np.max(np.diag(residual)), np.argmax(np.diag(residual))
            #print(f"Max residual: {max_res}, Index: {max_i}")
    
            diag = np.diag(residual)
            mask = np.sqrt(np.abs(diag * max_res)) <= thresh
            np.fill_diagonal(residual, diag * (~mask))
            
            L = residual[:, max_i] / np.sqrt(max_res)
         #   print(f"L vector (reshaped to {norb}x{norb}): {L.reshape(norb, norb)}")
            tensor_list.append(L.reshape(norb, norb))
            
            Ls = np.array([L.flatten() for L in tensor_list])  # Shape: (n_tensors, norb**2)

            # Use einsum to compute the matrix of dot products:
            products = np.einsum('ki,kj->ij', Ls, Ls)  # Shape: (norb**2, norb**2)

            # Subtract from v2e
            residual = v2e - products
            
            error = np.linalg.norm(residual)
          #  print(f"Current error: {error}")
            num_fields += 1
        
        tensor_list = jnp.array(tensor_list)
       # print("Finished modified Cholesky decomposition (NumPy version).")
        return tensor_list, num_fields


    def sampler(self, key):
        # Step 1: Reference chain with full adaptation
        print("sampling")
        ref_kernel = NUTS(potential_fn=self.potential_fn, target_accept_prob=0.7)
        ref_mcmc = MCMC(ref_kernel, num_warmup=1000, num_samples=100,
                        num_chains=1, chain_method="sequential", progress_bar=False)

        key, subkey = jax.random.split(key)
        init_ref = jax.random.normal(subkey, (2 * self.nsteps * self.nfields,))
        ref_mcmc.run(subkey, init_params={"x_flat": init_ref})
        # Extract tuned values
        step_size = ref_mcmc._last_state.adapt_state.step_size
        mass_matrix = ref_mcmc._last_state.adapt_state.inverse_mass_matrix
        ref_samples = ref_mcmc.get_samples()["x_flat"]

        # Step 2: Initialize multiple chains from the reference posterior
        key, subkey = jax.random.split(key)
        init_samples_idx = jax.random.choice(subkey, ref_samples.shape[0], shape=(self.num_chains,))
        init_points = ref_samples[init_samples_idx]
        # Step 3: Run new chains using tuned mass matrix and a short warm-up
        short_adapt_kernel = NUTS(
            potential_fn=self.potential_fn,
            target_accept_prob=0.7,
            step_size = step_size,
            inverse_mass_matrix=mass_matrix  # keep mass matrix fixed
        )
        mcmc = MCMC(
            short_adapt_kernel,
            num_warmup=10,  # short local tuning of step size
            num_samples=int(self.nwalkers / self.num_chains),
            num_chains=self.num_chains,
            chain_method='vectorized',
            progress_bar=False
        )

        key, subkey = jax.random.split(key)
        mcmc.run(subkey, init_params={"x_flat": init_points})
        samples = jax.lax.stop_gradient(mcmc.get_samples())
        return samples["x_flat"]


    
    def get_overlap(self, l_tensora, l_tensorb, r_tensora, r_tensorb):
        return _get_overlap(l_tensora, l_tensorb, r_tensora, r_tensorb)
    
    def propagate_one_step(self, h1e_mod, xi, l_tensor, tensora, tensorb, t, s):
        return _propagate_one_step(h1e_mod, xi, l_tensor, tensora, tensorb, t, s)
    
    def normal_logpdf(self, x):
        return _normal_logpdf(x)
 
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
        x = x.reshape(2, self.nsteps, self.nfields)
        return _variational_energy(x,  
                                   self.tensora_params, 
                                   self.tensorb_params, 
                                   self.h1e_params, 
                                   self.l_tensor_params, 
                                   self.t_params, 
                                   self.s_params, 
                                   self.nsteps, 
                                   self.v2e,
                                   self.h1e, 
                                   self.nuc)

    def energyfromparams(self, params):
        params = self.unpack_params(params)
        self.h1e_params, self.l_tensor_params, self.tensora_params, self.tensorb_params, self.t_params, self.s_params = params
        if self.samples is None:
            samples = self.sampler()
        samples = samples.reshape(self.nwalkers, 2, self.nsteps, self.nfields)
        vectorized_variational_energy_func = jax.vmap(_variational_energy, in_axes=(0, None, None, None, None, None, None, None, None, None, None)) # Vectorize over the first argument (x)
        h1e_params, l_tensor_params, tensora_params, tensorb_params,t_params, s_params = params
        # Vectorizsd calculation of energies and phases for all samples
        energies_phases = vectorized_variational_energy_func(samples, tensora_params,tensorb_params, h1e_params, l_tensor_params, t_params, s_params, self.nsteps, self.v2e, self.h1e, self.nuc)  # Call the vectorized function
        energies, phases = energies_phases # Unpack the tuple of array)
        num = jnp.sum(energies)
        denom = jnp.sum(phases)
        return num / denom

    def set_params(self, params):
        params = self.unpack_params(params)
        self.h1e_params, self.l_tensor_params, self.tensora_params, self.tensorb_params, self.t_params, self.s_params = params
   
    
    
    def objective_func(self, params):
        print("obj")
        params = self.unpack_params(params)
        self.h1e_params, self.l_tensor_params, self.tensora_params, self.tensorb_params, self.t_params, self.s_params = params
        if self.samples is None:
            key, self.key = jax.random.split(self.key)
            self.samples = self.sampler(key)
        samples = self.samples
        samples = samples.reshape(self.nwalkers, 2, self.nsteps, self.nfields)
        energy, penalty = _objective_func(samples, params, self.nsteps, self.v2e, self.h1e, self.nuc)
        if penalty > 0:
            print(penalty)
        return float(energy) #+ penalty


    def gradient(self, params):
        print("Gradient")
        params = self.unpack_params(params)
        self.h1e_params, self.l_tensor_params, self.tensora_params, self.tensorb_params, self.t_params, self.s_params = params 
        if self.samples is None:
            key, self.key = jax.random.split(self.key)
            self.samples = self.sampler(key)
        samples = self.samples
        samples = samples.reshape(self.nwalkers, 2, self.nsteps, self.nfields)
        grads = _gradient(samples, params, self.nsteps, self.v2e, self.h1e, self.nuc)
        grads = jnp.concatenate([x.ravel() for x in grads])
        grads = jnp.clip(grads, a_min=-0.5, a_max=0.5)
        self.samples = None
        return grads
    

    

    def run(self, max_iter=1000, tol=1e-5, disp=True, seed=1222):
        self.trial = Trial(self.mol)
        self.trial.get_trial()
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
        self.params = params
        energy_history = []
        grad_norm_history = []
        param_update_norm_history = []
        prev_params = jnp.real(params)

        def callback(xk, current_grad, current_energy):
            """Callback function to log optimization progress."""
            nonlocal prev_params
            current_grad = np.array(current_grad)
            grad_norm = jnp.linalg.norm(current_grad)
            energy_history.append(jnp.real(current_energy))
            grad_norm_history.append(grad_norm)

            param_update = xk - prev_params
            param_update_norm = jnp.linalg.norm(param_update)
            param_update_norm_history.append(param_update_norm)
            prev_params = xk.copy()

            if disp:
                print(f"Iteration {len(energy_history)}: "
                    f"Energy = {current_energy:.8f}, "
                    f"Grad Norm = {grad_norm:.8f}, "
                    f"Param Update Norm = {param_update_norm:.8f}")
        
        # **Optimization Loop**
        learning_rate = 3e-4
        def schedule(step):
            return learning_rate / (1 + step / 5000)

        # Create the Adam optimizer with the learning rate schedule
        optimizer = optax.adabelief(learning_rate=schedule, b1=0.9, b2=0.99)
        opt_state = optimizer.init(params)
    
        @jit
        def train_step(params, grads, opt_state):
            updates, opt_state = optimizer.update(grads, opt_state)
            params = optax.apply_updates(params, updates)
            return params, opt_state
        
        
        gradslist = []
        grad_norms = []
        losses = []
        patience = 100        
        best_loss = float('inf')
        wait = 0
        def moving_average(arr, window_size=20):
            return np.convolve(arr, np.ones(window_size)/window_size, mode='valid')
        
        for i in range(max_iter):
            print(f"Iteration {i+1}", flush=True)
            
            obj = float(self.objective_func(params))
            grads = np.array(self.gradient(params))
            new_params, opt_state = train_step(params, grads, opt_state)

            gradslist.append(grads)
            grad_norms.append(np.linalg.norm(grads))
            losses.append(obj)

            # --- Convergence Criteria ---
            grad_tol = np.linalg.norm(grads) < tol
            param_change_tol = np.linalg.norm(new_params - params) < tol

            # Moving average on loss (after enough iterations)
            if i >= 20:
                smoothed_losses = moving_average(losses)
                current_loss_ma = smoothed_losses[-1]
                loss_improved = best_loss - current_loss_ma > tol
                if loss_improved:
                    best_loss = current_loss_ma
                    wait = 0
                else:
                    wait += 1

                if grad_tol or param_change_tol or wait >= patience:
                    reason = []
                    if grad_tol:
                        reason.append("gradient norm below threshold")
                    if param_change_tol:
                        reason.append("parameter change below threshold")
                    if wait >= patience:
                        reason.append(f"no improvement for {patience} iterations")
                    
                    print(f"Converged due to: {', '.join(reason)}.")
                    print(obj)
                    break

            params = new_params
           
           # - the following should be deleted later
            if (i + 1) % 50 == 0:
                grads_array = np.array(gradslist)  # Stack all gradients

                # Gradient Norms and Loss
                grad_norms_arr = np.array(grad_norms)
                losses_arr = np.array(losses)

                plt.figure(figsize=(14,6))

                # Plot smoothed gradient norm
                plt.subplot(1, 2, 1)
                if len(grad_norms_arr) > 20:
                    plt.plot(moving_average(grad_norms_arr, window_size=20))
                else:
                    plt.plot(grad_norms_arr)
                plt.xlabel('Iteration')
                plt.ylabel('Gradient Norm (Smoothed)')
                plt.title('Gradient Norm Over Time')

                # Plot smoothed loss
                plt.subplot(1, 2, 2)
                if len(losses_arr) > 20:
                    plt.plot(moving_average(losses_arr, window_size=20))
                else:
                    plt.plot(losses_arr)
                plt.xlabel('Iteration')
                plt.ylabel('Loss (Smoothed)')
                plt.title('Loss Over Time')

                plt.tight_layout()
                plt.show()
                i0 = 0
                i1 = i0 + h1e_repeated.size
                print(i1)
                i2 = i1 + l_tensor.size 
                print(i2)
                i3 = i2 + self.trial.tensora.size
                i4 = i3 + self.trial.tensorb.size
                print(i4)
                i5 = i4 + t.size
                i6 = i5 + s.size

                param_types = {
                    'h1e': slice(i0, i1),
                    'l_tensor': slice(i1, i2),
                    'tensora': slice(i2, i3),
                    'tensorb': slice(i3, i4),
                    't': slice(i4, i5),
                    's': slice(i5, i6),
                }

                for name, sl in param_types.items():
                    plt.figure(figsize=(12, 6))
                    for j in range(sl.start, sl.stop):
                        plt.plot(grads_array[:, j], label=f'{name} param {j - sl.start}')
                    plt.xlabel('Iteration')
                    plt.ylabel('Gradient Magnitude')
                    plt.title(f'Gradient of {name.capitalize()} Parameters Up to Iteration {i+1}')
                    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small', ncol=1)
                    plt.tight_layout()
                    plt.show()
        #np.save('optimal_params_adams.npy', opt_params)

        # **Plot Results*
      
        warmup = 200 
        self.key, key = jax.random.split(self.key)
        opt_params = self.unpack_params(params)
        self.h1e_params, self.l_tensor_params, self.tensora_params, self.tensorb_params, self.t_params, self.s_params = opt_params
        newkey, self.key = jax.random.split(self.key)
        samples = self.sampler(newkey)
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
    mol = gto.M(atom='H 0 0 0; H 0 0 1.6', basis='sto-3g', unit='bohr')
    # Instantiate the Propagator class for the H2 molecule
    # Example parameters: time step dt=0.01, total simulation time total_t=1.0
    print("Running H2")
    t = 1
    times = np.linspace(0, t, int(t/0.005) + 1)
    times = [3]
    energy_list = []
    error_list = []
    
    for _ in times:
        
       # prop = JaxPropagator(mol, dt=0.01, total_t=time, nwalkers=100)  
        #time_list, energy = prop.sampler()
        #plt.plot(time_list, energy, label="jax_afqmc")
        prop = Propagator(mol, dt=0.1, nsteps=3, nwalkers=10000, num_chains=50, num_warmup=200)

    # Run the simulation to get time and energy lists

        energy = prop.run()



    #plt.errorbar(time_list, np.real(energy_list), np.real(error_list), label='Importance Propagator', color='r', marker='x')
    mf = scf.RHF(mol)
    hf_energy = mf.kernel()
    cisolver = fci.FCI(mf)
    fci_energy = cisolver.kernel()[0]
    print("fci-energy",fci_energy, flush=True)
    print("vafqmc-energy", energy, flush=True)
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
    plt.savefig("h2-adams.png")

