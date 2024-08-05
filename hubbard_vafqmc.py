
import numpy as np
import jax.numpy as jnp
import jax
from jax import jit, random
from scipy.optimize import minimize
import jax.scipy as jsp
jax.config.update('jax_enable_x64', True)
from scipy.optimize import fsolve
import ctypes
import numpy
import scipy.linalg
from pyscf import ao2mo
from pyscf.fci import cistring
from itertools import combinations
from pyscf import gto, scf, ao2mo, mcscf, fci, lo
from ipie.systems.generic import Generic
from ipie.legacy.hamiltonians.hubbard import Hubbard
from ipie.legacy.trial_wavefunction.hubbard_uhf import HubbardUHF
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
class KeyManager:
    def __init__(self, seed):
        self.key = random.PRNGKey(seed)

    def get_key(self):
        self.key, subkey = random.split(self.key)
        return subkey


def get_wavefunction(H_MF, num_electrons):
    """
    Constructs the Slater determinant wave function from the mean-field Hamiltonian.

    Parameters:
    H_MF (array): Mean-field Hamiltonian matrix of size (L^2, L^2)
    num_electrons (int): Number of electrons in the system

    Returns:
    psi_MF (array): Normalized Slater determinant wave function (L^2, num_electrons)
    """
    # Diagonalize the mean-field Hamiltonian
    eigenvalues, eigenvectors = np.linalg.eigh(H_MF)
    # Sort eigenvectors according to eigenvalues
    sorted_indices = np.argsort(eigenvalues)
    sorted_eigenvectors = eigenvectors[:, sorted_indices]

    # Construct the Slater determinant by filling up to the Fermi level
    psi_MF = sorted_eigenvectors[:, :num_electrons]

    # Normalize the wave function
    norms = jnp.linalg.norm(psi_MF, axis=0)
    psi_MF_normalized = psi_MF / norms
    return jnp.array(psi_MF_normalized)


def setup(L, nup, ndown, t, U):
    options = {
    "name": "Hubbard",
    "nup": nup,
    "ndown": ndown,
    "nx": L,
    "ny": L,
    "U": U,
    "t": t,
    "xpbc" :False,
    "ypbc" :False,
    "ueff": U
    }

    system = Generic(nelec=(nup, ndown))#, options=options)
    ham = Hubbard(options, verbose=True)
    uhf = HubbardUHF(
        system, ham, {"ueff": ham.U}, verbose=True
    )
    mol = gto.M()
    mol.nelectron = nup + ndown
    mol.spin = nup - ndown
    mol.verbose = 3

    # incore_anyway=True ensures the customized Hamiltonian (the _eri attribute)
    # to be used.  Without this parameter, the MO integral transformation may
    # ignore the customized Hamiltonian if memory is not enough.
    mol.incore_anyway = True

    n = ham.nbasis
    h1 = ham.T[0].copy()
    eri = np.zeros((n,n,n,n))
    for i in range(n):
        eri[i,i,i,i] = ham.U
    #eval, evec = np.linalg.eigh(h1)
   # tensor_alpha = evec[:, :mol.nelec[0]]
    #if mol.nelec[1] > 0:
     #   tensor_beta = evec[:, mol.nelec[0]:]
    #else:
    #    tensor_beta = np.empty()
   # return h1, eri, tensor_alpha, tensor_beta
    mf = scf.UHF(mol)
    mf.max_cycle = 100
    mf.max_memory = 10000
    mf.get_hcore = lambda *args: h1
    mf.get_ovlp = lambda *args: np.eye(n)

    eri_symm = ao2mo.restore(8, eri, n)
    mf._eri = eri_symm.copy()
    mf.init_guess = '1e'
    mf.kernel(uhf.G)

    mf.max_cycle = 1000
    mf = mf.newton().run()


    mo1 = mf.stability()[0]
    dm1 = mf.make_rdm1(mo1, mf.mo_occ)

    dmb = dm1[1].copy()
    dmb += np.random.randn(dmb.shape[0]*dmb.shape[1]).reshape(dmb.shape)
    dm1[1] = dmb.copy()
    mf = mf.run(dm1)
    mf.stability()

    dm_converged = mf.make_rdm1(mf.mo_coeff, mf.mo_occ)
    mf = mf.run(dm_converged)
    dm_alpha, dm_beta = mf.make_rdm1()
    print(dm_alpha - dm_beta)
    nalpha, _ = np.linalg.eigh(dm_alpha)
    nbeta, _ = np.linalg.eigh(dm_beta)
    mo_coeffs = mf.mo_coeff
    smat = np.eye(L * L)
    ao_coeff = lo.orth.lowdin(smat)
    xinv = np.linalg.inv(ao_coeff)
    tensor_alpha = xinv.dot(mo_coeffs[0][:, :mol.nelec[0]])
    if mol.nelec[1] > 0:
      tensor_beta = xinv.dot(mo_coeffs[1][:, :mol.nelec[1]])
    else:
      tensor_beta = np.empty()
    return h1, eri, tensor_alpha, tensor_beta


def initialize(L, nalpha, nbeta, t = 1.0, mu = 1.75, U = 4.0):
    H_MF, V, psi_alpha, psi_beta = setup(L, nalpha, nbeta, t, U)
    return psi_alpha, psi_beta, H_MF, V
@jit
def ovlp_helper(wf1_alpha, wf1_beta, wf2_alpha, wf2_beta):
    overlap_matrix_alpha = jnp.einsum('pr, pq -> rq', wf1_alpha.conj(), wf2_alpha)
    overlap_alpha = jnp.linalg.det(overlap_matrix_alpha)
    if wf1_beta.size != 0 and wf2_beta.size != 0:
        # Compute beta electron overlap if provided and not empty
        overlap_matrix_beta = jnp.einsum('pr, pq -> rq', wf1_beta.conj(), wf2_beta)
        overlap_beta = jnp.linalg.det(overlap_matrix_beta)
    else:
        # If no beta electrons, set overlap_beta to 1 (neutral element for multiplication)
        overlap_beta = 1.0
    overlap = overlap_alpha * overlap_beta
    magnitude = jnp.abs(overlap)
    phase = jnp.divide(overlap, magnitude)
    return magnitude, phase



def green_func_helper(wf1_alpha, wf1_beta, wf2_alpha, wf2_beta):
    inv_ovlp_alpha = jnp.linalg.inv(jnp.einsum('pr, pq -> rq', wf1_alpha.conj(), wf2_alpha))
    inv_ovlp_beta = jnp.linalg.inv(jnp.einsum('pr, pq -> rq', wf1_beta.conj(), wf2_beta))
    intermediate_alpha = jnp.einsum('jk, kl -> jl', wf2_alpha, inv_ovlp_alpha)
    green_func_alpha = jnp.einsum('jl, ml -> mj', intermediate_alpha, wf1_alpha.conj())
    intermediate_beta = jnp.einsum('jk, kl -> jl', wf2_beta, inv_ovlp_beta)
    green_func_beta = jnp.einsum('jl, ml -> mj', intermediate_beta, wf1_beta.conj())
    return green_func_alpha, green_func_beta



def energy_helper(galpha, gbeta, ku, V):
    e1 = jnp.einsum("pq, pq ->", galpha, ku)
    e1 += jnp.einsum("pq, pq ->", gbeta, ku)
    e2 = jnp.einsum("prqs, pr, qs ->", V, galpha, galpha)
    e2 += jnp.einsum("prqs, pr, qs ->", V, gbeta, galpha)
    e2 += jnp.einsum("prqs, pr, qs ->", V, galpha, gbeta)
    e2 += jnp.einsum("prqs, pr, qs ->", V, gbeta, gbeta)
    e2 -= jnp.einsum("prqs, ps, qr ->", V, galpha, galpha)
    e2 -= jnp.einsum("prqs, ps, qr ->", V, gbeta, gbeta)
    local_energy = e1 + 0.5 * e2
    return local_energy



#def pack(psi_alpha, psi_beta, dt, t, mu, U) -> jnp.ndarray:
#    return jnp.concatenate((psi_alpha.flatten(), psi_beta.flatten(), jnp.array([dt]), jnp.array([t]), jnp.array([mu]), jnp.array([U])), dtype=jnp.float64)

  # definitely Correct

#def unpack(packed_array: jnp.ndarray, shape_psi_alpha, shape_psi_beta):
 #   idx = 0

#    size_psi_alpha = jnp.prod(jnp.array(shape_psi_alpha))
#    psi_alpha = packed_array[idx:idx + size_psi_alpha].reshape(shape_psi_alpha)
#    idx += size_psi_alpha

#    size_psi_beta = jnp.prod(jnp.array(shape_psi_beta))
#    psi_beta = packed_array[idx:idx + size_psi_beta].reshape(shape_psi_beta)
#    idx += size_psi_beta

#    dt = packed_array[idx]
#    idx += 1
#    t = packed_array[idx]
#    idx += 1
#    mu = packed_array[idx]
#    idx += 1
#    U = packed_array[idx]
#    return jnp.array(psi_alpha), jnp.array(psi_beta), dt, t, mu, U

    # definitely Correct

def get_green_func(wf1_alpha, wf1_beta, wf2_alpha, wf2_beta):
    return green_func_helper(wf1_alpha, wf1_beta, wf2_alpha, wf2_beta)

    # definitely Correct
def get_overlap(wf1_alpha, wf1_beta, wf2_alpha, wf2_beta):
    return ovlp_helper(wf1_alpha, wf1_beta, wf2_alpha, wf2_beta)


def occupation_numbers(green_func_alpha, green_func_beta):
    n_alpha = jnp.diag(green_func_alpha)
    n_beta = jnp.diag(green_func_beta)
    return n_alpha, n_beta

@jit
def orthogonalize(U_matrix):
    Q, R = jnp.linalg.qr(U_matrix)
    return Q


        #Trying new propagator
def propagate(n_steps, fields, wf_alpha, wf_beta, H_MF, U, tau):
        def equation(gamma, n):
            return gamma**n - (n/2) * gamma - (1 - n/2)
        gamma_initial_guess = 1.1 # Initial guess

        gamma_solution = fsolve(equation, gamma_initial_guess, args=(n_steps))
        gamma = gamma_solution[0]
        def helper(wf_alpha, wf_beta, n_steps, H_MF, U, tau, gamma = 1.1):
            h = jnp.zeros(n_steps)
            t = jnp.zeros(n_steps + 1)
            lam = jnp.zeros(n_steps)
            U_matrix_a = jnp.eye(H_MF.shape[0])
            U_matrix_b = jnp.eye(H_MF.shape[0])
            U_matrix_a, U_matrix_b = jnp.matmul(U_matrix_a, wf_alpha), jnp.matmul(U_matrix_b, wf_beta)

            # Calculate h_i and t_i
            for i in range(0, n_steps):
                h = h.at[i].set(tau / n_steps)#gamma**(i) * tau / (n_steps))
                lam = lam.at[i].set(jnp.arccosh(jnp.exp(U * h[i])))
        # Calculate t_i
            for i in range(0, n_steps):
                t = t.at[i].set((h[i] + h[i - 1]) / 2 if i > 0 else h[i] / 2)
            t = t.at[n_steps].set(t[0])
            U_matrix_a = jnp.dot(jsp.linalg.expm(-H_MF * t[n_steps]), U_matrix_a)
            U_matrix_b = jnp.dot(jsp.linalg.expm(-H_MF * t[n_steps]), U_matrix_b)
            for i in range(n_steps - 1,  -1, -1):
              two_body_a = jsp.linalg.expm(lam[i] * jnp.diag(fields[i]))
              two_body_b = jsp.linalg.expm(lam[i] * -jnp.diag(fields[i]))
              U_matrix_a = jnp.dot(two_body_a, U_matrix_a)
              U_matrix_b = jnp.dot(two_body_b, U_matrix_b)
              one_body = jsp.linalg.expm(-t[i] * H_MF)
              U_matrix_a = jnp.dot(one_body, U_matrix_a)
              U_matrix_b = jnp.dot(one_body, U_matrix_b)
              if (n_steps - i) % 5 == 0:
                U_matrix_a = orthogonalize(U_matrix_a)
                U_matrix_b = orthogonalize(U_matrix_b)

            return orthogonalize(U_matrix_a), orthogonalize(U_matrix_b)
        return helper(wf_alpha, wf_beta, n_steps, H_MF, U, tau, gamma)



def get_local_energy(hmf, V, wf1_alpha, wf1_beta, wf2_alpha, wf2_beta):
    galpha, gbeta =  get_green_func(wf1_alpha, wf1_beta, wf2_alpha, wf2_beta)
    return energy_helper(galpha, gbeta, hmf, V)



def switcher(key, mat, frac):
    shape = mat.shape
    flat_mat = mat.reshape(-1)
    idx = jnp.arange(len(flat_mat))
    permuted_idx = random.permutation(key, idx)
    num_indices = int(len(flat_mat) * frac)
    indices = permuted_idx[:num_indices]
    key, subkey = random.split(key)
    new_values = random.choice(key, jnp.array([-1, 1]), shape=(num_indices,))
    flat_mat = flat_mat.at[indices].set(new_values)

    return flat_mat.reshape(shape)



    # I'm pretty confident
def met_hasting_sample(L, psi_alpha, psi_beta, ham, hmf_params, U, tau, n_steps, key, step_size=1, nsamples=10000, burn_in=100, target_acceptance_rate=0.23):
    key, subkey = random.split(key)
    n_sites = L * L
    currentsigma = random.choice(key, jnp.array([-1, 1]), shape=(n_steps, n_sites))
    currentsigmaprime = random.choice(subkey, jnp.array([-1, 1]), shape=(n_steps, n_sites))
    hmf, V = ham
    current_psi_alpha = psi_alpha.copy()
    current_psi_beta = psi_beta.copy()
    current_psi_alpha_prime = psi_alpha.copy()
    current_psi_beta_prime = psi_beta.copy()
    current_psi_alpha, current_psi_beta = propagate(n_steps, currentsigma, current_psi_alpha, current_psi_beta, hmf_params, U, tau)
    current_psi_alpha_prime, current_psi_beta_prime = propagate(n_steps, currentsigmaprime, current_psi_alpha, current_psi_beta, hmf_params, U, tau)
    current_magnitude, current_phase = get_overlap(current_psi_alpha, current_psi_beta, current_psi_alpha_prime, current_psi_beta_prime)
    samples, magnitudes, phases, energies = [], [], [], []
    total_steps = 0
    accepted_steps = 0
    burn_in_accepted = 0

    while accepted_steps < nsamples:
        key, subkey = random.split(key)
        proposedsigma = switcher(key, currentsigma, step_size)
        proposedsigmaprime = switcher(subkey, currentsigmaprime, step_size)
        proposed_psi_alpha = psi_alpha.copy()
        proposed_psi_beta = psi_beta.copy()
        proposed_psi_alpha_prime = psi_alpha.copy()
        proposed_psi_beta_prime = psi_beta.copy()
        proposed_psi_alpha, proposed_psi_beta = propagate(n_steps, proposedsigma, proposed_psi_alpha, proposed_psi_beta, hmf_params, U, tau)
        proposed_psi_alpha_prime, proposed_psi_beta_prime = propagate(n_steps, proposedsigmaprime, proposed_psi_alpha, proposed_psi_beta, hmf_params, U, tau)
        proposed_magnitude, proposed_phase = get_overlap(proposed_psi_alpha, proposed_psi_beta, proposed_psi_alpha_prime, proposed_psi_beta_prime)
        accept = proposed_magnitude / current_magnitude
        key, subkey = random.split(key)
        u = random.uniform(subkey)
        if accept > u:
            currentsigma = proposedsigma
            currentsigmaprime = proposedsigmaprime
            current_magnitude = proposed_magnitude
            current_phase = proposed_phase
            if burn_in > 0:
                burn_in_accepted += 1
                burn_in -= 1
            else:
                samples.append((currentsigma, currentsigmaprime))
                magnitudes.append(current_magnitude)
                phases.append(current_phase)
                current_energy = get_local_energy(hmf, V, proposed_psi_alpha, proposed_psi_beta, proposed_psi_alpha_prime, proposed_psi_beta_prime)
                energies.append(current_energy)
                accepted_steps += 1

        total_steps += 1
        # Adjust step size throughout the sampling process
        acceptance_rate = (burn_in_accepted + accepted_steps) / total_steps
        if acceptance_rate < target_acceptance_rate:
            step_size *= 0.9# Decay step size faster if acceptance rate is low
        else:
            step_size *= 1.1  # Increase step size if acceptance rate is high
        step_size = min(step_size, 1.0)  # Ensure step size does not exceed 1
        step_size = max(step_size, 0.1)  # Ensure step size does not go below 0.01
    print("Sampling completed.")
    print("step size", step_size)
    print("Acceptance rate", (accepted_steps + burn_in_accepted) / total_steps)
    return jnp.array(samples), jnp.array(magnitudes), jnp.array(phases), jnp.array(energies)


def get_variational_energy(dt, alpha, psi_alpha, psi_beta, hmf, V, tau, L, nalpha, nbeta, t, mu, U, key_manager):
        # psi_alpha_params, psi_beta_params, dt_params, t_params, mu_params, U_params = unpack(x, (L * L, nalpha), (L * L, nbeta))
      # print(alpha)
      ham = hmf, V
      n_steps = int(tau / dt) + 1
      hmf_params = hmf + alpha
      key = key_manager.get_key()
      ax, mags, phases, energies = met_hasting_sample(L, psi_alpha, psi_beta, ham, hmf_params, U, tau, n_steps, key)
      #print("auxilary fields", ax[:5])
      #print("magnitudes", mags[:5])
      # multiply magnitudes
      acc = jnp.sum(phases * energies)
      avg_ovlp = jnp.sum(phases)
      print("mags", mags[:5])
      print("energies", energies[:100])
      #print("phases", phases[:5])
     # print("Average sign", jnp.real(avg_ovlp))
      variational_energy = acc / avg_ovlp
      print("var energy", jnp.real(variational_energy))
      return jnp.real(variational_energy), jnp.var(energies)

def objective_function(params, psi_alpha, psi_beta, hmf, V, tau, L, nalpha, nbeta, t, mu, U, key_manager):
    dt, alpha = params[0], params[1:]
    alpha = alpha.reshape((L * L, L * L))
    energy, _ = get_variational_energy(dt, alpha, psi_alpha, psi_beta, hmf, V, tau, L, nalpha, nbeta, t, mu, U, key_manager)
    return energy

def gradient(params, psi_alpha, psi_beta, hmf, V, tau, L, nalpha, nbeta, t, mu, U, key_manager):
    grad = jax.grad(objective_function)(params, psi_alpha, psi_beta, hmf, V, tau, L, nalpha, nbeta, t, mu, U, key_manager)
    clip_value = 10.0  # You can adjust this value
    grad = jnp.clip(grad, -clip_value, clip_value)
    print("Gradient magnitudes:", jnp.linalg.norm(grad))
    return np.array(grad, dtype=np.float64)

def run(L, nalpha, nbeta, t, mu, U, tau, max_iter=50000, tol=1e-10, disp=True, seed=1222):
    n = np.maximum(int(tau * U / 0.4 - 1), 1)
    dt = tau / n
    psi_alpha, psi_beta, hmf, V = initialize(L, nalpha, nbeta, t, mu, U)

    # Instantiate KeyManager with a seed
    key_manager = KeyManager(seed)
    key = key_manager.get_key()
    alpha = jnp.zeros_like(hmf)# random.normal(key, hmf.shape) * 0.1
    eval, evec = jnp.linalg.eigh(hmf + alpha)
    psi_alpha = evec[:, :nalpha]
    psi_beta = evec[:, :nbeta]
    #alpha = alpha.flatten()
   # params = jnp.concatenate([jnp.array([dt]), alpha])
    opt_energy, opt_var = get_variational_energy(dt, alpha, psi_alpha, psi_beta, hmf, V, tau, L, nalpha, nbeta, t, mu, U, key_manager)
    return opt_energy, opt_var


L = 12
nalpha = 12
nbeta = 12
t = 1.0
mu = 0
U = 8.0
tau = 1.25
#psi_alpha, psi_beta, hmf, V = initialize(L, nalpha, nbeta, t, mu, U)
#eval, _ = jnp.linalg.eigh(pspace(hmf, V, L * L, nalpha, nbeta)[1])
print('Eigenvalues', eval)

#energy, var = run(L, nalpha, nbeta,t ,mu , U, tau)
print("Energy", energy)
print("Variance", var)
