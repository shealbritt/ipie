import numpy as np
import scipy as sp
from pyscf import gto, scf, ao2mo, mcscf, fci, lo
from ipie.systems.generic import Generic
from ipie.legacy.hamiltonians.hubbard import Hubbard
from ipie.legacy.trial_wavefunction.hubbard_uhf import HubbardUHF
import math
import jax
from jax import jit
import jax.numpy as jnp
import jax.scipy as jsp
from jax import random
import matplotlib.pyplot as plt
jax.config.update('jax_enable_x64', True)
from ipie.walkers.pop_controller import PopController
from ipie.walkers.base_walkers import BaseWalkers

class JaxWalkers(BaseWalkers):
    def __init__(self, nwalkers):
        super().__init__(nwalkers)
        self.nwalkers = nwalkers
        self.tensor_alpha = None
        self.tensor_beta = None

    def init_walkers(self, init_trial_alpha, init_trial_beta):
        self.right_tensor_alpha = jnp.array([init_trial_alpha] * self.nwalkers, dtype=jnp.complex128)
        self.right_tensor_beta = jnp.array([init_trial_beta] * self.nwalkers, dtype=jnp.complex128)
        self.left_tensor_alpha = jnp.array([init_trial_alpha] * self.nwalkers, dtype=jnp.complex128)
        self.left_tensor_beta = jnp.array([init_trial_beta] * self.nwalkers, dtype=jnp.complex128)
        self.weight = jnp.zeros(self.nwalkers)
        self.buff_names += ['left_tensor_alpha',  'left_tensor_beta', 'right_tensor_alpha',  'right_tensor_beta']
        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer =jnp.zeros((self.buff_size,), dtype = jnp.complex128)
    
    def reortho(self):
        pass

    def reortho_batched(self):  # gpu version
        pass    

class KeyManager:
    """
    Manages the random number generation keys for JAX.
    Handles the generation and splitting of PRNG keys.
    """
    def __init__(self, seed):
        self.key = random.PRNGKey(seed)
        self.index = 0
    
    def _get_next_key(self):
        """Splits the key and returns a new subkey."""
        new_key, self.key = random.split(self.key)
        return new_key

    def get_key(self):
        """Public method to retrieve a new PRNG key."""
        return self._get_next_key()

class Propagator():
    """
    Class that handles the propagation of wavefunctions using the AFQMC algorithm
    with the Hubbard model. Includes methods for sampling using Metropolis-Hastings.
    """

    def __init__(self, t, U, L, nelec, tau, dt, n_samples):
        """
        Initialize the system's Hamiltonian, interaction parameters, and wavefunctions.

        Args:
            t: Hopping parameter (Hubbard model).
            U: Interaction strength (Hubbard model).
            L: System size (LxL grid).
            nelec: Tuple (n_up, n_down) defining the number of up and down electrons.
            tau: Propagation time step.
            n_samples: Number of Metropolis-Hastings samples.
        """
        self.t = t
        self.U = U
        self.n_sites = L * L
        self.key_manager = KeyManager(np.random.randint(1000000))
        self.L = L
        self.tau = tau
        self.n_samples = n_samples
        self.nup = nelec[0]
        self.ndown = nelec[1]
        self.n_steps = int(tau / dt) + 1
        self.dt = dt
        self.fields = []
        self.lam = jnp.arccosh(jnp.exp(U*self.dt))
        self.psi_alpha = None
        self.psi_beta = None
        self.V = None
        self.hmf = None
        self.hmf_params = None
        key = self.key_manager.get_key()
        self.init_left = self.generate_random_choices(key, 0.5)
        key = self.key_manager.get_key()
        self.init_right = self.generate_random_choices(key, 0.5)
        self.initialize()

    def initialize(self):
        """
        Initializes the Hamiltonian, wavefunctions, and interaction matrices based
        on the Hubbard model parameters and mean-field solutions.
        """
        options = {
        "name": "Hubbard",
        "nup": self.nup,
        "ndown": self.ndown,
        "nx": self.L,
        "ny": self.L,
        "U": self.U,
        "t": self.t,
        "xpbc" : True,
        "ypbc" : True,
        "ueff": self.U
        } 

        system = Generic(nelec=(self.nup, self.ndown))
        ham = Hubbard(options, verbose=True)
        uhf = HubbardUHF(
            system, ham, {"ueff": ham.U}, verbose=True
        )
        mol = gto.M()
        mol.nelectron = self.nup + self.ndown
        mol.spin = self.nup - self.ndown
        mol.verbose = 3

        mol.incore_anyway = True

        n = ham.nbasis
        h1 = ham.T[0].copy()
        eri = np.zeros((n,n,n,n))
        for i in range(n):
            eri[i,i,i,i] = ham.U
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
        mo_coeffs = mf.mo_coeff
        tensor_alpha = mo_coeffs[0][:, :mol.nelec[0]]
        if mol.nelec[1] > 0:
            tensor_beta = mo_coeffs[1][:, :mol.nelec[1]]
        else:
            tensor_beta = np.empty()
        self.hmf = jnp.array(h1)
        self.V = jnp.array(eri)
        self.psi_alpha = jnp.array(tensor_alpha)
        self.psi_beta = jnp.array(tensor_beta)

    def init_walkers(self, psi_alpha, psi_beta):
        self.samples = JaxWalkers(self.n_samples)
        self.samples.init_walkers(psi_alpha, psi_beta)


    def get_overlap(self, wf1_alpha, wf1_beta, wf2_alpha, wf2_beta):
        """
        Calculates the overlap between two wavefunctions (alpha and beta components).
        Args:
            wf1_alpha, wf1_beta: Wavefunctions for the first walker.
            wf2_alpha, wf2_beta: Wavefunctions for the second walker.
        
        Returns:
            magnitude: Magnitude of the overlap.
            phase: Complex phase factor of the overlap.
        """
        ovlp_alpha = jnp.linalg.det(jnp.matmul(wf1_alpha.conj().T, wf2_alpha))
        ovlp_beta = jnp.linalg.det(jnp.matmul(wf1_beta.conj().T, wf2_beta))
        ovlp = ovlp_alpha * ovlp_beta
        magnitude = jnp.abs(ovlp)
        phase = jnp.exp(1j * jnp.angle(ovlp))
        return magnitude, phase

    def get_green_func(self, wf1_alpha, wf1_beta, wf2_alpha, wf2_beta):
        """
        Computes the Green's function for the system using wavefunctions.
        
        Args:
            wf1_alpha, wf1_beta: Wavefunctions for the first walker.
            wf2_alpha, wf2_beta: Wavefunctions for the second walker.
        
        Returns:
            green_func_alpha, green_func_beta: The Green's function matrices for alpha and beta spins.
        """
        inv_ovlp_alpha = jnp.linalg.inv(jnp.einsum('pr, pq -> rq', wf1_alpha.conj(), wf2_alpha))
        inv_ovlp_beta = jnp.linalg.inv(jnp.einsum('pr, pq -> rq', wf1_beta.conj(), wf2_beta))
        intermediate_alpha = jnp.einsum('jk, kl -> jl', wf2_alpha, inv_ovlp_alpha)
        green_func_alpha = jnp.einsum('jl, ml -> mj', intermediate_alpha, wf1_alpha.conj())
        intermediate_beta = jnp.einsum('jk, kl -> jl', wf2_beta, inv_ovlp_beta)
        green_func_beta = jnp.einsum('jl, ml -> mj', intermediate_beta, wf1_beta.conj())
        return green_func_alpha, green_func_beta


    def get_local_energy(self, galpha, gbeta):
        """
        Calculates the local energy based on the Green's functions for alpha and beta components.
        
        Args:
            galpha: Green's function for alpha spins.
            gbeta: Green's function for beta spins.
        
        Returns:
            local_energy: The calculated local energy of the system.
        """
        e1 = jnp.einsum("pq,pq->", galpha, self.hmf)
        e1 += jnp.einsum("pq,pq->", gbeta, self.hmf)
        e2 = jnp.einsum("prqs,pr,qs ->", self.V, galpha, galpha)
        e2 += jnp.einsum("prqs,pr,qs ->", self.V, gbeta, galpha)
        e2 += jnp.einsum("prqs,pr,qs ->", self.V, galpha, gbeta)
        e2 += jnp.einsum("prqs,pr,qs ->", self.V, gbeta, gbeta)
        e2 -= jnp.einsum("prqs,ps,qr ->", self.V, galpha, galpha)
        e2 -= jnp.einsum("prqs,ps,qr ->", self.V, gbeta, gbeta)
        local_energy = e1 + 0.5 * e2
        return local_energy
    
   
    def reorthogonalize(self, wf_alpha, wf_beta):
        """
    Orthogonalizes spin-up and spin-down orbitals separately using Cholesky decomposition.
    
    Parameters:
        psi_up (jnp.ndarray): An Ne x N matrix for spin-up orbitals.
        psi_down (jnp.ndarray): An Ne x N matrix for spin-down orbitals.
        
    Returns:
        tuple: The orthogonalized spin-up and spin-down orbitals.
    """
        # Step 1: Compute the overlap matrix M_up = ψ_up†ψ_up for spin-up orbitals
        overlap_matrix_up = wf_alpha.T.conj() @ wf_alpha
        # Step 2: Perform Cholesky decomposition on the overlap matrix M_up = L_up†L_up
        L_up = jnp.linalg.cholesky(overlap_matrix_up)
        
        # Step 3: Compute the inverse of L_up
        L_inv_up = jnp.linalg.inv(L_up.T.conj())
        # Step 4: Generate orthogonalized spin-up orbitals ψ_up' = ψ_up @ L_inv_up
        psi_up_prime = wf_alpha  @ L_inv_up
        # Step 5: Repeat for spin-down orbitals
        overlap_matrix_down = wf_beta.T.conj() @ wf_beta
        L_down = jnp.linalg.cholesky(overlap_matrix_down)
        L_inv_down = jnp.linalg.inv(L_down.T.conj())
        psi_down_prime = wf_beta @ L_inv_down    
        return psi_up_prime, psi_down_prime


    def propagate_one_step(self, psi_a, psi_b, fields, i):
        two_body_a = jsp.linalg.expm(self.lam * jnp.diag(fields[i]))
        two_body_b = jsp.linalg.expm(self.lam * jnp.diag(-fields[i]))
       # norm = 2**(-self.L * self.L) * jsp.linalg.expm(-self.U * self.dt * jnp.eye(self.L * self.L)/2)
        #two_body_a = jnp.matmul(norm, two_body_a)
        #two_body_b = jnp.matmul(norm, two_body_b)
        newpsi_a = jnp.matmul(self.one_body, jnp.matmul(two_body_a, psi_a.copy()))
        newpsi_b = jnp.matmul(self.one_body, jnp.matmul(two_body_b, psi_b.copy()))
        return newpsi_a, newpsi_b

  
    def generate_random_choices(self, key, frac):
        """
        Generates random binary choices (Ising fields).
        
        Args:
            key: Random number generator key.
            frac: Fraction of ones in the generated array.
        
        Returns:
            Randomly generated Ising field configuration with values -1 and 1.
        """
        bernoulli_samples = jax.random.bernoulli(key, p=frac, shape=(self.n_steps, self.n_sites))
        return 2 * bernoulli_samples - 1


    def switcher(self, mat, site_index, time_step):
        """
        Flip the Ising field at the specified site and time step.
        
        Args:
            mat: The current field configuration matrix.
            site_index: The index of the site to flip.
            time_step: The time step at which the flip is proposed.
            
        Returns:
            mat: The updated field with the proposed flip.
        """
        newmat = mat.copy()
        newmat = newmat.at[time_step, site_index].multiply(-1) 
        return newmat

     # Metropolis hasting
    def sample(self):
        key = self.key_manager.get_key()
        self.init_left = self.generate_random_choices(key, 0.5)
        key = self.key_manager.get_key()
        self.init_right = self.generate_random_choices(key, 0.5)
        self.one_body = jsp.linalg.expm(-self.dt * self.hmf_params)
        # Initial fields for left and right walkers
        fields1 = self.init_left
        fields2 = self.init_right
        alpha = self.psi_alpha.copy()
        beta = self.psi_beta.copy()
        variational_energies = jnp.zeros(self.n_steps)
        variational_errors = jnp.zeros(self.n_steps)
        nums = jnp.zeros(self.n_steps)
        denoms = jnp.ones(self.n_steps)
        ga0, gb0 = self.get_green_func(alpha, beta, alpha, beta)
        variational_energy = self.get_local_energy(ga0,gb0)
        variational_energies = variational_energies.at[0].set(jnp.real(variational_energy))
        nums = nums.at[0].set(jnp.real(variational_energy))
        #Set up original walker:
        og_magnitudes = jnp.ones(self.n_steps)
        og_phases = jnp.ones(self.n_steps, dtype = jnp.complex128)
        og_right_alphas = jnp.array([alpha] * self.n_steps, dtype=jnp.complex128)
        og_right_betas = jnp.array([beta] * self.n_steps, dtype=jnp.complex128)
        og_left_alphas = jnp.array([alpha] * self.n_steps, dtype=jnp.complex128)
        og_left_betas = jnp.array([beta] * self.n_steps, dtype=jnp.complex128)
        og_left_alpha = alpha
        og_left_beta = beta
        og_right_alpha = alpha
        og_right_beta = beta
        for t in range(1, self.n_steps):
            og_left_alpha, og_left_beta = self.propagate_one_step(og_left_alpha, og_left_beta, fields1, t)
            og_right_alpha, og_right_beta = self.propagate_one_step(og_right_alpha, og_right_beta, fields2, t)   
            ogmag, ogphase = self.get_overlap(og_left_alpha, og_left_beta, og_right_alpha, og_right_beta)
            og_magnitudes = og_magnitudes.at[t].set(ogmag)
            og_phases = og_phases.at[t].set(ogphase)
            if t % 5 == 0:
                og_left_alpha, og_left_beta = \
                    self.reorthogonalize(og_left_alpha, og_left_beta)
                og_right_alpha, og_right_beta = \
                    self.reorthogonalize(og_right_alpha, og_right_beta)
            og_left_alphas = og_left_alphas.at[t].set(og_left_alpha)
            og_left_betas = og_left_betas.at[t].set(og_left_beta)
            og_right_alphas = og_right_alphas.at[t].set(og_right_alpha)
            og_right_betas = og_right_betas.at[t].set(og_right_beta)
        for t in range(1, self.n_steps):
            #print(right_beta)
            phases = jnp.ones(self.n_samples, dtype=jnp.complex128)
            energies = jnp.zeros(self.n_samples, dtype=jnp.complex128)
            magnitude = og_magnitudes[t]
            new_left_alpha = og_left_alphas[t]            
            new_right_alpha = og_right_alphas[t]
            new_left_beta = og_left_betas[t]
            new_right_beta = og_right_betas[t]                        
            accepted = 0
            phase = og_phases[t]
            for i in range(self.n_samples):
                key = self.key_manager.get_key()
                randi1 = random.choice(key, jnp.arange(fields1[t].size))
                key = self.key_manager.get_key()
                randi2 = random.choice(key, jnp.arange(fields2[t].size))
                trialfields1 = self.switcher(fields1, randi1, t)
                trialfields2 = self.switcher(fields2, randi2, t)
                trial_left_alpha, trial_left_beta = self.propagate_one_step(og_left_alphas[t-1], og_left_betas[t-1], trialfields1, t)
                trial_right_alpha, trial_right_beta = self.propagate_one_step(og_right_alphas[t-1], og_right_betas[t-1], trialfields2, t)
                trialmagnitude, trialphase = self.get_overlap(trial_left_alpha, trial_left_beta, trial_right_alpha, trial_right_beta)
                accept = trialmagnitude / magnitude
                key = self.key_manager.get_key()
                u = random.uniform(key)
              #  print(magnitude, trialmagnitude,accept, accept > u)
                if accept > u:
                    accepted += 1
                    magnitude = trialmagnitude 
                    phase = trialphase
                    fields1 = trialfields1
                    fields2 = trialfields2
                    new_left_alpha = trial_left_alpha.copy()
                    new_left_beta = trial_left_beta.copy()
                    new_right_alpha = trial_right_alpha.copy()
                    new_right_beta = trial_right_beta.copy()
                galpha, gbeta = self.get_green_func(new_left_alpha, new_left_beta, new_right_alpha, new_right_beta)
                energy = self.get_local_energy(galpha, gbeta)
                phases = phases.at[i].set(phase)
                energies = energies.at[i].set(energy)   
            #Variational energy
           # print(accepted/self.n_samples)
            #print(energies)
            num = phases * energies
            denom = phases
            mean, sigma = self.jackknife_ratios(num, denom)
            variational_energy = jnp.sum(num) / jnp.sum(denom)
            #print(jnp.mean(num))
            #print(jnp.mean(denom))
            variational_energies = variational_energies.at[t].set(jnp.real(variational_energy))
            variational_errors = variational_errors.at[t].set(jnp.real(sigma))
        return variational_energies, variational_errors
    def jackknife_ratios(self, num: jnp.ndarray, denom: jnp.ndarray):
        r"""Jackknife estimation of standard deviation of the ratio of means.

        Parameters
        ----------
        num : :class:`np.ndarray
            Numerator samples.
        denom : :class:`np.ndarray`
            Denominator samples.

        Returns
        -------
        mean : :class:`np.ndarray`
            Ratio of means.
        sigma : :class:`np.ndarray`
            Standard deviation of the ratio of means.
        """
        n_samples = num.size
        num_mean = jnp.mean(num)
        denom_mean = jnp.mean(denom)
        mean = num_mean / denom_mean
        jackknife_estimates = jnp.zeros(n_samples, dtype=num.dtype)
        for i in range(n_samples):
            mean_num_i = (num_mean * n_samples - num[i]) / (n_samples - 1)
            mean_denom_i = (denom_mean * n_samples - denom[i]) / (n_samples - 1)
            jackknife_estimates = jackknife_estimates.at[i].set((mean_num_i / mean_denom_i).real)
        mean = jnp.mean(jackknife_estimates)
        sigma = jnp.sqrt((n_samples - 1) * jnp.var(jackknife_estimates))

        return mean, sigma

    def get_variational_energy(self, params):
            # Reshape the Hamiltonian matrix parameters
            self.hmf_params = self.hmf + params.reshape(self.L * self.L, self.L * self.L)

            num_runs = 1 # Number of Monte Carlo runs
            t = jnp.linspace(0, self.tau, self.n_steps)
            # Setting up the figure and axis for plotting
            plt.figure(figsize=(10, 6))
            # Run the Monte Carlo process and plot each run
            for i in range(num_runs):
                variational_energies, variational_errors = self.sample()
                print(jnp.mean(variational_energies))
                print(jnp.std(variational_energies))
                plt.errorbar(t, variational_energies, variational_errors, label=f'Run {i+1}', alpha=0.7, linewidth=2)
            plt.hlines(-6.681695234496717, xmin=0, xmax=5, label=f'FCI Reference')
            # Adding labels and title to the plot
            plt.xlabel('Time (tau)', fontsize=14)
            plt.ylabel('Variational Energy', fontsize=14)
            plt.title('Variational Energy vs Time for Multiple Runs', fontsize=16)
            # Adding a legend to distinguish different runs
            plt.legend(loc='upper right', fontsize=12)
              
            # Adding grid for better readability
            plt.grid(True, which='both', linestyle='--', linewidth=0.5)

            # Display the final plot
            plt.show()
            
            return 0
 
    def gradient(self, params):
        grad = jax.grad(self.get_variational_energy)(params)
        return np.array(grad, dtype=np.float64)

    def run(self, max_iter=50000, tol=1e-10, disp=True, seed=1222):
        alpha = np.zeros((self.L * self.L, self.L * self.L)) 
        params = jnp.array(alpha.flatten())
        res = sp.optimize.minimize(
            self.get_variational_energy,
            params,
            args=(),
            jac=self.gradient,
            tol=tol,
            method="L-BFGS-B",
            options={
                "maxls": 20,
                "gtol": tol,
                "eps": tol,
                "maxiter": max_iter,
                "ftol": tol,
                "maxcor": 1000,
                "maxfun": max_iter,
                "disp": disp,
            },
          )
        opt_params = res.x
        return res.fun

simple = Propagator(1, 2, 2, (2, 2), 5, 0.05, 100)
simple.run()
