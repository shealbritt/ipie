from pyscf import tools, lo, scf, fci, gto, cc
import numpy as np
import scipy
import logging
import h5py
import jax.numpy as jnp
import jax.scipy as jsp
import jax
import os
from jaxlib import xla_extension
from numpyro.infer.initialization import init_to_median

from ipie.qmc.comm import FakeComm
import matplotlib.pyplot as plt
from ipie.analysis.autocorr import reblock_by_autocorr
import sys
import multiprocessing
multiprocessing.set_start_method('fork')
import optax
import numpyro
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
    def __init__(self, mol, dt, nsteps, nwalkers=100,
                 taylor_order=6, scheme='local energy',
                 stab_freq=5, seed = 47193717):
        self.mol = mol
        self.nwalkers = nwalkers
        self.nsteps = nsteps
        self.dt = dt
        self.key_manager = KeyManager(seed)
        self.nfields = None
        self.trial = None
        self.walkers = None
        self.precomputed_l_tensora = None
        self.precomputed_l_tensorb = None
        self.taylor_order = taylor_order
        self.hybrid_energy = None
        self.mf_shift = None
        self.scheme = scheme
        self.stab_freq = stab_freq
        self.seed = seed

    def hamiltonian_integral(self):
        # 1e & 2e integrals
        with h5py.File("input.h5") as fa:
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
        with h5py.File("input.h5", "r+") as fa:
            fa["h1e"] = h1e
            fa["nuc"] = nuc
            fa["chol"] = l_tensor

        return jnp.array(h1e, dtype=jnp.complex128), jnp.array(eri,  dtype=jnp.complex128), jnp.array(nuc,  dtype=jnp.complex128), jnp.array(l_tensor,  dtype=jnp.complex128)
    
    def target(self, x):
        x = x.reshape(2, self.nsteps, self.nfields)
        x_l = x[0]
        x_r = x[1]
        l_tensora, l_tensorb = self.propagate(x_l)
        r_tensora, r_tensorb = self.propagate(x_r)     
        wfs = l_tensora, l_tensorb, r_tensora, r_tensorb    
        overlapa, overlapb = self.get_overlap(*wfs)
        overlap = jnp.linalg.det(overlapa) * jnp.linalg.det(overlapb)
        return overlap
    
    def potential_fn(self, x_flat): # NUTS expects a flattened array as input)
        x_flat = x_flat['x_flat']

        overlap = self.target(x_flat)
        is_nan_overlap = jnp.isnan(overlap)
        # Conditionally print only when NOT tracing (during runtime execution)
        if type(is_nan_overlap) == xla_extension.ArrayImpl: # Now you *can* safely use bool() in eager mode
            if(is_nan_overlap == True):
                print('overlap')
                exit()

        log_overlap_magnitude = jnp.log(jnp.abs(overlap))
        potential_energy = -log_overlap_magnitude
        return potential_energy

    def sampler(self, params, num_warmup): # Renamed to differentiate
        params = self.unpack_params(params)
        self.h1e_params, self.l_tensor_params, self.tensora_params, self.tensorb_params, self.t_params, self.s_params = params
        key = self.key_manager.get_key()

        init_x = jax.random.normal(key, (2 * self.nsteps * self.nfields,))
        init_x_reshaped = init_x.reshape(2, self.nsteps, self.nfields)
        init_overlap = self.target(init_x_reshaped)
        init_magnitude = jnp.abs(init_overlap)
        '''while jnp.any(init_magnitude == 0): # Keep trying until we find a non-zero overlap start
            key = self.key_manager.get_key()
            init_x = jax.random.normal(key, (2 * self.nsteps * self.nfields,))
            init_x_reshaped = init_x.reshape(2, self.nsteps, self.nfields)
            init_overlap = self.target(init_x_reshaped)
            init_magnitude = jnp.abs(init_overlap)
'''
        # Create NUTS kernel with the potential_fn
        nuts_kernel = NUTS(potential_fn=self.potential_fn, target_accept_prob=0.4)
        # Run MCMC
         # Split key for MCMC
        num_samples = self.nwalkers # Sample as many as your current walkers
        initial_params = {"x_flat": init_x} # Initial parameters for NUTS, must be a dict if model is None
        key = self.key_manager.get_key()
        mcmc = MCMC(nuts_kernel, num_warmup=num_warmup, num_samples=num_samples)
        mcmc.run(key, init_params=initial_params) # Pass init_params as a dictionary
        samples = mcmc.get_samples()
        x_samples_flat = samples["x_flat"] 
        acceptance_rate = self.acceptance(x_samples_flat)
        return x_samples_flat, acceptance_rate
    
    def get_overlap(self, l_tensora, l_tensorb, r_tensora, r_tensorb):
        ovlpa = jnp.einsum('pr, pq->rq', l_tensora.conj(), r_tensora)
        ovlpb = jnp.einsum('pr, pq->rq', l_tensorb.conj(), r_tensorb)
        is_nan_overlap = jnp.isnan(ovlpa)
        # Conditionally print only when NOT tracing (during runtime execution)
        if type(is_nan_overlap) == xla_extension.ArrayImpl: # Now you *can* safely use bool() in eager mode
            if(is_nan_overlap == True):
                print('overlapa', self.l_tensora)
                exit()
        return ovlpa, ovlpb
    
    def propagate_one_step(self, h1e_mod, xi, l_tensor, tensora, tensorb, t, s):
        # 1-body propagator propagation
        one_body_op_power = jsp.linalg.expm(-t/2 * h1e_mod)
        tensora = jnp.einsum('pq, qr->pr', one_body_op_power, tensora)
        tensorb = jnp.einsum('pq, qr->pr', one_body_op_power, tensorb)
        # 2-body propagator propagation
        # Conditionally print only when NOT tracing (during runtime execution)
        #i think this is - sqrt(s) when s is negative but should be positive sqrt s
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
        one_body_op_power = jsp.linalg.expm(-t/2 * h1e_mod)
        tensora = jnp.einsum('pq, qr->pr', one_body_op_power, tensora)
        tensorb = jnp.einsum('pq, qr->pr', one_body_op_power, tensorb)
        #I Think i need to add probability of the field i chose
        return tensora, tensorb

    def normal_pdf(self, x):
        prob = 1.0
        for i in range(len(x)):
            prob *= jnp.exp(-0.5*x[i]**2)/jnp.sqrt(2*jnp.pi)

        return prob

    def propagate(self, x):
        tensora = self.tensora_params.copy()
        tensorb = self.tensorb_params.copy()
        for i in range(self.nsteps):
            tensora, tensorb = self.propagate_one_step(self.h1e_params[i], x[i], self.l_tensor_params, tensora, tensorb, self.t_params[i], self.s_params[i])
        prob = self.normal_pdf(x.flatten())   
        return prob*tensora, prob*tensorb     
  

    def green_func(self, la, lb, ra, rb):
        ovlpa, ovlpb = self.get_overlap(la, lb, ra, rb)
        ovlp_inva = jnp.linalg.inv(ovlpa)
        ovlp_invb = jnp.linalg.inv(ovlpb)
        thetaa = jnp.einsum("qp, pr->qr", ra, ovlp_inva)
        thetab = jnp.einsum("qp, pr->qr", rb, ovlp_invb)
        green_funca =jnp.einsum("qr, pr->pq", thetaa, la.conj())
        green_funcb = jnp.einsum("qr, pr->pq", thetab, lb.conj())
        return green_funca, green_funcb
    
    def local_energy(self, la, lb, ra, rb):
        green_funca, green_funcb = self.green_func(la, lb, ra, rb)
        local_e2 = jnp.einsum("prqs, pr, qs->", self.v2e, green_funca, green_funca)
        local_e2 += jnp.einsum("prqs, pr, qs->", self.v2e, green_funca, green_funcb)
        local_e2 += jnp.einsum("prqs, pr, qs->", self.v2e, green_funcb, green_funca)
        local_e2 += jnp.einsum("prqs, pr, qs->", self.v2e, green_funcb, green_funcb)
        local_e2 -= jnp.einsum("prqs, ps, qr->", self.v2e, green_funca, green_funca)
        local_e2 -= jnp.einsum("prqs, ps, qr->", self.v2e, green_funcb, green_funcb)
        local_e1 = jnp.einsum("pq, pq->", green_funca, self.h1e)
        local_e1 += jnp.einsum("pq, pq->", green_funcb, self.h1e)
        local_e = (local_e1 + 0.5 * local_e2 + self.nuc)
        return local_e

    def reorthogonal(self,tensora, tensorb):
        ortho_walkersa = jnp.zeros_like(tensora)
        ortho_walkersb = jnp.zeros_like(tensorb)
        ortho_walkersb = jnp.linalg.qr(tensorb)[0]
        ortho_walkersa = jnp.linalg.qr(tensora)[0]
        return ortho_walkersa, ortho_walkersb

    
    def unpack_params(self, params):
        """
        Unpacks the parameter vector into the original h1e, l_tensor, tensora, tensorb, t, and s structures.
        """
        # Assuming the shapes are known
        h1e_shape = self.h1e.shape
        l_tensor_shape = self.l_tensor.shape  # Ensure this is correct, e.g., (4, 2, 2)
        tensora_shape = self.trial.tensora.shape
        tensorb_shape = self.trial.tensorb.shape
        
        # Modify h1e shape to incorporate nsteps as the first dimension
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
        x_l = x[0]
        x_r = x[1]
        l_tensora, l_tensorb = self.propagate(x_l)
        r_tensora, r_tensorb = self.propagate(x_r)
        wfs = l_tensora, l_tensorb, r_tensora, r_tensorb
        overlapa, overlapb = self.get_overlap(*wfs)
        overlap = jnp.linalg.det(overlapa) * jnp.linalg.det(overlapb)
        magnitude = jnp.abs(overlap)
        energy = self.local_energy(*wfs) 
        phase = overlap / magnitude
        return energy * phase, phase

    def acceptance(self, samples):    
        accepted_count = 0
        total_moves = samples.shape[0] - 1

        for i in range(1, samples.shape[0]):
            if not jnp.allclose(samples[i], samples[i-1]):  # Compare samples
                accepted_count += 1
        acceptance_rate = accepted_count / total_moves
        return acceptance_rate
    
    def objective_func(self, params):
        # Run the simulation
        lam = 1
        B = 0.7
        warmup = 2000
        samples, acceptance = jax.lax.stop_gradient(self.sampler(params, warmup))
        vectorized_variational_energy_func = jax.vmap(self.variational_energy, in_axes=0) # Vectorize over the first argument (x)

        # Vectorizsd calculation of energies and phases for all samples
        energies_phases = vectorized_variational_energy_func(samples)  # Call the vectorized function
        energies, phases = energies_phases # Unpack the tuple of arrays
        num = jnp.sum(energies)
        denom = jnp.sum(phases)

        '''
        mol = gto.M(atom='H 0 0 0; H 0 0 1.6', basis='sto-3g', unit='bohr')
        mf = scf.RHF(mol)
        hf_energy = mf.kernel()
        cisolver = fci.FCI(mf)
        fci_energy = cisolver.kernel()[0]
        plt.hlines(fci_energy, xmin=0, xmax=10, color='k', linestyle='--', label='Reference Energy')
        plt.hlines(hf_energy, xmin=0, xmax=10, color = 'r', linestyle='--', label='HF Energy')
        plt.hlines(variational_energy, xmin=0, xmax=10, linestyle=':', label='Variational Energy')
        plt.legend()
        plt.show()
        exit()'''
        return jnp.real(num / denom) + lam *(max(B-jnp.real(jnp.mean(phases)), 0))**2

    def gradient(self, params):
        print("gradient called")
        params = jnp.array(params)
        grad = jax.grad(self.objective_func)(params)
       # print('gradient', grad)
        return np.array(grad, dtype=np.float64)

    def run(self, max_iter=30, tol=1e-5, disp=True, seed=1222):
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
        
        max_iter = 1000
        tol = 1e-6
        learning_rate = 1e-3
        optimizer = optax.adam(learning_rate)
        opt_state = optimizer.init(params)

        def train_step(params, opt_state):
            grads = self.gradient(params)  # Compute gradients
            updates, opt_state = optimizer.update(grads, opt_state)  # Compute updates
            params = optax.apply_updates(params, updates)  # Apply updates
            return params, opt_state

        for i in range(max_iter):
            new_params, opt_state = train_step(params, opt_state)
            
            # Check for convergence
            if jnp.linalg.norm(new_params - params) < tol:
                break
            
            params = new_params

        opt_params = params
        np.save('optimal_params_adams.npy', opt_params)
        '''iterations = range(1, len(energy_history) + 1)

        plt.figure(figsize=(12, 4))

        plt.subplot(1, 3, 1)
        plt.plot(iterations, energy_history)
        plt.xlabel("Iteration")
        plt.ylabel("Variational Energy")
        plt.title("Energy Convergence")

        plt.subplot(1, 3, 2)
        plt.plot(iterations, grad_norm_history)
        plt.xlabel("Iteration")
        plt.ylabel("Gradient Norm")
        plt.yscale('log') # Use log scale for gradient norm
        plt.title("Gradient Norm Convergence")

        plt.subplot(1, 3, 3)
        plt.plot(iterations, param_update_norm_history)
        plt.xlabel("Iteration")
        plt.ylabel("Parameter Update Norm")
        plt.yscale('log') # Use log scale for parameter update norm
        plt.title("Parameter Update Convergence")


        plt.tight_layout()
        plt.show()'''
        warmup = 500
        samples, acceptance = jax.lax.stop_gradient(self.sampler(opt_params, warmup))
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
    
    for time in times:
        
       # prop = JaxPropagator(mol, dt=0.01, total_t=time, nwalkers=100)  
        #time_list, energy = prop.sampler()
        #plt.plot(time_list, energy, label="jax_afqmc")
        prop = Propagator(mol, dt=0.1, nsteps=10, nwalkers=10000)

    # Run the simulation to get time and energy lists

        energy = prop.run()

        plt.hlines(energy,xmin=0, xmax=10, linestyle=':', label="vafqmc")



    #plt.errorbar(time_list, np.real(energy_list), np.real(error_list), label='Importance Propagator', color='r', marker='x')
    mf = scf.RHF(mol)
    hf_energy = mf.kernel()
    cisolver = fci.FCI(mf)
    fci_energy = cisolver.kernel()[0]
    print("fci-energy",fci_energy)
    print("vafqmc-energy", energy)
    # Optionally, plot a reference energy line if available
    plt.hlines(fci_energy, xmin=0, xmax=10, color='k', linestyle='--', label='Reference Energy')

    # Add labels, title, and legend
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Energy (Hartree)')
    plt.title('Comparison of AFQMC Propagators: JAX vs VAFQMC for H2')
    plt.grid(True)
    plt.legend()

    # Show the plot
    plt.savrefig("h2-20walkers.png")
