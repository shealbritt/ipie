from pyscf import tools, lo, scf, fci, gto, cc
import numpy as np
import scipy
import logging
import h5py
import jax.numpy as jnp
import jax.scipy as jsp
import jax
import os
from ipie.qmc.comm import FakeComm
import matplotlib.pyplot as plt
from ipie.analysis.autocorr import reblock_by_autocorr
import sys
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
        with h5py.File("input.h5", "r+") as fa:
            fa["h1e"] = h1e
            fa["nuc"] = nuc
            fa["chol"] = l_tensor

        return jnp.array(h1e, dtype=jnp.complex128), jnp.array(eri,  dtype=jnp.complex128), jnp.array(nuc,  dtype=jnp.complex128), jnp.array(l_tensor,  dtype=jnp.complex128)

    def initialize_simulation(self):
        tempa, tempb = self.trial.tensora.copy(), self.trial.tensorb.copy()
        self.walkers = Walkers(self.mol, self.nwalkers)
        self.walkers.init_walkers(tempa, tempb)
        self.walkers.tensora = jnp.array(self.walkers.tensora, dtype=jnp.complex128)
        self.walkers.tensorb = jnp.array(self.walkers.tensorb, dtype=jnp.complex128)
        self.walkers.weight = jnp.array(self.walkers.weight, dtype=jnp.complex128)
    def target(self, x):
        x_l = x[0]
        x_r = x[1]
        self.l_tensora = self.trial.tensora.copy()
        self.l_tensorb = self.trial.tensorb.copy()
        self.r_tensora = self.trial.tensora.copy()
        self.r_tensorb = self.trial.tensorb.copy()
        self.l_tensora, self.l_tensorb = self.propagate(x_l)
        self.r_tensora, self.r_tensorb = self.propagate(x_r)
        overlapa, overlapb = self.get_overlap()
        overlap = jnp.linalg.det(overlapa) * jnp.linalg.det(overlapb)
        return overlap
    
    def simulate_afqmc(self, params):
        self.initialize_simulation()
        self.numerator = 0
        self.denominator = 0
        self.h1e_params, self.l_tensor_params, self.t_params, self.s_params = self.unpack_params(params)
        key = self.key_manager.get_key()
        init_x = jax.random.normal(key, (2 * self.nsteps * self.nfields,))
        init_x = init_x.reshape(2, self.nsteps, self.nfields)
        init_overlap = self.target(init_x)
        init_magnitude = jnp.abs(init_overlap)
        while init_magnitude == 0:
            key = self.key_manager.get_key()
            init_x = jax.random.normal(key, (2 * self.nsteps * self.nfields,))
            init_x = init_x.reshape(2, self.nsteps, self.nfields)
            init_overlap = self.target(init_x)
            init_magnitude = jnp.abs(init_overlap)
        init_energy = self.local_energy() 
        init_phase = init_overlap / init_magnitude
        x = init_x
        magnitude = init_magnitude
        energy = init_energy
        phase = init_phase
        for _ in range(self.nwalkers):
            key = self.key_manager.get_key()
            prop_x = x.flatten() + jax.random.normal(key, (2* self.nsteps * self.nfields,))
            prop_x = prop_x.reshape(2, self.nsteps, self.nfields)
            prop_overlap = self.target(prop_x)
            prop_magnitude = jnp.abs(prop_overlap)
            accept_prob = min(1, prop_magnitude / magnitude)
            key = self.key_manager.get_key()
            u = jax.random.uniform(key)
            if (u <= accept_prob):
                x = prop_x
                magnitude = prop_magnitude
                energy = self.local_energy() 
                phase = prop_overlap / prop_magnitude
                self.numerator += energy * phase
                self.denominator += phase
            else:
                self.numerator += energy * phase
                self.denominator += phase
        return self.numerator / self.denominator

    def get_overlap(self):
        ovlpa = jnp.einsum('pr, pq->rq', self.l_tensora.conj(), self.r_tensora)
        ovlpb = jnp.einsum('pr, pq->rq', self.l_tensorb.conj(), self.r_tensorb)
        return ovlpa, ovlpb

    def propagate_one_step(self, h1e_mod, xi, l_tensor, tensora, tensorb, t, s):
        # 1-body propagator propagation
        one_body_op_power = jsp.linalg.expm(-t/2 * h1e_mod)
        tensora = jnp.einsum('pq, qr->pr', one_body_op_power, tensora)
        tensorb = jnp.einsum('pq, qr->pr', one_body_op_power, tensorb)
        # 2-body propagator propagation
        two_body_op_power = 1j * jnp.sqrt(s) * jnp.einsum('n, npq->pq', xi, l_tensor)
        Tempa = tensora.copy()
        Tempb = tensorb.copy()
        for order_i in range(1, 1+self.taylor_order):
            Tempa = jnp.einsum('pq, qr->pr', two_body_op_power, Tempa) / order_i
            Tempb = jnp.einsum('pq, qr->pr', two_body_op_power, Tempb) / order_i
            self.walkers.tensora += Tempa
            self.walkers.tensorb += Tempb
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
        tensora = self.trial.tensora.copy()
        tensorb = self.trial.tensorb.copy()
        for i in range(self.nsteps):
            tensora, tensorb = self.propagate_one_step(self.h1e_params[i], x[i], self.l_tensor_params, tensora, tensorb, self.t_params[i], self.s_params[i])
        prob = self.normal_pdf(x.flatten())       
        return prob*tensora, prob*tensorb     
  

    def green_func(self):
        ovlpa, ovlpb = self.get_overlap()
        ovlp_inva = jnp.linalg.inv(ovlpa)
        ovlp_invb = jnp.linalg.inv(ovlpb)
        thetaa = jnp.einsum("qp, pr->qr", self.r_tensora, ovlp_inva)
        thetab = jnp.einsum("qp, pr->qr", self.r_tensorb, ovlp_invb)
        green_funca =jnp.einsum("qr, pr->pq", thetaa, self.l_tensora.conj())
        green_funcb = jnp.einsum("qr, pr->pq", thetab, self.l_tensorb.conj())
        return green_funca, green_funcb
    
    def local_energy(self):
        green_funca, green_funcb = self.green_func()
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
        Unpacks the parameter vector into the original h1e and l_tensor structures,
        as well as t and s.
        """
        # Assuming the shapes are known for h1e and l_tensor
        h1e_shape = self.h1e.shape
        l_tensor_shape = self.l_tensor.shape  # Ensure this is correct, e.g., (4, 2, 2)
        
        # Modify h1e shape to incorporate nsteps as the first dimension
        h1e_shape_with_time = (self.nsteps, *h1e_shape)  # Adding nsteps as the first dimension
        
        # Calculate the length of h1e and l_tensor
        h1e_length = self.nsteps * h1e_shape[0] * h1e_shape[1]
        l_tensor_length = l_tensor_shape[0] * l_tensor_shape[1] * l_tensor_shape[2]  # Ensure this matches the total elements

        # Unpack the flattened params into h1e, l_tensor, t, and s
        h1e_unpacked = params[:h1e_length].reshape(h1e_shape_with_time)
        l_tensor_unpacked = params[h1e_length:h1e_length + l_tensor_length].reshape(l_tensor_shape)
        
        # Extract t and s from the remaining params
        t = params[h1e_length + l_tensor_length:h1e_length + l_tensor_length + self.nsteps]
        s = params[h1e_length + l_tensor_length + self.nsteps:]
        
        return h1e_unpacked, l_tensor_unpacked, t, s


    
    def objective_func(self, params):
        # Run the simulation
        params = jnp.array(params)
        variational_energy = self.simulate_afqmc(params)
        lam = 3
        '''mol = gto.M(atom='H 0 0 0; H 0 0 1.6', basis='sto-3g', unit='bohr')
        mf = scf.RHF(mol)
        hf_energy = mf.kernel()
        cisolver = fci.FCI(mf)
        fci_energy = cisolver.kernel()[0]
        plt.hlines(fci_energy, xmin=0, xmax=10, color='k', linestyle='--', label='Reference Energy')
        plt.hlines(hf_energy, xmin=0, xmax=10, color = 'r', linestyle='--', label='HF Energy')
        plt.hlines(variational_energy, xmin=0, xmax=10, linestyle=':', label='Variational Energy')
        plt.show()
        exit()'''
        return jnp.real(variational_energy)

    def gradient(self, params):
        print("gradient called")
        params = jnp.array(params)
        grad = jax.grad(self.objective_func)(params)
        print('gradient', grad)
        return np.array(grad, dtype=np.float64)

    def run(self, max_iter=1000, tol=1e-8, disp=True, seed=1222):
        self.trial = Trial(self.mol)
        self.trial.get_trial()
        self.trial.tensora = jnp.array(self.trial.tensora, dtype=jnp.complex128)
        self.trial.tensorb = jnp.array(self.trial.tensorb, dtype=jnp.complex128)
        self.initialize_simulation()
        h1e, v2e, nuc, l_tensor = self.hamiltonian_integral()
        self.h1e = jnp.array(h1e)
        self.v2e = jnp.array(v2e)
        self.nuc = nuc
        self.l_tensor = jnp.array(l_tensor)
        h1e_repeated = jnp.tile(h1e, (self.nsteps, 1, 1))  # Repeat h1e nsteps times
        t = jnp.array([self.dt] * self.nsteps)
        s = t.copy()
        params = jnp.concatenate([h1e_repeated.flatten(), l_tensor.flatten(), t, s])
        res = scipy.optimize.minimize(
            self.objective_func,
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
        h1e_opt_params, l_tensor_opt_params, t, s = self.unpack_params(opt_params)
        var_energy = self.simulate_afqmc(opt_params)
        return var_energy
    

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
        #time_list, energy = prop.simulate_afqmc()
        #plt.plot(time_list, energy, label="jax_afqmc")
        prop = Propagator(mol, dt=0.1, nsteps=5, nwalkers=10000)

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
    plt.savefig("h2-20walkers.png")
