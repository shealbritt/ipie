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

    def simulate_afqmc(self, params):
        self.initialize_simulation()
        energy_list = jnp.zeros(self.nsteps, dtype=jnp.complex128)
        time_list = jnp.zeros(self.nsteps, dtype=jnp.complex128)
        h1e_params, l_tensor_params = self.unpack_params(params)
        h1e_mod = jnp.zeros(h1e_params.shape)
        Gmfa = self.trial.tensora.dot(self.trial.tensora.T.conj())
        Gmfb = self.trial.tensorb.dot(self.trial.tensorb.T.conj())
        self.mf_shift = 0.5 * (1j * jnp.einsum("npq,pq->n", l_tensor_params, Gmfa) + 1j * jnp.einsum("npq,pq->n", l_tensor_params, Gmfb))
        temp = jnp.einsum('npr, nrq -> pq', l_tensor_params, l_tensor_params)
        h1e_mod = h1e_params - 0.5 * temp[None, :, :]
        h1e_mod = h1e_mod - jnp.einsum("n, npq->pq", self.mf_shift, 1j*l_tensor_params)[None, :, :]
        self.precomputed_l_tensora = jnp.einsum("pr, npq->nrq", self.trial.tensora.conj(), l_tensor_params)
        self.precomputed_l_tensorb = jnp.einsum("pr, npq->nrq", self.trial.tensorb.conj(), l_tensor_params)
        time = 0
        for i in range(len(time_list)):
            time_list = time_list.at[i].set(time)
            # tensors preparation
            #TODO NEED TO CHANGE SO IT IS SYMMETRIC
            #TODO WRITE ONE TO COMPUTE OVERLAPS WITH TRIAL TOO
            ovlpa, ovlpb = self.get_overlap()
            ovlp_inva = jnp.linalg.inv(ovlpa)
            ovlp_invb = jnp.linalg.inv(ovlpb)
            thetaa = jnp.einsum("zqp, zpr->zqr", self.walkers.tensora, ovlp_inva)
            thetab = jnp.einsum("zqp, zpr->zqr", self.walkers.tensorb, ovlp_invb)
            green_funca =jnp.einsum("zqr, pr->zpq", thetaa, self.trial.tensora.conj())
            green_funcb = jnp.einsum("zqr, pr->zpq", thetab, self.trial.tensorb.conj())
            l_thetaa = jnp.einsum('npq, zqr->znpr', self.precomputed_l_tensora, thetaa)
            l_thetab = jnp.einsum('npq, zqr->znpr', self.precomputed_l_tensorb, thetab)
            trace_l_theta = jnp.einsum('znpp->zn', l_thetaa)
            trace_l_theta += jnp.einsum('znpp->zn', l_thetab)
            # calculate the local energy for each walker
            local_e = self.local_energy(self.h1e, self.v2e, self.nuc, green_funca, green_funcb)
            energy = jnp.sum(jnp.array([self.walkers.weight[i] * local_e[i] for i in range(len(local_e))]))
            energy = energy / jnp.sum(self.walkers.weight)
            # WRITE A FUNCTION HERE THAT CALCULATES SYMMETRIC ENERGY
            if (jnp.sum(self.walkers.weight) == 0):
                print("Invalid Denominator")
            energy_list = energy_list.at[i].set(energy)
            # imaginary time propagation
            #TODO KEEP FORCE BIAS THE SAME
            xbar = -jnp.sqrt(self.dt) * (1j * trace_l_theta - self.mf_shift)
            cfb, cmf = self.propagate(h1e_mod[i], xbar, l_tensor_params)
            self.update_weight(ovlpa, ovlpb, cfb, cmf, local_e, time)
            # periodic re-orthogonalization
            if int(time / self.dt) % self.stab_freq == 0:
                self.reorthogonal()
               # self.pop_control()
            time = time + self.dt
        variational_energy = self.variational_energy(self.h1e, self.v2e, self.nuc, local_e)
        return time_list, energy_list, variational_energy

    def variational_energy(self, h1e, v2e, nuc, ga):
        tempa = self.trial.tensora.copy()
        tempb = self.trial.tensorb.copy()
        tensora = np.array([tempa] * self.nwalkers, dtype=np.complex128)
        tensorb = np.array([tempb] * self.nwalkers, dtype=np.complex128)
        ovlpawalkers = jnp.einsum('npr, mpq->nmrq', self.walkers.tensora.conj(), self.walkers.tensora)
        ovlpbwalkers = jnp.einsum('npr, mpq->nmrq', self.walkers.tensorb.conj(), self.walkers.tensorb)
        ovlpawalkers_inv = jnp.linalg.inv(ovlpawalkers)
        ovlpbwalkers_inv = jnp.linalg.inv(ovlpbwalkers)
        thetaa = jnp.einsum("mqp, nmpr->nmqr", self.walkers.tensora, ovlpawalkers_inv)
        thetab = jnp.einsum("mqp, nmpr->nmqr", self.walkers.tensorb, ovlpbwalkers_inv)
        green_funca =jnp.einsum("nmqr, npr->nmpq", thetaa, self.walkers.tensora.conj())
        green_funcb = jnp.einsum("nmqr, npr->nmpq", thetab, self.walkers.tensorb.conj())
        local_e2 = jnp.einsum("prqs, nmpr, nmqs->nm", v2e, green_funca, green_funca)
        local_e2 += jnp.einsum("prqs, nmpr, nmqs->nm", v2e, green_funca, green_funcb)
        local_e2 += jnp.einsum("prqs, nmpr, nmqs->nm", v2e, green_funcb, green_funca)
        local_e2 += jnp.einsum("prqs, nmpr, nmqs->nm", v2e, green_funcb, green_funcb)
        local_e2 -= jnp.einsum("prqs, nmps, nmqr->nm", v2e, green_funca, green_funca)
        local_e2 -= jnp.einsum("prqs, nmps, nmqr->nm", v2e, green_funcb, green_funcb)
        local_e1 = jnp.einsum("nmpq, pq->nm", green_funca, h1e)
        local_e1 += jnp.einsum("nmpq, pq->nm", green_funcb, h1e)
        local_e = (local_e1 + 0.5 * local_e2 + nuc)
        ovlpa = jnp.linalg.det(ovlpawalkers)
        ovlpb = jnp.linalg.det(ovlpbwalkers)
        totalovlp = ovlpa * ovlpb
        ones = jnp.ones_like(self.walkers.weight)
        coeffs = jnp.einsum("n, m -> nm", self.walkers.weight.conj(), self.walkers.weight)
        print("energies", local_e)
        print("Coefficients",coeffs)
        variational_energy = jnp.einsum("nm, nm -> nm", coeffs, local_e)
        variational_energy = jnp.einsum("nm ->",variational_energy) / jnp.sum(coeffs)
        return variational_energy

    def get_overlap(self):
        ovlpa = jnp.einsum('pr, zpq->zrq', self.trial.tensora.conj(), self.walkers.tensora)
        ovlpb = jnp.einsum('pr, zpq->zrq', self.trial.tensorb.conj(), self.walkers.tensorb)
        return ovlpa, ovlpb

    def propagate(self, h1e_mod, xbar, l_tensor):
        # 1-body propagator propagation
        one_body_op_power = jsp.linalg.expm(-self.dt/2 * h1e_mod)
        self.walkers.tensora = jnp.einsum('pq, zqr->zpr', one_body_op_power, self.walkers.tensora)
        self.walkers.tensorb = jnp.einsum('pq, zqr->zpr', one_body_op_power, self.walkers.tensorb)
        # 2-body propagator propagation
        key = self.key_manager.get_key()
        xi = jax.random.normal(key, (self.nfields * self.nwalkers,))
        xi = xi.reshape(self.nwalkers, self.nfields)
        two_body_op_power = 1j * jnp.sqrt(self.dt) * jnp.einsum('zn, npq->zpq', xi-xbar, l_tensor)
        Tempa = self.walkers.tensora.copy()
        Tempb = self.walkers.tensorb.copy()
        for order_i in range(1, 1+self.taylor_order):
            Tempa = jnp.einsum('zpq, zqr->zpr', two_body_op_power, Tempa) / order_i
            Tempb = jnp.einsum('zpq, zqr->zpr', two_body_op_power, Tempb) / order_i
            self.walkers.tensora += Tempa
            self.walkers.tensorb += Tempb
        # 1-body propagator propagation
        one_body_op_power = jsp.linalg.expm(-self.dt/2 * h1e_mod)
        self.walkers.tensora = jnp.einsum('pq, zqr->zpr', one_body_op_power, self.walkers.tensora)
        self.walkers.tensorb = jnp.einsum('pq, zqr->zpr', one_body_op_power, self.walkers.tensorb)
        cfb = jnp.einsum("zn, zn->z", xi, xbar)-0.5*jnp.einsum("zn, zn->z", xbar, xbar)
        cmf = -jnp.sqrt(self.dt)*jnp.einsum('zn, n->z', xi-xbar, self.mf_shift)
        return cfb, cmf

    def pop_control(self, zeta = 0):
        def stochastic_reconfiguration(walkersalpha, walkersbeta, weights, zeta=0):
            nwalkers = walkersalpha.shape[0]
            cumulative_weights = jnp.cumsum(jnp.abs(weights))
            total_weight = cumulative_weights[-1]
            average_weight = total_weight / nwalkers
            weights = jnp.ones(nwalkers) * average_weight
            #Shea added the line below
            max_weight = max(weights)
            weights = weights / max_weight
            z = total_weight * (jnp.arange(nwalkers) + zeta) / nwalkers
            indices = jax.vmap(jnp.searchsorted, in_axes=(None, 0))(cumulative_weights, z)
            #Shea added the line below 
            #indices = jnp.minimum(indices, nwalkers - 1) 
            walkersalpha = walkersalpha[indices]
            walkersbeta = walkersbeta[indices]
            return walkersalpha, walkersbeta, weights
        alpha = self.walkers.tensora
        beta = self.walkers.tensorb
        weights = self.walkers.weight
        alpha, beta, weights = stochastic_reconfiguration(alpha, beta, weights, zeta)
        self.walkers.tensora = alpha
        self.walkers.tensorb= beta
        self.walkers.weight = weights
       

    def update_weight(self, ovlpa, ovlpb, cfb, cmf, local_e, time):
        ovlp_newa, ovlp_newb = self.get_overlap()
        # be cautious! power of 2 was neglected before.
        ovlp_ratio = (jnp.linalg.det(ovlp_newa) * jnp.linalg.det(ovlp_newb)) / (jnp.linalg.det(ovlpa) * jnp.linalg.det(ovlpb))
        # the hybrid energy scheme
        if self.scheme == "hybrid energy":
            self.ebound = (2.0 / self.dt) ** 0.5
            hybrid_energy = -(jnp.log(ovlp_ratio) + cfb + cmf) / self.dt
            hybrid_energy = jnp.clip(hybrid_energy.real, a_min=-self.ebound, a_max=self.ebound, out=hybrid_energy.real)
            self.hybrid_energy = hybrid_energy if self.hybrid_energy is None else self.hybrid_energy
            importance_func = jnp.exp(-self.dt * 0.5 * (hybrid_energy + self.hybrid_energy))
            self.hybrid_energy = hybrid_energy
            phase = (-self.dt * self.hybrid_energy-cfb).imag
            phase_factor = jnp.array([max(0, jnp.cos(iphase)) for iphase in phase])
            importance_func = jnp.abs(importance_func) * phase_factor

        # The local energy formalism
        if self.scheme == "local energy":
            ovlp_ratio = ovlp_ratio * jnp.exp(cmf)
            phase_factor = jnp.array([max(0, jnp.cos(jnp.angle(iovlp))) for iovlp in ovlp_ratio])
            importance_func = jnp.exp(-self.dt * jnp.real(local_e)) * phase_factor
        self.walkers.weight = self.walkers.weight * importance_func
        max_weight = max(self.walkers.weight)
        if max_weight == 0:
            max_weight = 1
        #self.walkers.weight = self.walkers.weight / max_weight

    def local_energy(self, h1e, v2e, nuc, green_funca, green_funcb):
        local_e2 = jnp.einsum("prqs, zpr, zqs->z", v2e, green_funca, green_funca)
        local_e2 += jnp.einsum("prqs, zpr, zqs->z", v2e, green_funca, green_funcb)
        local_e2 += jnp.einsum("prqs, zpr, zqs->z", v2e, green_funcb, green_funca)
        local_e2 += jnp.einsum("prqs, zpr, zqs->z", v2e, green_funcb, green_funcb)
        local_e2 -= jnp.einsum("prqs, zps, zqr->z", v2e, green_funca, green_funca)
        local_e2 -= jnp.einsum("prqs, zps, zqr->z", v2e, green_funcb, green_funcb)
        local_e1 = jnp.einsum("zpq, pq->z", green_funca, h1e)
        local_e1 += jnp.einsum("zpq, pq->z", green_funcb, h1e)
        local_e = (local_e1 + 0.5 * local_e2 + nuc)
        return local_e

    def reorthogonal(self):
        ortho_walkersa = jnp.zeros_like(self.walkers.tensora)
        ortho_walkersb = jnp.zeros_like(self.walkers.tensorb)
        for idx in range(self.walkers.tensorb.shape[0]):
            ortho_walkersb = ortho_walkersb.at[idx].set(jnp.linalg.qr(self.walkers.tensorb[idx])[0])
            ortho_walkersa = ortho_walkersa.at[idx].set(jnp.linalg.qr(self.walkers.tensora[idx])[0])
        self.walkers.tensorb = ortho_walkersb
        self.walkers.tensora = ortho_walkersa

    
    def unpack_params(self, params):
        """
        Unpacks the parameter vector into the original h1e and l_tensor structures.
        """
        # Assuming the shapes are known for h1e and l_tensor
        h1e_shape = self.h1e.shape
        l_tensor_shape = self.l_tensor.shape
        
        # Modify h1e shape to incorporate nsteps as the first dimension
        h1e_shape_with_time = (self.nsteps, *h1e_shape)  # Adding nsteps as the first dimension
        
        # Unpack the flattened params into h1e and l_tensor
        h1e_unpacked = params[:self.nsteps * h1e_shape[0] * h1e_shape[1]].reshape(h1e_shape_with_time)
        l_tensor_unpacked = params[self.nsteps * h1e_shape[0] * h1e_shape[1]:].reshape(l_tensor_shape)
        
        return h1e_unpacked, l_tensor_unpacked
    
    def objective_func(self, params):
        # Run the simulation
        params = jnp.array(params)
        time_list, energy_list, variational_energy = self.simulate_afqmc(params)
        lam = 3
        plt.plot(time_list, energy_list)
        mol = gto.M(atom='H 0 0 0; H 0 0 1.6', basis='sto-3g', unit='bohr')
        mf = scf.RHF(mol)
        hf_energy = mf.kernel()
        cisolver = fci.FCI(mf)
        fci_energy = cisolver.kernel()[0]
        plt.hlines(fci_energy, xmin=0, xmax=10, color='k', linestyle='--', label='Reference Energy')
        plt.hlines(hf_energy, xmin=0, xmax=10, color = 'r', linestyle='--', label='HF Energy')
        plt.hlines(variational_energy, xmin=0, xmax=10, linestyle='--', label='Variational Energy')
        plt.plot(time_list, energy_list , marker='o',label="prop")
        print(variational_energy)
        plt.legend()
        plt.show()
        exit()
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

        params = jnp.concatenate([h1e_repeated.flatten(), l_tensor.flatten()])
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
        h1e_opt_params, l_tensor_opt_params = self.unpack_params(opt_params)
        time, energy, var_energy = self.simulate_afqmc(opt_params)
        return time, energy
    

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
        prop = Propagator(mol, dt=0.01, nsteps=100, nwalkers=5)

    # Run the simulation to get time and energy lists

        time_list, energy = prop.run()

        plt.plot(time_list, energy, label="vafqmc")



    #plt.errorbar(time_list, np.real(energy_list), np.real(error_list), label='Importance Propagator', color='r', marker='x')
    mf = scf.RHF(mol)
    hf_energy = mf.kernel()
    cisolver = fci.FCI(mf)
    fci_energy = cisolver.kernel()[0]
    print("fci-energy",fci_energy)
    print("vafqmc-energy", energy[-1])
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
