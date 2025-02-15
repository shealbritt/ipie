from pyscf import tools, lo, scf, fci, gto, cc
import numpy as np
import scipy
import logging
import h5py
import jax.numpy as jnp
import jax.scipy as jsp
import jax
from ipie.walkers.pop_controller import PopController
from ipie.qmc.comm import FakeComm
import matplotlib.pyplot as plt
from ipie.analysis.autocorr import reblock_by_autocorr
from .trial import Trial
from. walkers import Walkers
from .utils import read_fcidump, get_fci
from .keymanager import KeyManager
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



class JaxPropagator(object):
    def __init__(self, mol, dt, total_t, nwalkers=100,
                 taylor_order=6, scheme='local energy',
                 stab_freq=5, seed = 47193717):
        self.mol = mol
        self.nwalkers = nwalkers
        self.total_t = total_t
        self.dt = dt
        self.key_manager = KeyManager(seed)
        self.nfields = None
        self.trial = None
        self.walkers = None
        self.nsteps = int(self.total_t / self.dt) + 1
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

        return jnp.array(h1e), jnp.array(eri), jnp.array(nuc), jnp.array(l_tensor)

    def initialize_simulation(self):
        self.trial = Trial(self.mol)
        self.trial.get_trial()
        self.trial.triala = jnp.array(self.trial.triala)
        self.trial.trialb = jnp.array(self.trial.trialb)
        tempa, tempb = self.trial.triala.copy(), self.trial.trialb.copy()
        self.walkers = Walkers(self.mol, self.nwalkers)
        self.walkers.init_walkers(tempa, tempb)
        self.walkers.walker_tensora = jnp.array(self.walkers.walker_tensora)
        self.walkers.walker_tensorb = jnp.array(self.walkers.walker_tensorb)

    def simulate_afqmc(self):
        self.initialize_simulation()
        h1e, v2e, nuc, l_tensor = self.hamiltonian_integral()
        h1e_mod = jnp.zeros(h1e.shape)
        Gmfa = self.trial.triala.dot(self.trial.triala.T.conj())
        Gmfb = self.trial.trialb.dot(self.trial.trialb.T.conj())
        self.mf_shift = 0.5 * (1j * jnp.einsum("npq,pq->n", l_tensor, Gmfa) + 1j * jnp.einsum("npq,pq->n", l_tensor, Gmfb))
        temp = jnp.einsum('npr, nrq -> pq', l_tensor, l_tensor)
        for p, q in itertools.product(range(h1e.shape[0]), repeat=2):
            h1e_mod = h1e_mod.at[p, q].set(h1e[p, q] - 0.5 * temp[p, q])
        h1e_mod = h1e_mod - jnp.einsum("n, npq->pq", self.mf_shift, 1j*l_tensor)
        self.precomputed_l_tensora = jnp.einsum("pr, npq->nrq", self.trial.triala.conj(), l_tensor)
        self.precomputed_l_tensorb = jnp.einsum("pr, npq->nrq", self.trial.trialb.conj(), l_tensor)
        time = 0
        energy_list = jnp.zeros(self.nsteps)
        time_list = jnp.zeros(self.nsteps)
        for i in range(len(time_list)):
            time_list = time_list.at[i].set(time)
            # tensors preparation
            ovlpa, ovlpb = self.get_overlap()
            ovlp_inva = jnp.linalg.inv(ovlpa)
            ovlp_invb = jnp.linalg.inv(ovlpb)
            thetaa = jnp.einsum("zqp, zpr->zqr", self.walkers.walker_tensora, ovlp_inva)
            thetab = jnp.einsum("zqp, zpr->zqr", self.walkers.walker_tensorb, ovlp_invb)
            green_funca =jnp.einsum("zqr, pr->zpq", thetaa, self.trial.triala.conj())
            green_funcb = jnp.einsum("zqr, pr->zpq", thetab, self.trial.trialb.conj())
            l_thetaa = jnp.einsum('npq, zqr->znpr', self.precomputed_l_tensora, thetaa)
            l_thetab = jnp.einsum('npq, zqr->znpr', self.precomputed_l_tensorb, thetab)
            trace_l_theta = jnp.einsum('znpp->zn', l_thetaa)
            trace_l_theta += jnp.einsum('znpp->zn', l_thetab)
            # calculate the local energy for each walker
            local_e = self.local_energy(h1e, v2e, nuc, green_funca, green_funcb)
            energy = jnp.sum(jnp.array([self.walkers.walker_weight[i] * local_e[i] for i in range(len(local_e))]))
            energy = energy / jnp.sum(self.walkers.walker_weight)
            energy_list = energy_list.at[i].set(energy)
            # imaginary time propagation
            xbar = -jnp.sqrt(self.dt) * (1j * trace_l_theta - self.mf_shift)
            cfb, cmf = self.propagate(h1e_mod, xbar, l_tensor)
            self.update_weight(ovlpa, ovlpb, cfb, cmf, local_e, time)
            # periodic re-orthogonalization
            if int(time / self.dt) % self.stab_freq == 0:
                self.reorthogonal()
                #self.pop_control()
            time = time + self.dt
        return time_list, energy_list

    def get_overlap(self):
        ovlpa = jnp.einsum('pr, zpq->zrq', self.trial.triala.conj(), self.walkers.walker_tensora)
        ovlpb = jnp.einsum('pr, zpq->zrq', self.trial.trialb.conj(), self.walkers.walker_tensorb)
        return ovlpa, ovlpb

    def propagate(self, h1e_mod, xbar, l_tensor):
        # 1-body propagator propagation
        one_body_op_power = jsp.linalg.expm(-self.dt/2 * h1e_mod)
        self.walkers.walker_tensora = jnp.einsum('pq, zqr->zpr', one_body_op_power, self.walkers.walker_tensora)
        self.walkers.walker_tensorb = jnp.einsum('pq, zqr->zpr', one_body_op_power, self.walkers.walker_tensorb)
        # 2-body propagator propagation
        key = self.key_manager.get_key()
        xi = jax.random.normal(key, (self.nfields * self.nwalkers,))
        xi = xi.reshape(self.nwalkers, self.nfields)
        two_body_op_power = 1j * jnp.sqrt(self.dt) * jnp.einsum('zn, npq->zpq', xi-xbar, l_tensor)
        Tempa = self.walkers.walker_tensora.copy()
        Tempb = self.walkers.walker_tensorb.copy()
        for order_i in range(1, 1+self.taylor_order):
            Tempa = jnp.einsum('zpq, zqr->zpr', two_body_op_power, Tempa) / order_i
            Tempb = jnp.einsum('zpq, zqr->zpr', two_body_op_power, Tempb) / order_i
            self.walkers.walker_tensora += Tempa
            self.walkers.walker_tensorb += Tempb
        # 1-body propagator propagation
        one_body_op_power = jsp.linalg.expm(-self.dt/2 * h1e_mod)
        self.walkers.walker_tensora = jnp.einsum('pq, zqr->zpr', one_body_op_power, self.walkers.walker_tensora)
        self.walkers.walker_tensorb = jnp.einsum('pq, zqr->zpr', one_body_op_power, self.walkers.walker_tensorb)

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
            #weights = weights / max_weight
            z = total_weight * (jnp.arange(nwalkers) + zeta) / nwalkers
            indices = jax.vmap(jnp.searchsorted, in_axes=(None, 0))(cumulative_weights, z)
            #Shea added the line below 
            #indices = jnp.minimum(indices, nwalkers - 1) 
            walkersalpha = walkersalpha[indices]
            walkersbeta = walkersbeta[indices]
            return walkersalpha, walkersbeta, weights
        alpha = self.walkers.walker_tensora
        beta = self.walkers.walker_tensorb
        weights = self.walkers.walker_weight
        alpha, beta, weights = stochastic_reconfiguration(alpha, beta, weights, zeta)
        self.walkers.walker_tensora = alpha
        self.walkers.walker_tensorb= beta
        self.walkers.walker_weight = weights
       

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
        self.walkers.walker_weight = self.walkers.walker_weight * importance_func
        max_weight = max(self.walkers.walker_weight)
        if max_weight == 0:
            max_weight = 1
        self.walkers.walker_weight = self.walkers.walker_weight / max_weight

    def local_energy(self, h1e, v2e, nuc, green_funca, green_funcb):
        local_e2 =  jnp.einsum("prqs, zpr, zqs->z", v2e, green_funca, green_funca)
        local_e2 +=  jnp.einsum("prqs, zpr, zqs->z", v2e, green_funca, green_funcb)
        local_e2 +=  jnp.einsum("prqs, zpr, zqs->z", v2e, green_funcb, green_funca)
        local_e2 +=  jnp.einsum("prqs, zpr, zqs->z", v2e, green_funcb, green_funcb)
        local_e2 -= jnp.einsum("prqs, zps, zqr->z", v2e, green_funca, green_funca)
        local_e2 -=  jnp.einsum("prqs, zps, zqr->z", v2e, green_funcb, green_funcb)
        local_e1 =  jnp.einsum("zpq, pq->z", green_funca, h1e)
        local_e1 +=  jnp.einsum("zpq, pq->z", green_funcb, h1e)
        local_e = (local_e1 + 0.5 * local_e2 + nuc)
        return local_e

    def reorthogonal(self):
        ortho_walkersa = jnp.zeros_like(self.walkers.walker_tensora)
        ortho_walkersb = jnp.zeros_like(self.walkers.walker_tensorb)
        for idx in range(self.walkers.walker_tensorb.shape[0]):
            ortho_walkersb = ortho_walkersb.at[idx].set(jnp.linalg.qr(self.walkers.walker_tensorb[idx])[0])
            ortho_walkersa = ortho_walkersa.at[idx].set(jnp.linalg.qr(self.walkers.walker_tensora[idx])[0])
        self.walkers.walker_tensorb = ortho_walkersb
        self.walkers.walker_tensora = ortho_walkersa

