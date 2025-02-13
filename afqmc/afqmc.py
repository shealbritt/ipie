#MY VERSION THAT IM HAPPY WITH BUT STEALS SOME TONG
from pyscf import tools, lo, scf, fci, gto, cc
import numpy as np
import scipy
import logging
import h5py
import jax.numpy as jnp
import jax
from ipie.walkers.pop_controller import PopController
from ipie.qmc.comm import FakeComm
import matplotlib.pyplot as plt
from ipie.analysis.autocorr import reblock_by_autocorr
from trial import Trial
from walkers import Walkers
from utils import read_fcidump, get_fci

 

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
    def __init__(self, mol, dt, total_t, nwalkers=100,
                 taylor_order=6, scheme='local energy',
                 stab_freq=5, seed = 47193717):
        self.mol = mol
        self.nwalkers = nwalkers
        self.total_t = total_t
        self.dt = dt
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

        return h1e, eri, nuc, l_tensor

    def initialize_simulation(self):
        self.trial = Trial(self.mol)
        self.trial.get_trial()
        tempa, tempb = self.trial.triala.copy(), self.trial.trialb.copy()
        self.walkers = Walkers(self.mol, self.nwalkers)
        self.walkers.init_walkers(tempa, tempb)

    def simulate_afqmc(self):
        np.random.seed(self.seed)
        self.initialize_simulation()
        h1e, v2e, nuc, l_tensor = self.hamiltonian_integral()
        h1e_mod = np.zeros(h1e.shape)
        Gmfa = self.trial.triala.dot(self.trial.triala.T.conj())
        Gmfb = self.trial.trialb.dot(self.trial.trialb.T.conj())
        self.mf_shift = 0.5 * (1j * np.einsum("npq,pq->n", l_tensor, Gmfa) + 1j * np.einsum("npq,pq->n", l_tensor, Gmfb))
        temp = np.einsum('npr, nrq -> pq', l_tensor, l_tensor)
        for p, q in itertools.product(range(h1e.shape[0]), repeat=2):
            h1e_mod[p, q] = h1e[p, q] - 0.5 * temp[p,q]
        
        h1e_mod = h1e_mod - np.einsum("n, npq->pq", self.mf_shift, 1j*l_tensor)
        self.precomputed_l_tensora = np.einsum("pr, npq->nrq", self.trial.triala.conj(), l_tensor)
        self.precomputed_l_tensorb = np.einsum("pr, npq->nrq", self.trial.trialb.conj(), l_tensor)
        time = 0
        energy_list = []
        time_list = []
        while time <= self.total_t:
            print(f"time: {time}")
            time_list.append(time)
            # tensors preparation
            ovlpa, ovlpb = self.get_overlap()
            ovlp_inva = np.linalg.inv(ovlpa)
            ovlp_invb = np.linalg.inv(ovlpb)
            thetaa = np.einsum("zqp, zpr->zqr", self.walkers.walker_tensora, ovlp_inva)
            thetab = np.einsum("zqp, zpr->zqr", self.walkers.walker_tensorb, ovlp_invb)
            green_funca = np.einsum("zqr, pr->zpq", thetaa, self.trial.triala.conj())
            green_funcb = np.einsum("zqr, pr->zpq", thetab, self.trial.trialb.conj())
            l_thetaa = np.einsum('npq, zqr->znpr', self.precomputed_l_tensora, thetaa)
            l_thetab = np.einsum('npq, zqr->znpr', self.precomputed_l_tensorb, thetab)
            trace_l_theta = np.einsum('znpp->zn', l_thetaa)
            trace_l_theta += np.einsum('znpp->zn', l_thetab)
            # calculate the local energy for each walker
            local_e = self.local_energy(h1e, v2e, nuc, green_funca, green_funcb)
            energy = np.sum([self.walkers.walker_weight[i]*local_e[i] for i in range(len(local_e))])
            energy = energy / np.sum(self.walkers.walker_weight)
            energy_list.append(energy)
            # imaginary time propagation
            xbar = -np.sqrt(self.dt) * (1j * trace_l_theta - self.mf_shift)
            cfb, cmf = self.propagate(h1e_mod, xbar, l_tensor)
            self.update_weight(ovlpa, ovlpb, cfb, cmf, local_e, time)
            # periodic re-orthogonalization
            if int(time / self.dt) % self.stab_freq == 0:
                self.reorthogonal()
               # self.pop_control()
            time = time + self.dt
        return time_list, energy_list

    def get_overlap(self):
        ovlpa = np.einsum('pr, zpq->zrq', self.trial.triala.conj(), self.walkers.walker_tensora)
        ovlpb = np.einsum('pr, zpq->zrq', self.trial.trialb.conj(), self.walkers.walker_tensorb)
        return ovlpa, ovlpb

    def propagate(self, h1e_mod, xbar, l_tensor):
        # 1-body propagator propagation
        one_body_op_power = scipy.linalg.expm(-self.dt/2 * h1e_mod)
        self.walkers.walker_tensora = np.einsum('pq, zqr->zpr', one_body_op_power, self.walkers.walker_tensora)
        self.walkers.walker_tensorb = np.einsum('pq, zqr->zpr', one_body_op_power, self.walkers.walker_tensorb)
        # 2-body propagator propagation
        xi = np.random.normal(0.0, 1.0, self.nfields * self.nwalkers)
        xi = xi.reshape(self.nwalkers, self.nfields)
        two_body_op_power = 1j * np.sqrt(self.dt) * np.einsum('zn, npq->zpq', xi-xbar, l_tensor)
        Tempa = self.walkers.walker_tensora.copy()
        Tempb = self.walkers.walker_tensorb.copy()
        for order_i in range(1, 1+self.taylor_order):
            Tempa = np.einsum('zpq, zqr->zpr', two_body_op_power, Tempa) / order_i
            Tempb = np.einsum('zpq, zqr->zpr', two_body_op_power, Tempb) / order_i
            self.walkers.walker_tensora += Tempa
            self.walkers.walker_tensorb += Tempb
        # 1-body propagator propagation
        one_body_op_power = scipy.linalg.expm(-self.dt/2 * h1e_mod)
        self.walkers.walker_tensora = np.einsum('pq, zqr->zpr', one_body_op_power, self.walkers.walker_tensora)
        self.walkers.walker_tensorb = np.einsum('pq, zqr->zpr', one_body_op_power, self.walkers.walker_tensorb)
        # self.walkers.walker_tensorb = np.exp(-self.dt * nuc) * self.walkers.walker_tensorb

        cfb = np.einsum("zn, zn->z", xi, xbar)-0.5*np.einsum("zn, zn->z", xbar, xbar)
        cmf = -np.sqrt(self.dt)*np.einsum('zn, n->z', xi-xbar, self.mf_shift)
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
        self.walkers.walker_tensora = np.array(alpha)
        self.walkers.walker_tensorb= np.array(beta)
        self.walkers.walker_weight = np.array(weights)   
       

    def update_weight(self, ovlpa, ovlpb, cfb, cmf, local_e, time):
        ovlp_newa, ovlp_newb = self.get_overlap()
        # be cautious! power of 2 was neglected before.
        ovlp_ratio = (np.linalg.det(ovlp_newa) * np.linalg.det(ovlp_newb)) / (np.linalg.det(ovlpa) * np.linalg.det(ovlpb))
        # the hybrid energy scheme
        if self.scheme == "hybrid energy":
            self.ebound = (2.0 / self.dt) ** 0.5
            hybrid_energy = -(np.log(ovlp_ratio) + cfb + cmf) / self.dt
            hybrid_energy = np.clip(hybrid_energy.real, a_min=-self.ebound, a_max=self.ebound, out=hybrid_energy.real)
            self.hybrid_energy = hybrid_energy if self.hybrid_energy is None else self.hybrid_energy
            importance_func = np.exp(-self.dt * 0.5 * (hybrid_energy + self.hybrid_energy))
            self.hybrid_energy = hybrid_energy
            phase = (-self.dt * self.hybrid_energy-cfb).imag
            phase_factor = np.array([max(0, np.cos(iphase)) for iphase in phase])
            importance_func = np.abs(importance_func) * phase_factor

        # The local energy formalism
        if self.scheme == "local energy":
            ovlp_ratio = ovlp_ratio * np.exp(cmf)
            phase_factor = np.array([max(0, np.cos(np.angle(iovlp))) for iovlp in ovlp_ratio])
            importance_func = np.exp(-self.dt * np.real(local_e)) * phase_factor
        self.walkers.walker_weight = self.walkers.walker_weight * importance_func
        max_weight = max(self.walkers.walker_weight)
        if max_weight == 0:
            max_weight = 1
        self.walkers.walker_weight = self.walkers.walker_weight / max_weight

    def local_energy(self, h1e, v2e, nuc, green_funca, green_funcb):
        local_e2 =  np.einsum("prqs, zpr, zqs->z", v2e, green_funca, green_funca)
        local_e2 +=  np.einsum("prqs, zpr, zqs->z", v2e, green_funca, green_funcb)
        local_e2 +=  np.einsum("prqs, zpr, zqs->z", v2e, green_funcb, green_funca)
        local_e2 +=  np.einsum("prqs, zpr, zqs->z", v2e, green_funcb, green_funcb)
        local_e2 -=  np.einsum("prqs, zps, zqr->z", v2e, green_funca, green_funca)
        local_e2 -=  np.einsum("prqs, zps, zqr->z", v2e, green_funcb, green_funcb)
        local_e1 =  np.einsum("zpq, pq->z", green_funca, h1e)
        local_e1 +=  np.einsum("zpq, pq->z", green_funcb, h1e)
        local_e = (local_e1 + 0.5 * local_e2 + nuc)
        return local_e

    def reorthogonal(self):
        ortho_walkersa = np.zeros_like(self.walkers.walker_tensora)
        ortho_walkersb = np.zeros_like(self.walkers.walker_tensorb)
        for idx in range(self.walkers.walker_tensorb.shape[0]):
            ortho_walkersb[idx] = np.linalg.qr(self.walkers.walker_tensorb[idx])[0]
            ortho_walkersa[idx] = np.linalg.qr(self.walkers.walker_tensora[idx])[0]
        self.walkers.walker_tensorb = ortho_walkersb
        self.walkers.walker_tensora = ortho_walkersa

