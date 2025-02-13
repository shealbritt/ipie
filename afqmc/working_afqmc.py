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
        self.walker_tensorb = None
        self.walker_weight = None
        self.precomputed_l_tensor = None
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

    def get_trial(self):
        # RHF
        mf = scf.UHF(self.mol)
        mf.kernel()
        s_mat = self.mol.intor('int1e_ovlp')
        ao_coeff = lo.orth.lowdin(s_mat)
        xinv = np.linalg.inv(ao_coeff)

        self.trial = mf.mo_coeff
        self.triala = xinv.dot(mf.mo_coeff[0][:, :self.mol.nelec[0]])
        self.trialb = xinv.dot(mf.mo_coeff[1][:, :self.mol.nelec[1]])
        self.trial = self.trialb

        with h5py.File("input.h5", "w") as fa:
            fa["ao_coeff"] = ao_coeff
            fa["xinv"] = xinv
            fa["phia"] = self.triala
            fa["phib"] = self.trialb

    def init_walker(self):
        self.get_trial()
        tempa = self.trial.copy()
        tempb = self.trialb.copy()
        self.walker_tensora = np.array([tempa] * self.nwalkers, dtype=np.complex128)
        self.walker_tensorb = np.array([tempb] * self.nwalkers, dtype=np.complex128)
        self.walker_weight = np.array([1.] * self.nwalkers)

    def simulate_afqmc(self):
        np.random.seed(self.seed)
        self.init_walker()
        h1e, v2e, nuc, l_tensor = self.hamiltonian_integral()
        h1e_mod = np.zeros(h1e.shape)
        Gmfa = self.triala.dot(self.triala.T.conj())
        Gmfb = self.trialb.dot(self.trialb.T.conj())
        self.mf_shift = 0.5 * (1j * np.einsum("npq,pq->n", l_tensor, Gmfa) + 1j * np.einsum("npq,pq->n", l_tensor, Gmfb))
        for p, q in itertools.product(range(h1e.shape[0]), repeat=2):
            h1e_mod[p, q] = h1e[p, q] - 0.5 * np.trace(v2e[p, :, :, q])
        h1e_mod = h1e_mod - np.einsum("n, npq->pq", self.mf_shift, 1j*l_tensor)
        self.precomputed_l_tensora = np.einsum("pr, npq->nrq", self.triala.conj(), l_tensor)
        self.precomputed_l_tensorb = np.einsum("pr, npq->nrq", self.trialb.conj(), l_tensor)
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
            thetaa = np.einsum("zqp, zpr->zqr", self.walker_tensora, ovlp_inva)
            thetab = np.einsum("zqp, zpr->zqr", self.walker_tensorb, ovlp_invb)
            green_funca = np.einsum("zqr, pr->zpq", thetaa, self.triala.conj())
            green_funcb = np.einsum("zqr, pr->zpq", thetab, self.trialb.conj())
            l_thetaa = np.einsum('npq, zqr->znpr', self.precomputed_l_tensora, thetaa)
            l_thetab = np.einsum('npq, zqr->znpr', self.precomputed_l_tensorb, thetab)
            trace_l_theta = np.einsum('znpp->zn', l_thetaa)
            trace_l_theta += np.einsum('znpp->zn', l_thetab)
            # calculate the local energy for each walker
            local_e = self.local_energy(h1e, v2e, nuc, green_funca, green_funcb)
            energy = np.sum([self.walker_weight[i]*local_e[i] for i in range(len(local_e))])
            energy = energy / np.sum(self.walker_weight)
            energy_list.append(energy)
            # imaginary time propagation
            xbar = -np.sqrt(self.dt) * (1j * trace_l_theta - self.mf_shift)
            cfb, cmf = self.propagate(h1e_mod, xbar, l_tensor)
            self.update_weight(ovlpa, ovlpb, cfb, cmf, local_e, time)
            # periodic re-orthogonalization
            if int(time / self.dt) == self.stab_freq:
                self.reorthogonal()
            time = time + self.dt
        return time_list, energy_list

    def get_overlap(self):
        ovlpa = np.einsum('pr, zpq->zrq', self.triala.conj(), self.walker_tensora)
        ovlpb = np.einsum('pr, zpq->zrq', self.trialb.conj(), self.walker_tensorb)
        return ovlpa, ovlpb

    def propagate(self, h1e_mod, xbar, l_tensor):
        # 1-body propagator propagation
        one_body_op_power = scipy.linalg.expm(-self.dt/2 * h1e_mod)
        self.walker_tensora = np.einsum('pq, zqr->zpr', one_body_op_power, self.walker_tensora)
        self.walker_tensorb = np.einsum('pq, zqr->zpr', one_body_op_power, self.walker_tensorb)
        # 2-body propagator propagation
        xi = np.random.normal(0.0, 1.0, self.nfields * self.nwalkers)
        xi = xi.reshape(self.nwalkers, self.nfields)
        two_body_op_power = 1j * np.sqrt(self.dt) * np.einsum('zn, npq->zpq', xi-xbar, l_tensor)
        Tempa = self.walker_tensora.copy()
        Tempb = self.walker_tensorb.copy()
        for order_i in range(1, 1+self.taylor_order):
            Tempa = np.einsum('zpq, zqr->zpr', two_body_op_power, Tempa) / order_i
            Tempb = np.einsum('zpq, zqr->zpr', two_body_op_power, Tempb) / order_i
            self.walker_tensora += Tempa
            self.walker_tensorb += Tempb
        # 1-body propagator propagation
        one_body_op_power = scipy.linalg.expm(-self.dt/2 * h1e_mod)
        self.walker_tensora = np.einsum('pq, zqr->zpr', one_body_op_power, self.walker_tensora)
        self.walker_tensorb = np.einsum('pq, zqr->zpr', one_body_op_power, self.walker_tensorb)
        # self.walker_tensorb = np.exp(-self.dt * nuc) * self.walker_tensorb

        cfb = np.einsum("zn, zn->z", xi, xbar)-0.5*np.einsum("zn, zn->z", xbar, xbar)
        cmf = -np.sqrt(self.dt)*np.einsum('zn, n->z', xi-xbar, self.mf_shift)
        return cfb, cmf

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
        self.walker_weight = self.walker_weight * importance_func

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
        ortho_walkersa = np.zeros_like(self.walker_tensora)
        ortho_walkersb = np.zeros_like(self.walker_tensorb)
        for idx in range(self.walker_tensorb.shape[0]):
            ortho_walkersa[idx] = np.linalg.qr(self.walker_tensora[idx])[0]
            ortho_walkersb[idx] = np.linalg.qr(self.walker_tensorb[idx])[0]
        self.walker_tensora = ortho_walkersa
        self.walker_tensorb = ortho_walkersb

