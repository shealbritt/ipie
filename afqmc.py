#MY VERSION THAT IM HAPPY WITH BUT STEALS SOME TONG
from pyscf import tools, lo, scf, fci, gto, cc
import numpy as np
import scipy
import logging
import h5py
from ipie.walkers.pop_controller import PopController
from ipie.walkers.base_walkers import BaseWalkers
from ipie.qmc.comm import FakeComm

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

class Trial(object):
    def __init__(self, mol, trial_type="uhf"):
        self.mol = mol
        self.trial_type = trial_type
        self.tensor_alpha = None
        self.tensor_beta = None
        self.mo_coeffs = None
        self.energy = None

    def get_trial(self):
        if self.trial_type in ["uhf", "uccsd"]:
            mf = scf.UHF(self.mol)
            mf.kernel()
            mo1 = mf.stability(external=True)[0] # davidso solver 10 roots
            mf = mf.newton().run(mo1, mf.mo_occ)
        elif self.trial_type in ["rhf", "rccsd"]:
            mf = scf.RHF(self.mol)
        mf.kernel()  # Run SCF calculation
        if 'ccsd' in self.trial_type:
            if self.trial_type == "uccsd":
                print("Running UCCSD using UHF orbitals")
                cc_solver = cc.UCCSD(mf)
            elif self.trial_type == "rccsd":
                print("Running RCCSD using RHF orbitals")
                cc_solver = cc.RCCSD(mf)
            cc_solver.kernel()
            print(f"Original SCF Energy: {mf.e_tot}, Updated CCSD Energy: {cc_solver.e_tot}")
            self.energy = cc_solver.e_tot
        else:
            print(f"Running {self.trial_type} with SCF Energy: {mf.e_tot}")
            self.energy = mf.e_tot

    
        # Process molecular orbitals
        s_mat = self.mol.intor('int1e_ovlp')
        ao_coeff = lo.orth.lowdin(s_mat)
        xinv = np.linalg.inv(ao_coeff)
        self.mo_coeffs = mf.mo_coeff

        if self.trial_type in ["uhf", "uccsd"]:  # Handling UHF and UCCSD
            self.tensor_alpha = xinv.dot(self.mo_coeffs[0][:, :self.mol.nelec[0]])
            if self.mol.nelec[1] > 0:
                self.tensor_beta = xinv.dot(self.mo_coeffs[1][:, :self.mol.nelec[1]])
            else:
                self.tensor_beta = np.zeros((self.mo_coeffs[1].shape[0], 0))
        else:  # Handling RHF and RCCSD
            self.tensor_alpha = xinv.dot(self.mo_coeffs[:, :self.mol.nelec[0]])
            self.tensor_beta = self.tensor_alpha  # Same as alpha for RHF
        with h5py.File("input.h5", "w") as fa:
            fa["ao_coeff"] = ao_coeff
            fa["xinv"] = xinv
            fa["phia0_alpha"] = self.tensor_alpha
            fa["phia0_beta"] = self.tensor_beta

class Walkers(BaseWalkers):
    def __init__(self, mol, nwalkers):
        super().__init__(nwalkers)
        self.mol = mol
        self.nwalkers = nwalkers
        self.tensor_alpha = None
        self.tensor_beta = None

    def init_walkers(self, init_trial_alpha, init_trial_beta):
        self.tensor_alpha = np.array([init_trial_alpha] * self.nwalkers, dtype=np.complex128)
        self.tensor_beta = np.array([init_trial_beta] * self.nwalkers, dtype=np.complex128)
        self.weight = np.ones(self.nwalkers)
        self.buff_names += ['tensor_alpha',  'tensor_beta']
        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer = np.zeros((self.buff_size,), dtype = np.complex128)
    
    def reortho(self):
        pass

    def reortho_batched(self):  # gpu version
        pass    

class Propagator(object):

    def __init__(self, mol, dt, total_t, stab_freq=5, seed=47193717, nwalkers=100, trial_type = "uhf", scheme = "hybrid energy"):
        self.mol = mol
        self.dt = dt
        self.nwalkers = nwalkers
        self.total_t = total_t
        self.stab_freq = stab_freq
        self.seed = seed
        self.scheme = scheme
        self.trial_type = trial_type
        self.mf_shift = None
        self.hybrid_energy = None
        self.trial = None
        self.walkers = None
        self.nfields = None
        self.overlap = None

    def read_fcidump(self, fname, norb):
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
            
            for i in range(norb**2):
                if np.sqrt(np.abs(residual[i, i] * max_res)) <= thresh:
                    residual[i, i] = 0
            
            L = residual[:, max_i] / np.sqrt(max_res)
         #   print(f"L vector (reshaped to {norb}x{norb}): {L.reshape(norb, norb)}")
            tensor_list.append(L.reshape(norb, norb))
            
            for i in range(norb**2):
                for j in range(norb**2):
                    products = np.array([L.flatten()[i] * L.flatten()[j] for L in tensor_list])
                    residual[i, j] = v2e[i, j] - np.sum(products)
            
            error = np.linalg.norm(residual)
          #  print(f"Current error: {error}")
            num_fields += 1
        
        tensor_list = np.array(tensor_list)
       # print("Finished modified Cholesky decomposition (NumPy version).")
        return tensor_list, num_fields

    def hamiltonian_integral(self):
        #Checked
        with h5py.File("input.h5") as fa:
            ao_coeff = fa["ao_coeff"][()]
        norb = ao_coeff.shape[0]
        import tempfile
        ftmp = tempfile.NamedTemporaryFile()
        tools.fcidump.from_mo(self.mol, ftmp.name, ao_coeff)
        h1e, eri, nuc = self.read_fcidump(ftmp.name, norb)
        l_tensor, self.nfields = self.modified_cholesky_decomposition(eri)
        with h5py.File("input.h5", "r+") as fa:
            fa["h1e"] = h1e
            fa["nuc"] = nuc
            fa["chol"] = l_tensor

        return h1e, eri, nuc, l_tensor

    def initialize_simulation(self):
        #Checked
        self.trial = Trial(self.mol, self.trial_type)
        self.trial.get_trial()
        tempa, tempb = self.trial.tensor_alpha.copy(), self.trial.tensor_beta.copy()
        self.walkers = Walkers(self.mol, self.nwalkers)
        self.walkers.init_walkers(tempa, tempb)

    def get_overlap(self):
        #Checked
        overlap_alpha = np.linalg.det(np.einsum('pr, zpq -> zrq', self.trial.tensor_alpha.conj(), self.walkers.tensor_alpha))
        overlap_beta = np.linalg.det(np.einsum('pr, zpq -> zrq', self.trial.tensor_beta.conj(), self.walkers.tensor_beta))
        return overlap_alpha * overlap_beta
    
    def get_theta(self):
        #Checked
        inv_ovlp_alpha = np.linalg.inv(np.einsum('pr, zpq -> zrq', self.trial.tensor_alpha.conj(), self.walkers.tensor_alpha))
        inv_ovlp_beta = np.linalg.inv(np.einsum('pr, zpq -> zrq', self.trial.tensor_beta.conj(), self.walkers.tensor_beta))
        theta_alpha = np.einsum('zpr, zrq -> zpq', self.walkers.tensor_alpha, inv_ovlp_alpha)
        theta_beta = np.einsum('zpr, zrq -> zpq', self.walkers.tensor_beta, inv_ovlp_beta)
        #print("Theta_alpha", theta_alpha[0])
        #print("Theta_beta", theta_beta[0])
        return theta_alpha, theta_beta
    
    def get_l_theta_trace(self, modified_l_tensor_alpha, modified_l_tensor_beta, theta_alpha, theta_beta):
        # Checked
        l_theta_alpha = np.einsum('npq, mqr -> mnpr', modified_l_tensor_alpha, theta_alpha) #m is numwalkers n is num fields
        l_theta_beta = np.einsum('npq, mqr -> mnpr', modified_l_tensor_beta, theta_beta)
        l_theta_trace_alpha = np.einsum('mnkk -> mn', l_theta_alpha)
        l_theta_trace_beta = np.einsum('mnkk -> mn', l_theta_beta)
        return l_theta_trace_alpha + l_theta_trace_beta
        
    def get_force_bias(self, l_theta_trace):
        force_bias = -1j * np.sqrt(self.dt) * l_theta_trace
        #print("Shape of Force_bias", force_bias.shape)
        return force_bias
    
    def propagate_walkers(self, h1e, xbar, l_tensor):
        v0 = np.zeros(h1e.shape)
        temp = np.einsum('npr, nrq -> pq', l_tensor, l_tensor)
        for p in range(h1e.shape[0]):
            for q in range(h1e.shape[0]):
                v0[p, q] = h1e[p, q] - 0.5 * temp[p, q]
        print("v0", v0)
        xi = np.random.normal(0.0, 1.0, self.nfields * self.nwalkers).reshape(self.nwalkers, self.nfields)
        print("Auxilary fields", xi[0])
        #print("force_bias", xbar[0])
        #print("force _bias  subtracted", (xi-xbar)[0])
        #print("L tensors", l_tensor)
        mat = np.einsum('nm, mpq -> npq', xi - xbar, l_tensor)
        print("matrix after summing", mat[0])
        v0_broadcast = v0[np.newaxis, :, :]
        #print("v0_broadcast", v0_broadcast)
        matrixA = 1j*np.sqrt(self.dt)*mat-self.dt*v0_broadcast
        print("A matrix", matrixA[0])
        tempa = self.walkers.tensor_alpha.copy()
        tempb = self.walkers.tensor_beta.copy()
        expA = scipy.linalg.expm(matrixA)
        #print("Shape of expA", expA.shape)
        #print("expA", expA[0])
        #print("Before prop ", tempa[0])
        self.walkers.tensor_alpha = np.einsum('npq, nqr -> npr', expA, tempa)
        print("Aftee prop ", self.walkers.tensor_alpha[0])
        
        exit()
        self.walkers.tensor_beta = np.einsum('npq, nqr -> npr', expA, tempb)


    def get_green_func(self):
        #Checked
        inv_ovlp_alpha = np.linalg.inv(np.einsum('pr, zpq -> zrq', self.trial.tensor_alpha.conj(), self.walkers.tensor_alpha))
        inv_ovlp_beta = np.linalg.inv(np.einsum('pr, zpq -> zrq', self.trial.tensor_beta.conj(), self.walkers.tensor_beta))
        intermediate_alpha = np.einsum('ijk, ikl -> ijl', self.walkers.tensor_alpha, inv_ovlp_alpha)
        green_func_alpha = np.einsum('ijl, ml -> imj', intermediate_alpha, self.trial.tensor_alpha.conj())
        intermediate_beta = np.einsum('ijk, ikl -> ijl', self.walkers.tensor_beta, inv_ovlp_beta)
        green_func_beta = np.einsum('ijl, ml -> imj', intermediate_beta, self.trial.tensor_beta.conj())

        return green_func_alpha, green_func_beta
    
    def get_local_energy(self, l_tensor, theta_alpha, theta_beta, v2e, h1e, nuc, green_func_alpha, green_func_beta):
        #Checked

        modified_l_tensor_alpha = np.einsum('pr, npq -> nrq', self.trial.tensor_alpha.conj(), l_tensor)
        modified_l_tensor_beta = np.einsum('pr, npq -> nrq', self.trial.tensor_beta.conj(), l_tensor)
        l_theta_trace = self.get_l_theta_trace(modified_l_tensor_alpha, modified_l_tensor_beta, theta_alpha, theta_beta)
        l_theta_alpha = np.einsum('nik, mkj -> mnij', modified_l_tensor_alpha, theta_alpha)
        l_theta_beta = np.einsum('nik, mkj -> mnij', modified_l_tensor_beta, theta_beta)
        double_l_theta_alpha = np.einsum('mnik, mnkj -> mnij', l_theta_alpha, l_theta_alpha)
        double_l_theta_beta = np.einsum('mnik, mnkj -> mnij', l_theta_beta, l_theta_beta)
        double_l_theta_trace = np.einsum('mnpp -> mn', double_l_theta_alpha) + np.einsum('mnpp -> mn', double_l_theta_beta)
        l_theta_trace_squared = np.square(l_theta_trace)
        local_energy_field = l_theta_trace_squared - double_l_theta_trace
        local_e2 = np.einsum('ij -> i', local_energy_field)
        local_e1 = np.einsum("pq, zpq -> z", h1e, green_func_alpha)
        local_e1 += np.einsum("pq, zpq -> z", h1e, green_func_beta)
        local_energy = (local_e1 + 0.5*local_e2 + nuc)
        return local_energy

    def update_walker(self, local_energy, energy, overlap):
        # checked
        new_ovlp = self.get_overlap()
        ovlp_ratio = new_ovlp / overlap
        phase_factor = np.array([max(0, np.cos(np.angle(iovlp))) for iovlp in ovlp_ratio]) 
        importance_func = np.exp(-self.dt * (np.real(local_energy) - energy)) * phase_factor 
        self.walkers.weight = self.walkers.weight * importance_func
        

    def orthonormalize(self):
        #Checked
        ortho_walkers_alpha = np.zeros_like(self.walkers.tensor_alpha)
        ortho_walkers_beta = np.zeros_like(self.walkers.tensor_beta)
        for i in range(self.walkers.tensor_alpha.shape[0]):
            ortho_walkers_alpha[i] = np.linalg.qr(self.walkers.tensor_alpha[i])[0]
        for i in range(self.walkers.tensor_beta.shape[0]):
            ortho_walkers_beta[i] = np.linalg.qr(self.walkers.tensor_beta[i])[0]
        self.walkers.tensor_alpha = ortho_walkers_alpha
        self.walkers.tensor_beta = ortho_walkers_beta
    
    def run(self):
        comm = FakeComm()
        np.random.seed(self.seed)
        self.initialize_simulation()
        h1e, v2e, nuc, l_tensor = self.hamiltonian_integral()
        time = 0
        energy_list = []
        time_list = []
        ref_energy = self.trial.energy
        self.pcontrol = PopController(
            self.nwalkers,
            1,
            comm,
            verbose=0,
        )
        modified_l_tensor_alpha = np.einsum('pr, npq -> nrq', self.trial.tensor_alpha.conj(), l_tensor)
        modified_l_tensor_beta = np.einsum('pr, npq -> nrq', self.trial.tensor_beta.conj(), l_tensor)
        for step in range(int(self.total_t / self.dt) + 1):
            time = step * self.dt
            time_list.append(time)
            ovlp = self.get_overlap()
            theta_alpha, theta_beta = self.get_theta()
            green_func_alpha, green_func_beta = self.get_green_func()
            local_e = self.get_local_energy(l_tensor, theta_alpha, theta_beta, v2e, h1e, nuc, green_func_alpha, green_func_beta)
            energy = np.sum([self.walkers.weight[i] * local_e[i] for i in range(self.nwalkers)])
            energy = energy / np.sum(self.walkers.weight)
            energy_list.append(energy)
            l_theta_trace = self.get_l_theta_trace(modified_l_tensor_alpha, modified_l_tensor_beta, theta_alpha, theta_beta)
            force_bias = self.get_force_bias(l_theta_trace)
            self.propagate_walkers(h1e, force_bias, l_tensor)
            print(self.walkers.tensor_alpha[:5])
            exit()
            self.update_walker(local_e, energy, ovlp)
            if step % self.stab_freq == 0:
                self.pcontrol.pop_control(self.walkers, comm)

            if step % self.stab_freq == 0:
                self.orthonormalize()
        
        return time_list, energy_list