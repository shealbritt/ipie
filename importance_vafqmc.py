#MY VERSION THAT IM HAPPY WITH BUT STEALS SOME TONG
from pyscf import tools, lo, scf, fci, gto, cc
import numpy as np
import scipy
import logging
import h5py
from ipie.walkers.pop_controller import PopController
from ipie.walkers.base_walkers import BaseWalkers
from ipie.qmc.comm import FakeComm
import matplotlib.pyplot as plt
import jax
import jax.random as random
import jax.numpy as jnp
import jax.scipy as jsp
jax.config.update('jax_enable_x64', True)

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
            self.tensor_alpha = jnp.array(xinv.dot(self.mo_coeffs[0][:, :self.mol.nelec[0]]))
            if self.mol.nelec[1] > 0:
                self.tensor_beta = jnp.array(xinv.dot(self.mo_coeffs[1][:, :self.mol.nelec[1]]))
            else:
                self.tensor_beta = jnp.zeros((self.mo_coeffs[1].shape[0], 0))
        else:  # Handling RHF and RCCSD
            self.tensor_alpha = jnp.array(xinv.dot(self.mo_coeffs[:, :self.mol.nelec[0]]))
            self.tensor_beta = jnp.array(self.tensor_alpha)  # Same as alpha for RHF
        with h5py.File("input.h5", "w") as fa:
            fa["ao_coeff"] = ao_coeff
            fa["xinv"] = xinv
            fa["phia0_alpha"] = self.tensor_alpha
            fa["phia0_beta"] = self.tensor_beta

class JaxWalkers(BaseWalkers):
    def __init__(self, mol, nwalkers):
        super().__init__(nwalkers)
        self.mol = mol
        self.nwalkers = nwalkers
        self.tensor_alpha = None
        self.tensor_beta = None

    def init_walkers(self, init_trial_alpha, init_trial_beta):
        self.tensor_alpha = jnp.array([init_trial_alpha] * self.nwalkers, dtype=jnp.complex128)
        self.tensor_beta = jnp.array([init_trial_beta] * self.nwalkers, dtype=jnp.complex128)
        self.weight = jnp.ones(self.nwalkers)
        self.buff_names += ['tensor_alpha',  'tensor_beta']
        self.buff_size = round(self.set_buff_size_single_walker() / float(self.nwalkers))
        self.walker_buffer = jnp.zeros((self.buff_size,), dtype = jnp.complex128)
    
    def reortho(self):
        pass

    def reortho_batched(self):  # gpu version
        pass    

class Propagator(object):

    def __init__(self, mol, dt, total_t, stab_freq=5, seed=47193717, nwalkers=100, trial_type = "uhf", scheme = "hybrid energy"):
        self.mol = mol
        self.dt = dt
        self.key_manager = KeyManager(np.random.randint(1000000))
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
        return jnp.array(h1e), jnp.array(v2e), nuc

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
        
        tensor_list = jnp.array(tensor_list)
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

        return jnp.array(h1e), jnp.array(eri), jnp.array(nuc), jnp.array(l_tensor)

    def initialize_simulation(self):
        #Checked
        self.trial = Trial(self.mol, self.trial_type)
        self.trial.get_trial()

    def get_overlap(self):
        #Checked
        overlap_alpha = jnp.linalg.det(jnp.einsum('pr, zpq -> zrq', self.trial.tensor_alpha.conj(), self.walkers.tensor_alpha))
        overlap_beta = jnp.linalg.det(jnp.einsum('pr, zpq -> zrq', self.trial.tensor_beta.conj(), self.walkers.tensor_beta))
        return overlap_alpha * overlap_beta
    
    def get_theta(self):
        #Checked
        inv_ovlp_alpha = jnp.linalg.inv(jnp.einsum('pr, zpq -> zrq', self.trial.tensor_alpha.conj(), self.walkers.tensor_alpha))
        inv_ovlp_beta = jnp.linalg.inv(jnp.einsum('pr, zpq -> zrq', self.trial.tensor_beta.conj(), self.walkers.tensor_beta))
        theta_alpha = jnp.einsum('zpr, zrq -> zpq', self.walkers.tensor_alpha, inv_ovlp_alpha)
        theta_beta = jnp.einsum('zpr, zrq -> zpq', self.walkers.tensor_beta, inv_ovlp_beta)
        #print("Theta_alpha", theta_alpha[0])
        #print("Theta_beta", theta_beta[0])
        return theta_alpha, theta_beta
    
    def get_l_theta_trace(self, modified_l_tensor_alpha, modified_l_tensor_beta, theta_alpha, theta_beta):
        # Checked
        l_theta_alpha = jnp.einsum('npq, mqr -> mnpr', modified_l_tensor_alpha, theta_alpha) #m is numwalkers n is num fields
        l_theta_beta = jnp.einsum('npq, mqr -> mnpr', modified_l_tensor_beta, theta_beta)
        l_theta_trace_alpha = jnp.einsum('mnkk -> mn', l_theta_alpha)
        l_theta_trace_beta = jnp.einsum('mnkk -> mn', l_theta_beta)
        return l_theta_trace_alpha + l_theta_trace_beta
        
    def get_force_bias(self, l_theta_trace):
        force_bias = -1j * jnp.sqrt(self.dt) * l_theta_trace
        #print("Shape of Force_bias", force_bias.shape)
        return force_bias
    
    def propagate_walkers(self, h1e, xbar, l_tensor):
        v0 = jnp.zeros(h1e.shape)
        temp = jnp.einsum('npr, nrq -> pq', l_tensor, l_tensor)
        for p in range(h1e.shape[0]):
            for q in range(h1e.shape[0]):
                v0 = v0.at[p, q].set(h1e[p, q] - 0.5 * temp[p, q])
        key = self.key_manager.get_key()
        xi = random.normal(key, shape =(self.nwalkers, self.nfields))
        #print("force_bias", xbar[0])
        #print("force _bias  subtracted", (xi-xbar)[0])
        #print("L tensors", l_tensor)
        mat = jnp.einsum('nm, mpq -> npq', xi - xbar, l_tensor)
        v0_broadcast = v0[jnp.newaxis, :, :]
        #print("v0_broadcast", v0_broadcast)
        matrixA = 1j*jnp.sqrt(self.dt)*mat-self.dt*v0_broadcast
        tempa = self.walkers.tensor_alpha.copy()
        tempb = self.walkers.tensor_beta.copy()
        expA = jsp.linalg.expm(matrixA)
        #print("Shape of expA", expA.shape)
        #print("expA", expA[0])
        #print("Before prop ", tempa[0])
        self.walkers.tensor_alpha = jnp.einsum('npq, nqr -> npr', expA, tempa)
        self.walkers.tensor_beta = jnp.einsum('npq, nqr -> npr', expA, tempb)


    def get_green_func(self):
        #Checked
        inv_ovlp_alpha = jnp.linalg.inv(jnp.einsum('pr, zpq -> zrq', self.trial.tensor_alpha.conj(), self.walkers.tensor_alpha))
        inv_ovlp_beta = jnp.linalg.inv(jnp.einsum('pr, zpq -> zrq', self.trial.tensor_beta.conj(), self.walkers.tensor_beta))
        intermediate_alpha = jnp.einsum('ijk, ikl -> ijl', self.walkers.tensor_alpha, inv_ovlp_alpha)
        green_func_alpha = jnp.einsum('ijl, ml -> imj', intermediate_alpha, self.trial.tensor_alpha.conj())
        intermediate_beta = jnp.einsum('ijk, ikl -> ijl', self.walkers.tensor_beta, inv_ovlp_beta)
        green_func_beta = jnp.einsum('ijl, ml -> imj', intermediate_beta, self.trial.tensor_beta.conj())

        return green_func_alpha, green_func_beta
    
    def get_local_energy(self, l_tensor, theta_alpha, theta_beta, h1e, nuc, green_func_alpha, green_func_beta):
        #Checked

        modified_l_tensor_alpha = jnp.einsum('pr, npq -> nrq', self.trial.tensor_alpha.conj(), l_tensor)
        modified_l_tensor_beta = jnp.einsum('pr, npq -> nrq', self.trial.tensor_beta.conj(), l_tensor)
        l_theta_trace = self.get_l_theta_trace(modified_l_tensor_alpha, modified_l_tensor_beta, theta_alpha, theta_beta)
        l_theta_alpha = jnp.einsum('nik, mkj -> mnij', modified_l_tensor_alpha, theta_alpha)
        l_theta_beta = jnp.einsum('nik, mkj -> mnij', modified_l_tensor_beta, theta_beta)
        double_l_theta_alpha = jnp.einsum('mnik, mnkj -> mnij', l_theta_alpha, l_theta_alpha)
        double_l_theta_beta = jnp.einsum('mnik, mnkj -> mnij', l_theta_beta, l_theta_beta)
        double_l_theta_trace = jnp.einsum('mnpp -> mn', double_l_theta_alpha) + jnp.einsum('mnpp -> mn', double_l_theta_beta)
        l_theta_trace_squared = jnp.square(l_theta_trace)
        local_energy_field = l_theta_trace_squared - double_l_theta_trace
        local_e2 = jnp.einsum('ij -> i', local_energy_field)
        local_e1 = jnp.einsum("pq, zpq -> z", h1e, green_func_alpha)
        local_e1 += jnp.einsum("pq, zpq -> z", h1e, green_func_beta)
        local_energy = (local_e1 + 0.5*local_e2 + nuc)
        return local_energy

    def update_walker(self, local_energy, energy, overlap):
        # checked
        new_ovlp = self.get_overlap()
        ovlp_ratio = new_ovlp / overlap
        phase_factor = jnp.array([max(0, jnp.cos(jnp.angle(iovlp))) for iovlp in ovlp_ratio]) 
        importance_func = jnp.exp(-self.dt * (jnp.real(local_energy) - energy)) * phase_factor 
        self.walkers.weight = self.walkers.weight * importance_func
        

    def orthonormalize(self):
        #Checked
        ortho_walkers_alpha = jnp.zeros_like(self.walkers.tensor_alpha)
        ortho_walkers_beta = jnp.zeros_like(self.walkers.tensor_beta)
        for i in range(self.walkers.tensor_alpha.shape[0]):
            ortho_walkers_alpha = ortho_walkers_alpha.at[i].set(jnp.linalg.qr(self.walkers.tensor_alpha[i])[0])
        for i in range(self.walkers.tensor_beta.shape[0]):
            ortho_walkers_beta = ortho_walkers_beta.at[i].set(jnp.linalg.qr(self.walkers.tensor_beta[i])[0])
        self.walkers.tensor_alpha = ortho_walkers_alpha
        self.walkers.tensor_beta = ortho_walkers_beta
    
    def run_afqmc(self, params):
       # comm = FakeComm()
        time = 0
        energy_list = jnp.zeros(int(self.total_t / self.dt) + 1, dtype=jnp.complex128)
        error_list = jnp.zeros(int(self.total_t / self.dt) + 1, dtype=jnp.complex128)
        time_list = jnp.zeros(int(self.total_t / self.dt) + 1)

        #ref_energy = self.trial.energy
        #self.pcontrol = PopController(
         #   self.nwalkers,
          #  1,
           # comm,
            #verbose=0,
        #)
        print(params)
        h1e_params, l_tensor_params = self.unpack_params(params)
        tempa, tempb = self.trial.tensor_alpha.copy(), self.trial.tensor_beta.copy()
        self.walkers = JaxWalkers(self.mol, self.nwalkers)
        self.walkers.init_walkers(tempa, tempb)
        modified_l_tensor_alpha = jnp.einsum('pr, npq -> nrq', self.trial.tensor_alpha.conj(), l_tensor_params)
        modified_l_tensor_beta = jnp.einsum('pr, npq -> nrq', self.trial.tensor_beta.conj(), l_tensor_params)
        for step in range(int(self.total_t / self.dt) + 1):
            time = step * self.dt
            time_list = time_list.at[step].set(time)
            ovlp = self.get_overlap()
            theta_alpha, theta_beta = self.get_theta()
            green_func_alpha, green_func_beta = self.get_green_func()
            local_e = self.get_local_energy(self.l_tensor, theta_alpha, theta_beta, self.h1e, self.nuc, green_func_alpha, green_func_beta)
            energy = jnp.sum(self.walkers.weight * local_e)
            energy = energy / jnp.sum(self.walkers.weight)
            error = jnp.sum(self.walkers.weight**2*(local_e - energy)**2)
            error = error / jnp.sum(self.walkers.weight)**2
            energy_list = energy_list.at[step].set(energy)
            error_list = error_list.at[step].set(jnp.sqrt(error))
            l_theta_trace = self.get_l_theta_trace(modified_l_tensor_alpha, modified_l_tensor_beta, theta_alpha, theta_beta)
            force_bias = self.get_force_bias(l_theta_trace)
            # do we do force bias based on hamiltonian or based on parameters?
            self.propagate_walkers(h1e_params, force_bias, l_tensor_params)
            self.update_walker(local_e, energy, ovlp)
            #if step % self.stab_freq == 0:
             #   self.pcontrol.pop_control(self.walkers, comm)
            #need to implement pop_control
            if step % self.stab_freq == 0:
                self.orthonormalize()
        
        return time_list, energy_list, error_list
    
    def unpack_params(self, params):
        """
        Unpacks the parameter vector into the original h1e and l_tensor structures.
        """
        # Assuming the shapes are known for h1e and l_tensor
        h1e_shape = self.h1e.shape
        l_tensor_shape = self.l_tensor.shape
        
        # Unpack the flattened params into h1e and l_tensor
        h1e_unpacked = params[:h1e_shape[0] * h1e_shape[1]].reshape(h1e_shape)
        l_tensor_unpacked = params[h1e_shape[0] * h1e_shape[1]:].reshape(l_tensor_shape)
        
        return h1e_unpacked, l_tensor_unpacked
    
    def objective_func(self, params):
        # make these arrays not lists i think we use append
        time_list, energy_list, error_list= self.run_afqmc(params)
        lam = 1
        idx = int(0.2 * len(energy_list))
        naive_error = jnp.std(energy_list)
        return jnp.real(energy_list[-1]) + lam * (jnp.real(error_list[-1]))**2

    def gradient(self, params):
        grad = jax.grad(self.objective_func)(params)
        return np.array(grad, dtype=np.float64)

    def run(self, max_iter=50000, tol=1e-10, disp=True, seed=1222):
        self.initialize_simulation()
        h1e, v2e, nuc, l_tensor = self.hamiltonian_integral()
        self.h1e = jnp.array(h1e)
        self.nuc = nuc
        self.l_tensor = jnp.array(l_tensor)
        params = jnp.concatenate([h1e.flatten(), l_tensor.flatten()])
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
        time, energy = self.run_afqmc(opt_params)
        return time, energy

# Define the H2 molecule with PySCF
mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', unit='angstrom')

# Instantiate the Propagator class for the H2 molecule
# Example parameters: time step dt=0.01, total simulation time total_t=1.0
print("Running H2O")
prop = Propagator(mol, dt=0.005, total_t=2, nwalkers=100, trial_type="uhf")

# Run the simulation to get time and energy lists
time_list, energy_list, error_list = prop.run()

print(np.mean(energy_list))
print(np.var(energy_list))
# Plot time vs energy for both propagators
plt.figure(figsize=(8, 6))
plt.errorbar(time_list, energy_list, error_list, label='Importance Propagator', color='r', marker='x')

# Optionally, plot a reference energy line if available
plt.hlines(-1.1372838344885023, xmin=0, xmax=3, color='k', linestyle='--', label='Reference Energy')

# Add labels, title, and legend
plt.xlabel('Time (a.u.)')
plt.ylabel('Energy (Hartree)')
plt.title('Comparison of AFQMC Propagators: JAX vs Importance for H2')
plt.grid(True)
plt.legend()

# Show the plot
plt.show()