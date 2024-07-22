from pyscf import tools, scf, fci, gto
import numpy as np
from jax import random, numpy as jnp, scipy as jsp, jit
from afqmc import Propagator, Trial
import h5py
import jax
from scipy.optimize import minimize
from jax.scipy.stats import multivariate_normal
jax.config.update('jax_enable_x64', True)

@jit
def ovlp_helper(wf1_alpha, wf1_beta, wf2_alpha, wf2_beta):
        overlap_matrix_alpha = jnp.einsum('pr, pq -> rq', wf1_alpha.conj(), wf2_alpha)
        overlap_matrix_beta = jnp.einsum('pr, pq -> rq', wf1_beta.conj(), wf2_beta)
        overlap_alpha = jnp.linalg.det(overlap_matrix_alpha)
        overlap_beta = jnp.linalg.det(overlap_matrix_beta)
        overlap = overlap_alpha * overlap_beta
        magnitude = jnp.abs(overlap)
        phase = jnp.divide(overlap, magnitude)
        return magnitude, phase

@jit
def green_func_helper(wf1_alpha, wf1_beta, wf2_alpha, wf2_beta):
        inv_ovlp_alpha = jnp.linalg.inv(jnp.einsum('pr, pq -> rq', wf1_alpha.conj(), wf2_alpha))
        inv_ovlp_beta = jnp.linalg.inv(jnp.einsum('pr, pq -> rq', wf1_beta.conj(), wf2_beta))
        intermediate_alpha = jnp.einsum('jk, kl -> jl', wf2_alpha, inv_ovlp_alpha)
        green_func_alpha = jnp.einsum('jl, ml -> mj', intermediate_alpha, wf1_alpha.conj())
        intermediate_beta = jnp.einsum('jk, kl -> jl', wf2_beta, inv_ovlp_beta)
        green_func_beta = jnp.einsum('jl, ml -> mj', intermediate_beta, wf1_beta.conj())
        return green_func_alpha, green_func_beta


@jit
def energy_helper(galpha, gbeta, h1e, v2e, nuc):
        e1 = jnp.einsum("pq, pq ->", galpha, h1e)
        e1 += jnp.einsum("pq, pq ->", gbeta, h1e)
        e2 = jnp.einsum("prqs, pr, qs ->", v2e, galpha, galpha)
        e2 += jnp.einsum("prqs, pr, qs ->", v2e, gbeta, galpha)
        e2 += jnp.einsum("prqs, pr, qs ->", v2e, galpha, gbeta)
        e2 += jnp.einsum("prqs, pr, qs ->", v2e, gbeta, gbeta)
        e2 -= jnp.einsum("prqs, ps, qr ->", v2e, galpha, galpha)
        e2 -= jnp.einsum("prqs, ps, qr ->", v2e, gbeta, gbeta)
        local_energy = nuc + e1 + 0.5 * e2
        return local_energy



class Jax_Propagator(Propagator):
    def __init__(self, mol, dt, total_t, stab_freq=5, seed=4719377, nwalkers=100, trial_type="uhf", scheme="hybrid energy"):
        super().__init__(mol, dt, total_t, stab_freq, seed, nwalkers, trial_type, scheme)
        self.opt_wavefunction = None
        self.nwalkers = 1
        self.nsteps = int(total_t / dt) + 1
        self.key = None

# definitely Correct
    def read_fcidump(self, fname, norb):
        h1e, v2e, nuc = super().read_fcidump(fname, norb)
        return jnp.array(h1e), jnp.array(v2e), jnp.array(nuc)

# definitely Correct
    def hamiltonian_integral(self):
        with h5py.File("input.h5") as fa:
            ao_coeff = fa["ao_ceff"][()]
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
        return h1e, eri, nuc





# definitely Correct

    def initialize_simulation(self):
        self.trial = Trial(self.mol, self.trial_type)
        self.trial.get_trial()
        self.key = random.PRNGKey(self.seed)
        self.key, subkey = random.split(self.key)
        self.trial.tensor_alpha = jnp.array(self.trial.tensor_alpha)
        self.trial.tensor_beta = jnp.array(self.trial.tensor_beta)
        self.shape_psi_alpha = self.trial.tensor_alpha.shape
        self.shape_psi_beta = self.trial.tensor_beta.shape
        self.opt_wavefunction = Trial(self.mol, self.trial_type)
        self.opt_wavefunction.get_trial()
        self.opt_wavefunction.tensor_alpha = jnp.array(self.opt_wavefunction.tensor_alpha) * (1 + 0.01 * random.normal(self.key, self.opt_wavefunction.tensor_alpha.shape))
        self.opt_wavefunction.tensor_beta = jnp.array(self.opt_wavefunction.tensor_beta) * (1 + 0.01 * random.normal(subkey, self.opt_wavefunction.tensor_beta.shape))
        self.shape_h1e = None
        self.shape_v2e = None



  # definitely correct

    def pack(self, psi_alpha, psi_beta, dt, ds, h1e, v2e, nuc) -> jnp.ndarray:
        return jnp.concatenate((psi_alpha.flatten(), psi_beta.flatten(), jnp.array([dt]), jnp.array([ds]), h1e.flatten(), v2e.flatten(), jnp.array([nuc])), dtype=jnp.float64)

  # definitely Correct
    def unpack(self, packed_array: jnp.ndarray):
        idx = 0
        size_psi_alpha = jnp.prod(jnp.array(self.shape_psi_alpha))
        psi_alpha = packed_array[idx:idx + size_psi_alpha].reshape(self.shape_psi_alpha)
        idx += size_psi_alpha
        size_psi_beta = jnp.prod(jnp.array(self.shape_psi_beta))
        psi_beta = packed_array[idx:idx + size_psi_beta].reshape(self.shape_psi_beta)
        idx += size_psi_beta
        dt = packed_array[idx]
        idx += 1
        ds = packed_array[idx]
        idx += 1
        size_h1e = jnp.prod(jnp.array(self.shape_h1e))
        h1e = packed_array[idx:idx + size_h1e].reshape(self.shape_h1e)
        idx += size_h1e
        size_v2e = jnp.prod(jnp.array(self.shape_v2e))
        v2e = packed_array[idx:idx + size_v2e].reshape(self.shape_v2e)
        idx += size_v2e
        nuc = packed_array[idx]
        return jnp.array(psi_alpha), jnp.array(psi_beta), dt, ds, jnp.array(h1e), jnp.array(v2e), nuc



    # definitely Correct
    def get_green_func(self, wf1, wf2):
        return green_func_helper(wf1.tensor_alpha, wf1.tensor_beta, wf2.tensor_alpha, wf2.tensor_beta)



    # definitely Correct



    def modified_cholesky_decomposition(self, eri):
        eri = np.array(eri)
        norb = eri.shape[0]
        v2e = eri.reshape((norb**2, -1))
        thresh = 10**(-9)
        residual = v2e.copy()
        num_fields = 0
        tensor_list = []
        error = np.linalg.norm(residual)
        while error > thresh:
            max_res, max_i = np.max(np.diag(residual)), np.argmax(np.diag(residual))
            for i in range(norb**2):
                if np.sqrt(np.abs(residual[i, i] * max_res)) <= thresh:
                    residual[i, i] = 0
            L = residual[:, max_i] / np.sqrt(max_res)
            tensor_list.append(L.reshape(norb, norb))
            for i in range(norb**2):
                for j in range(norb**2):
                    products = np.array([L.flatten()[i] * L.flatten()[j] for L in tensor_list])
                    residual[i, j] = v2e[i, j] - np.sum(products)
            error = np.linalg.norm(residual)
            num_fields += 1
        tensor_list = np.array(tensor_list)
        return jnp.array(tensor_list), num_fields



    # definitely Correct
    def get_overlap(self, wf1, wf2):
        return ovlp_helper(wf1.tensor_alpha, wf1.tensor_beta, wf2.tensor_alpha, wf2.tensor_beta)



        #Trying new propagator

    def propagate(self, wf, h1e, l_tensor, dt, ds, aux):

        mean = jnp.zeros_like(aux[0])

        cov = jnp.eye(len(aux[0]))

        prob = [multivariate_normal.pdf(i, mean = mean, cov = cov) for i in aux]

        def helper(wf, h1e, l_tensor, dt, ds, aux, prob):
            v0 = jnp.zeros(h1e.shape)
            temp = jnp.einsum('npr, nrq -> pq', l_tensor, l_tensor)
            v0 = v0.at[:, :].set(h1e - 0.5 * temp)
            mat = jnp.einsum('m, mpq -> pq', aux, l_tensor)
            if ds > 0:
                matrixA = 1j * jnp.sqrt(ds) * mat - dt * v0
            else:
                matrixA = jnp.sqrt(-ds) * mat - dt * v0
            tempa = wf.tensor_alpha.copy()
            tempb = wf.tensor_beta.copy()
            expA = jsp.linalg.expm(matrixA)
            wf.tensor_alpha = prob * jnp.einsum('pq, qr -> pr', expA, tempa)
            wf.tensor_beta = prob * jnp.einsum('pq, qr -> pr', expA, tempb)
        for i in range(self.nsteps):
            helper(wf, h1e, l_tensor, dt, ds, aux[i], prob[i])





    # Definitely Correct
    def get_local_energy(self, h1e, v2e, nuc, wf1, wf2):
        galpha, gbeta = self.get_green_func(wf1, wf2)
        return energy_helper(galpha, gbeta, h1e, v2e, nuc)


    # I'm pretty confident
    def met_hasting_sample(self, ham, h1e, l_tensor, dt, ds, key, nsteps=100, step_size=0.15, burn_in=200):
        key, subkey = random.split(key)
        currentx =  random.normal(key, shape = (self.nsteps, self.nfields))
        currentxprime = random.normal(subkey, shape = (self.nsteps, self.nfields))
        currentxprime = currentx
        trueh1e, v2e, nuc = ham
        wf1 = Trial(self.mol, self.trial_type)
        wf2 = Trial(self.mol, self.trial_type)
        wf1.tensor_alpha = self.opt_wavefunction.tensor_alpha.copy()
        wf1.tensor_beta = self.opt_wavefunction.tensor_beta.copy()
        wf2.tensor_alpha = self.opt_wavefunction.tensor_alpha.copy()
        wf2.tensor_beta = self.opt_wavefunction.tensor_beta.copy()
        proposedwf1 = Trial(self.mol, self.trial_type)
        proposedwf2 = Trial(self.mol, self.trial_type)
        self.propagate(wf1, h1e, l_tensor, dt, ds, currentx)
        self.propagate(wf2, h1e, l_tensor, dt, ds, currentxprime)
        current_magnitude, current_phase = self.get_overlap(wf1, wf2)
        samples, magnitudes, phases, energies = [], [], [], []
        total_steps = 0
        accepted_steps = 0

        while accepted_steps < nsteps:
            key, subkey = random.split(key)
            proposedx = currentx + step_size * random.normal(key, shape=currentx.shape)
            proposedxprime = currentxprime + step_size * random.normal(subkey, shape=currentxprime.shape)
            # Reset wavefunctions
            proposedwf1.tensor_alpha = self.opt_wavefunction.tensor_alpha.copy()
            proposedwf1.tensor_beta = self.opt_wavefunction.tensor_beta.copy()
            proposedwf2.tensor_alpha = self.opt_wavefunction.tensor_alpha.copy()
            proposedwf2.tensor_beta = self.opt_wavefunction.tensor_beta.copy()
            self.propagate(proposedwf1, h1e, l_tensor, dt, ds, proposedx)
            self.propagate(proposedwf2, h1e, l_tensor, dt, ds, proposedxprime)
            proposed_magnitude, proposed_phase = self.get_overlap(proposedwf1, proposedwf2)
            accept = proposed_magnitude / current_magnitude
            key, subkey = random.split(key)
            u = random.uniform(subkey)
            if accept > u:
                if burn_in > 0:
                    burn_in -= 1
                else:
                    samples.append((currentx, currentxprime))
                    magnitudes.append(current_magnitude)
                    phases.append(current_phase)
                    current_energy = self.get_local_energy(trueh1e, v2e, nuc, proposedwf1, proposedwf2)
                    energies.append(current_energy)
                    accepted_steps += 1
                currentx = proposedx
                currentxprime = proposedxprime
                current_magnitude = proposed_magnitude
                current_phase = proposed_phase
            total_steps += 1
        print("Sampling completed.")
        print("Acceptance rate", accepted_steps / total_steps)
        return jnp.array(samples), jnp.array(magnitudes), jnp.array(phases), jnp.array(energies), key



    def auxiliary_distribution(self, h1e, l_tensor, dt, ds, initialxprime):
        currentx = jnp.linspace(-5, 5, 100)
        magnitudes = []
        currentxprime = initialxprime
        for i in range(len(currentx)):
            wf1 = Trial(self.mol, self.trial_type)
            wf1.tensor_alpha = self.opt_wavefunction.tensor_alpha.copy()
            wf1.tensor_beta = self.opt_wavefunction.tensor_beta.copy()
            wf2 = Trial(self.mol, self.trial_type)
            wf2.tensor_alpha = self.opt_wavefunction.tensor_alpha.copy()
            wf2.tensor_beta = self.opt_wavefunction.tensor_beta.copy()
            for _ in range(self.nsteps):
                self.propagate(wf1, h1e, l_tensor, dt, ds, jnp.array([currentx[i]]))
                self.propagate(wf2, h1e, l_tensor, dt, ds, currentxprime)
            current_magnitude, current_phase = self.get_overlap(wf1, wf2)
            magnitudes.append(current_magnitude)
        magnitudes = jnp.array(magnitudes)
        # Normalize the magnitudes to form a probability distribution
        dx = currentx[1] - currentx[0]
        total_area = jnp.sum(magnitudes) * dx
        normalized_magnitudes = magnitudes / total_area
        plt.plot(currentx, normalized_magnitudes)





    def get_variational_energy(self, x, h1e, v2e, nuc):
        lam = 1
        B = 0.7
        psi_alpha_params, psi_beta_params, dt_params, ds_params, h1e_params, l_tensor_params, nuc_params = self.unpack(x)
        self.opt_wavefunction.tensor_alpha = psi_alpha_params
        self.opt_wavefunction.tensor_beta = psi_beta_params
        ham = h1e, v2e, nuc
        print("psi_alpha_params", psi_alpha_params)
        #print("psi_beta_params", psi_beta_params)
        print("ds_params", ds_params)
        print("dt_params", dt_params)
        #print("h1e_params", h1e_params[0])
        #print("l_tensor_params", l_tensor_params[0])
        #print("nuc_params", nuc_params)
        #xprime = jnp.array([0.6])
        #x = jnp.array([x[0]])
       # self.auxiliary_distribution(h1e, l_tensor_params, dt_params, ds_params, xprime)
        ax, mags, phases, energies, self.key = self.met_hasting_sample(ham, h1e_params, l_tensor_params, dt_params, self.dt, self.key)
       # plt.hist(ax, density=True, bins=30, alpha=0.6, color='g')
       # plt.show()
        #raise StopExecution
        mags_total = jnp.sum(mags)
        norm_mags = mags/mags_total
        print("Auxilary Fields", ax[:5])
        print("Normalized Magnitudes", mags[:5])
        print("Phases", phases[:5])
        print("Energies", energies[:5])
        #print("Energy Variance", jnp.var(energies))
        acc = jnp.zeros_like(phases)
        #print("Auxilary Fields", ax)
        for i in range(len(phases)):
            acc = acc.at[i].set(phases[i] * energies[i])
            #acc2 = acc2.at[i].set(phases[i] * energies[i] * mags[i])
            #acc3 = acc3.at[i].set(mags[i] * phases[i])
        avg_ovlp = jnp.mean(phases)
       # if jnp.abs(jnp.real(avg_ovlp)) < 0.1:
          #print("Propagating too far encountering sign problem")
        print("Average sign", jnp.real(avg_ovlp))
        variational_energy = jnp.mean(acc) / avg_ovlp
        #var2 = jnp.sum(acc2) / jnp.sum(acc3)
        #print("energy calc:", variational_energy == var2)
        print("var energy", jnp.real(variational_energy))
        #print("var2", jnp.real(var2))
        return jnp.real(variational_energy) + jnp.multiply(lam, jnp.square(jnp.maximum(B - jnp.real(avg_ovlp), 0)))



    def gradient(self, x, h1e, v2e, nuc):
        grad = jax.grad(self.get_variational_energy)(x, h1e, v2e, nuc)
        print("Gradient magnitudes:", jnp.linalg.norm(grad))
        return np.array(grad, dtype=np.float64)







#Add constraint for ds and dt

    def run(self, tau, max_iter=50000, tol=1e-10, disp=True):
        self.initialize_simulation()
        h1e, v2e, nuc = self.hamiltonian_integral()
        key = random.PRNGKey(self.seed)
        random_key, subkey = random.split(key)
        h1e_opt = h1e  * (1 + 0.01 * random.normal(random_key, h1e.shape))
        l_tensor, self.nfields = self.modified_cholesky_decomposition(v2e)
        l_tensor_opt = l_tensor * (1 + 0.01 * random.normal(subkey, l_tensor.shape))
        self.shape_h1e = h1e.shape
        self.shape_v2e = l_tensor.shape
        x = self.pack(self.opt_wavefunction.tensor_alpha, self.opt_wavefunction.tensor_beta, tau, tau, h1e_opt, l_tensor_opt, nuc)
        res = minimize(
            self.get_variational_energy,
            x,
            args=(h1e, v2e, nuc),
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
        etrial = res.fun
        return etrial



if __name__ == "__main__":
    tau_init = 0.8
    dt = 0.05
    r = 1.42
    atomstring = f'''
                 H 0. 0. 0.
                 H 0. 0. {r}
              '''
    mol = gto.M(atom=atomstring, basis='sto-3g', unit='Bohr', verbose=3, symmetry=0)
    job = Jax_Propagator(mol, dt, tau_init)
    efinal = job.run(dt)
    mf = scf.UHF(mol)
    mf.kernel()
    cisolver = fci.FCI(mf)
    fci_energy = cisolver.kernel()[0]
    print("fci energy", fci_energy)
    print(efinal)