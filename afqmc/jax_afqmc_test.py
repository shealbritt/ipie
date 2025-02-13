import unittest
import numpy as np
import jax.numpy as jnp
from pyscf import gto
from afqmc import Propagator  
from trial import Trial
from walkers import Walkers

class TestPropagator(unittest.TestCase):
    def setUp(self):
        """Set up the molecule, trial, and propagator for all tests."""
        # Set up a simple molecule for testing
        self.mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', unit='angstrom')

        # Trial setup for the propagator
        self.trial = Trial(self.mol, trial_type="uhf")
        self.trial.get_trial()  # Get the trial wavefunction (it may initialize tensors)
        
        # Propagator setup
        self.dt = 0.001
        self.total_t = 2.0
        self.nwalkers = 100
        self.norb = self.trial.tensor_alpha.shape[0]
        self.propagator = Propagator(self.mol, self.dt, self.total_t, nwalkers=self.nwalkers, trial_type="uhf")
    
    def test_initialization(self):
        """Test if the Propagator is initialized correctly."""
        self.assertEqual(self.propagator.mol, self.mol)
        self.assertEqual(self.propagator.dt, self.dt)
        self.assertEqual(self.propagator.nwalkers, self.nwalkers)
        self.assertEqual(self.propagator.trial_type, "uhf")
        
        # Check if the trial and walkers are initialized
        self.assertIsNotNone(self.propagator.trial)
        self.assertIsNotNone(self.propagator.walkers)

    def test_overlap_calculation(self):
        """Test the calculation of the overlap between trial and walkers."""
        
        # Set specific trial and walker tensors (V and U) for testing
        # Here we generate simple random complex tensors for trial and walkers
        norb = self.norb  # number of orbital
        self.propagator.trial.tensor_alpha = jnp.array(np.random.random((norb, norb)) + 1j * np.random.random((norb, norb)))
        self.propagator.trial.tensor_beta = jnp.array(np.random.random((norb, norb)) + 1j * np.random.random((norb, norb)))

        self.propagator.walkers.tensor_alpha = jnp.array(np.random.random((self.propagator.nwalkers, norb, norb))\
                                               + 1j * np.random.random((self.propagator.nwalkers, norb, norb)))
        self.propagator.walkers.tensor_beta = jnp.array(np.random.random((self.propagator.nwalkers, norb, norb)) \
                                              + 1j * np.random.random((self.propagator.nwalkers, norb, norb)))

        # Compute the overlap by explicitly calculating V† * U (for both alpha and beta)
        V_alpha = self.propagator.trial.tensor_alpha  # Trial tensor for alpha
        U_alpha = self.propagator.walkers.tensor_alpha  # Walker tensor for alpha
        V_beta = self.propagator.trial.tensor_beta  # Trial tensor for beta
        U_beta = self.propagator.walkers.tensor_beta  # Walker tensor for beta
        get_overlap = self.propagator.get_overlap()
        # Initialize overlap lists
        overlap_alpha_list = []
        overlap_beta_list = []

        # Loop over each walker and calculate individual overlaps
        for i in range(self.propagator.nwalkers):
            # V† * U for alpha (for each walker)
            overlap_alpha = np.linalg.det(V_alpha.conj().T @ U_alpha[i]) # V† * U for each walker (alpha)
            overlap_alpha_list.append(overlap_alpha)

            # V† * U for beta (for each walker)
            overlap_beta = np.linalg.det(V_beta.conj().T @ U_beta[i] ) # V† * U for each walker (beta)
            overlap_beta_list.append(overlap_beta)

        # Calculate total overlap by taking the product of alpha and beta overlaps for each walker
        total_overlap_list = [overlap_alpha_list[i] * overlap_beta_list[i] for i in range(self.propagator.nwalkers)]

        # For testing, we will check that the overlap is positive for each walker
        for i in range(self.propagator.nwalkers):
            self.assertTrue(jnp.allclose(total_overlap_list[i], get_overlap[i]))

       

    def test_get_theta(self):
        """Test the calculation of the theta values."""
        nwalkers = self.propagator.nwalkers
        self.propagator.initialize_simulation()  # Initialize walkers and trial
        norb = self.norb  # number of orbitals
        self.propagator.trial.tensor_alpha = np.random.random((norb, norb)) + 1j * np.random.random((norb, norb))
        self.propagator.trial.tensor_beta = np.random.random((norb, norb)) + 1j * np.random.random((norb, norb))

        self.propagator.walkers.tensor_alpha = np.random.random((nwalkers, norb, norb)) + 1j * np.random.random((nwalkers, norb, norb))
        self.propagator.walkers.tensor_beta = np.random.random((nwalkers, norb, norb)) + 1j * np.random.random((nwalkers, norb, norb))
        # Compute the overlap by explicitly calculating V† * U (for both alpha and beta)
        V_alpha = self.propagator.trial.tensor_alpha  # Trial tensor for alpha
        U_alpha = self.propagator.walkers.tensor_alpha  # Walker tensor for alpha
        V_beta = self.propagator.trial.tensor_beta  # Trial tensor for beta
        U_beta = self.propagator.walkers.tensor_beta  # Walker tensor for beta

        theta_alpha, theta_beta = self.propagator.get_theta()
        norb = self.propagator.trial.tensor_alpha.shape[0]
        # We check if theta_alpha and theta_beta are numpy arrays with expected shapes
        self.assertEqual(V_alpha.shape, (norb, norb))
        self.assertEqual(V_beta.shape, (norb, norb))
        self.assertEqual(U_alpha.shape, (nwalkers,norb, norb))
        self.assertEqual(U_beta.shape, (nwalkers, norb, norb))
        self.assertTrue(isinstance(theta_alpha, np.ndarray))
        self.assertTrue(isinstance(theta_beta, np.ndarray))
        self.assertEqual(theta_alpha.shape, (nwalkers, norb, norb)) 
        self.assertEqual(theta_beta.shape, (nwalkers, norb, norb))  
        theta_alpha_test = np.zeros((nwalkers, norb, norb), dtype=complex)  
        theta_beta_test = np.zeros((nwalkers, norb, norb), dtype=complex) 
        for i in range(self.propagator.nwalkers):
            theta_alpha_test[i] = U_alpha[i] @ np.linalg.inv(V_alpha.conj().T @ U_alpha[i]) 
            theta_beta_test[i] = U_beta[i] @ np.linalg.inv(V_beta.conj().T @ U_beta[i])
            self.assertTrue(np.allclose(theta_alpha[i], theta_alpha_test[i]))
            self.assertTrue(np.allclose(theta_beta[i], theta_beta_test[i]))

    def test_get_green_func(self):
        """Test the calculation of the Green's function."""
        
        # Set specific trial and walker tensors (V and U) for testing
        norb = self.norb  # number of orbitals
        self.propagator.trial.tensor_alpha = np.random.random((norb, norb)) + 1j * np.random.random((norb, norb))
        self.propagator.trial.tensor_beta = np.random.random((norb, norb)) + 1j * np.random.random((norb, norb))

        self.propagator.walkers.tensor_alpha = np.random.random((self.propagator.nwalkers, norb, norb)) + 1j * np.random.random((self.propagator.nwalkers, norb, norb))
        self.propagator.walkers.tensor_beta = np.random.random((self.propagator.nwalkers, norb, norb)) + 1j * np.random.random((self.propagator.nwalkers, norb, norb))

        # Call the get_green_func method to calculate Green's function
        green_func_alpha, green_func_beta = self.propagator.get_green_func()

        # Perform assertions to check if the Green's function is calculated properly
        # 1. Check that the shapes of the Green's function matrices match the expected dimensions
        self.assertEqual(green_func_alpha.shape, (self.propagator.nwalkers, norb, norb))
        self.assertEqual(green_func_beta.shape, (self.propagator.nwalkers, norb, norb))

        theta_alpha, theta_beta = self.propagator.get_theta()
        V_alpha = self.propagator.trial.tensor_alpha  # Trial tensor for alpha
        V_beta = self.propagator.trial.tensor_beta 
        green_func_alpha_test = np.einsum('zqr, rp -> zpq', theta_alpha, V_alpha.conj().T)
        green_func_beta_test = np.einsum('zqr, rp -> zpq', theta_beta, V_beta.conj().T)
        for i in range(self.nwalkers):
            print("green_func", green_func_alpha[i])
            print("green_func_test", green_func_alpha_test[i])
            self.assertTrue(np.allclose(green_func_alpha[i], green_func_alpha_test[i]))
            self.assertTrue(np.allclose(green_func_beta[i], green_func_beta_test[i]))


    def test_get_local_energy(self):
        """Test the calculation of the local energy."""
        self.propagator.initialize_simulation()
        h1e, v2e, nuc, l_tensor = self.propagator.hamiltonian_integral()
        theta_alpha, theta_beta = self.propagator.get_theta()
        green_func_alpha, green_func_beta = self.propagator.get_green_func()
        
        local_energy = self.propagator.get_local_energy(
            l_tensor, theta_alpha, theta_beta, v2e, h1e, nuc, green_func_alpha, green_func_beta)
        
        # We check if the local energy is a numpy array
        #Check with tong local energy
        def tong_local_energy(h1e, v2e, nuc, green_func_alpha, green_func_beta):
            local_e2 =  np.einsum("prqs, zpr, zqs->z", v2e, green_func_alpha, green_func_alpha)
            local_e2 +=  np.einsum("prqs, zpr, zqs->z", v2e, green_func_beta, green_func_beta)
            local_e2 +=  np.einsum("prqs, zpr, zqs->z", v2e, green_func_alpha, green_func_beta)
            local_e2 += np.einsum("prqs, zps, zqr->z", v2e, green_func_beta, green_func_alpha)
            local_e2 -= np.einsum("prqs, zps, zqr->z", v2e, green_func_alpha, green_func_alpha)
            local_e2 -= np.einsum("prqs, zps, zqr->z", v2e, green_func_beta, green_func_beta)
            local_e1 =  np.einsum("zpq, pq->z", green_func_alpha, h1e)
            local_e1 +=  np.einsum("zpq, pq->z", green_func_beta, h1e)
            local_e = (local_e1 + 0.5 * local_e2 + nuc)
            return local_e
        tong_energy = tong_local_energy(h1e, v2e,nuc, green_func_alpha, green_func_beta)
        self.assertTrue(isinstance(local_energy, np.ndarray))
        self.assertEqual(local_energy.shape, (self.nwalkers,))  # Expecting an array of size (nwalkers,)
        for i in range(len(local_energy)):
            self.assertAlmostEqual(local_energy[i], tong_energy[i])

    def test_propagation(self):

        # TODO
        """Test the propagation of walkers in time."""
        self.propagator.initialize_simulation()  # Initialize walkers and trial
        time_list, energy_list = self.propagator.run()
        
        # Check that time and energy lists have the expected length
        self.assertEqual(len(time_list), len(energy_list))
        self.assertGreater(len(time_list), 0)
        self.assertGreater(len(energy_list), 0)

        # Check that the energy list is not empty and the first energy value is the initial energy
        self.assertAlmostEqual(energy_list[0], self.propagator.trial.energy)  # Assuming a reasonable energy value for the system
        # TODO Finish Testing 


    def test_orthonormalization(self):
        """Test the orthonormalization of walkers."""
        norb = self.propagator.trial.tensor_alpha.shape[0]
        self.propagator.walkers.tensor_alpha = np.random.random((self.propagator.nwalkers, norb, norb)) + 1j * np.random.random((self.propagator.nwalkers, norb, norb))
        self.propagator.walkers.tensor_beta = np.random.random((self.propagator.nwalkers, norb, norb)) + 1j * np.random.random((self.propagator.nwalkers, norb, norb))
        # Perform orthonormalization
        self.propagator.orthonormalize()
        # Check orthogonality and normalization for each walker
        for walker in range(self.propagator.nwalkers):
            # For alpha orbitals
            alpha = self.propagator.walkers.tensor_alpha[walker]
            # Check if each orbital is normalized (should be 1)
            for i in range(self.norb):
                norm = np.vdot(alpha[i], alpha[i])  # Should be 1 for normalized orbital
                self.assertAlmostEqual(norm, 1.0, msg=f"Walker {walker} alpha orbital {i} is not normalized: {norm}")

            # Check if the orbitals are orthogonal (dot product should be 0 for different orbitals)
            for i in range(self.norb):
                for j in range(i + 1, self.norb):
                    overlap = np.vdot(alpha[i], alpha[j])  # Should be close to 0
                    self.assertAlmostEqual(overlap, 0.0, msg=f"Walker {walker} alpha orbitals {i} and {j} are not orthogonal: {overlap}")

            # For beta orbitals
            beta = self.propagator.walkers.tensor_beta[walker]
            # Check if each orbital is normalized (should be 1)
            for i in range(self.norb):
                norm = np.vdot(beta[i], beta[i])  # Should be 1 for normalized orbital
                self.assertAlmostEqual(norm, 1.0, msg=f"Walker {walker} beta orbital {i} is not normalized: {norm}")

            # Check if the orbitals are orthogonal (dot product should be 0 for different orbitals)
            for i in range(self.norb):
                for j in range(i + 1, self.norb):
                    overlap = np.vdot(beta[i], beta[j])  # Should be close to 0
                    self.assertAlmostEqual(overlap, 0.0, msg=f"Walker {walker} beta orbitals {i} and {j} are not orthogonal: {overlap}")


        
    def test_update_walker(self):
        """Test if the walkers are updated correctly."""

        # TODO
        self.propagator.initialize_simulation()  # Initialize walkers and trial
        local_energy = np.zeros(self.nwalkers)  # Fake local energy for testing
        energy = np.zeros(1)  # Fake energy for testing
        overlap = self.propagator.get_overlap()  # Get initial overlap

        # Perform the update on the walkers
        self.propagator.update_walker(local_energy, energy, overlap)
        
        # Check that the walkers' weights have changed (i.e., update has occurred)
        self.assertFalse(np.array_equal(self.propagator.walkers.weight, np.ones(self.nwalkers)))
        # TODO


    def test_hamiltonian_integral(self):
        """Test if the Hamiltonian integral is correctly calculated."""
        self.propagator.initialize_simulation()  # Initialize simulation
        h1e, v2e, nuc, l_tensor = self.propagator.hamiltonian_integral()
        
        # Check that the Hamiltonian components are numpy arrays
        self.assertTrue(isinstance(h1e, np.ndarray))
        self.assertTrue(isinstance(v2e, np.ndarray))
        self.assertTrue(isinstance(nuc, float))
        self.assertTrue(isinstance(l_tensor, np.ndarray))
        #ERI symmetry
        for p in range(self.norb):
                for r in range(self.norb):
                    for q in range(self.norb):
                        for s in range(self.norb):
                            # Check if the given symmetry holds
                            self.assertTrue(v2e[p, q, r, s] == v2e[r, q, p, s] == v2e[r, s, p, q] == v2e[p, s, r, q])
        # Check Cholesky
        eri_reconstructed = np.einsum('gpr, gqs -> pqrs', l_tensor, l_tensor)
        self.assertTrue(np.allclose(v2e, eri_reconstructed))
if __name__ == '__main__':
    unittest.main()