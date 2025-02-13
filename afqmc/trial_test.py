import unittest
import numpy as np
import h5py
from pyscf import gto, scf, cc
from trial import Trial  # Assuming the class is saved in 'trial.py'

class TestTrialClass(unittest.TestCase):

    def setUp(self):
        """Setup a simple molecule for all tests."""
        self.mol = gto.Mole()
        self.mol.build(
            atom='H 0 0 0; H 0 0 0.74',
            basis='sto-3g'
        )
        self.trial = Trial(self.mol, trial_type="uhf")
    
    def test_initialization(self):
        """Test if initialization works correctly."""
        self.assertEqual(self.trial.mol, self.mol)
        self.assertEqual(self.trial.trial_type, "uhf")
        self.assertIsNone(self.trial.tensor_alpha)
        self.assertIsNone(self.trial.tensor_beta)
        self.assertIsNone(self.trial.mo_coeffs)
        self.assertIsNone(self.trial.energy)

    def test_get_trial_uhf(self):
        """Test the get_trial method for UHF trial type."""
        self.trial.get_trial()
        
        # Assert energy is set (UHF should run SCF calculation and CCSD if applicable)
        self.assertIsNotNone(self.trial.energy)
        
        # Assert tensor_alpha and tensor_beta are computed
        self.assertIsNotNone(self.trial.tensor_alpha)
        self.assertEqual(self.trial.tensor_alpha.shape[1], self.mol.nelec[0])  # Check if the tensor has the right number of orbitals
        
        # For UHF, tensor_beta should be computed and non-empty if there are beta electrons
        if self.mol.nelec[1] > 0:
            self.assertIsNotNone(self.trial.tensor_beta)
            self.assertEqual(self.trial.tensor_beta.shape[1], self.mol.nelec[1])  # Check if tensor_beta matches the number of beta electrons
        else:
            self.assertEqual(self.trial.tensor_beta.shape[1], 0)  # No beta electrons, should be empty tensor

    def test_get_trial_rhf(self):
        """Test the get_trial method for RHF trial type."""
        self.trial.trial_type = "rhf"
        self.trial.get_trial()
        
        # Assert energy is set (RHF should run SCF calculation)
        self.assertIsNotNone(self.trial.energy)
        
        # Assert tensor_alpha is computed (for RHF, tensor_alpha should equal tensor_beta)
        self.assertIsNotNone(self.trial.tensor_alpha)
        self.assertEqual(self.trial.tensor_alpha.shape[1], self.mol.nelec[0])  # RHF should have only alpha electrons
        
        # Tensor beta should be the same as tensor alpha for RHF
        self.assertTrue(np.array_equal(self.trial.tensor_alpha, self.trial.tensor_beta))

    def test_hdf5_output(self):
        """Test that the HDF5 output contains the expected datasets."""
        self.trial.get_trial()
        
        with h5py.File("input.h5", "r") as fa:
            self.assertIn("ao_coeff", fa)
            self.assertIn("xinv", fa)
            self.assertIn("phia0_alpha", fa)
            self.assertIn("phia0_beta", fa)
            
            # Check if the datasets have reasonable dimensions
            self.assertEqual(fa["ao_coeff"].shape[0], self.mol.nao)
            self.assertEqual(fa["xinv"].shape[0], self.mol.nao)
            self.assertEqual(fa["phia0_alpha"].shape[1], self.mol.nelec[0])
            self.assertEqual(fa["phia0_beta"].shape[1], self.mol.nelec[1] if self.mol.nelec[1] > 0 else 0)

    def test_energy_consistency(self):
        """Test if the SCF energy and CCSD energy (if applicable) are reasonable."""
        # Test for UHF and UCCSD
        self.trial.trial_type = "uccsd"
        self.trial.get_trial()
        # For UCCSD, the energy should be a reasonable float value (non-NaN)
        self.assertIsInstance(self.trial.energy, float)

        
        # Test for RHF and RCCSD
        self.trial.trial_type = "rccsd"
        self.trial.get_trial()
        self.assertIsInstance(self.trial.energy, float)

if __name__ == "__main__":
    unittest.main()