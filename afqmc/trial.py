import unittest
import numpy as np
import h5py
from pyscf import gto, scf, cc, lo
class Trial(object):
    def __init__(self, mol):
        self.mol = mol
        self.triala = None
        self.trialb = None
        self.mo_coeffs = None
        self.energy = None

    def get_trial(self):
        # UHF
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