import unittest
import numpy as np
import h5py
import datetime
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
        timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
        filename = f"input_{timestamp}.h5"
        self.input = filename
        mf = scf.UHF(self.mol)
        dm_alpha, dm_beta = mf.get_init_guess()
        dm_beta[:2,:2] = 0
        dm = (dm_alpha,dm_beta)
        mf.kernel(dm)
        s_mat = self.mol.intor('int1e_ovlp')
        ao_coeff = lo.orth.lowdin(s_mat)
        xinv = np.linalg.inv(ao_coeff)
        self.input = filename
        self.trial = mf.mo_coeff
        self.tensora = xinv.dot(mf.mo_coeff[0][:, :self.mol.nelec[0]])
        self.tensorb = xinv.dot(mf.mo_coeff[1][:, :self.mol.nelec[1]])
        with h5py.File(filename, "w") as fa:
            fa["ao_coeff"] = ao_coeff
            fa["xinv"] = xinv
            fa["phia"] = self.tensora
            fa["phib"] = self.tensorb
