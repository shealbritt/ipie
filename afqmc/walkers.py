
import numpy as np
import scipy
from ipie.walkers.base_walkers import BaseWalkers

class Walkers():
    def __init__(self, mol, nwalkers):
        if nwalkers <= 0:
            raise ValueError
        self.mol = mol
        self.nwalkers = nwalkers
        self.tensora = None
        self.tensorb= None

    def init_walkers(self, tempa, tempb):
        self.tensora = np.array([tempa] * self.nwalkers, dtype=np.complex128)
        self.tensorb = np.array([tempb] * self.nwalkers, dtype=np.complex128)
        self.weight = np.array([1.] * self.nwalkers)

class JaxWalkers():
    def __init__(self, mol, nwalkers):
        if nwalkers <= 0:
            raise ValueError
        self.mol = mol
        self.nwalkers = nwalkers
        self.tensorar = None
        self.tensorbr = None
        self.tensoral = None
        self.tensorbl = None

    def init_walkers(self, tempa, tempb):
        self.tensorar = np.array([tempa] * self.nwalkers, dtype=np.complex128)
        self.tensorbr = np.array([tempb] * self.nwalkers, dtype=np.complex128)
        self.tensoral = np.array([tempa] * self.nwalkers, dtype=np.complex128)
        self.tensorbl = np.array([tempb] * self.nwalkers, dtype=np.complex128)
        self.weight = np.array([1.] * self.nwalkers)
