
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
        self.walker_tensora = np.array([tempa] * self.nwalkers, dtype=np.complex128)
        self.walker_tensorb = np.array([tempb] * self.nwalkers, dtype=np.complex128)
        self.walker_weight = np.array([1.] * self.nwalkers)
