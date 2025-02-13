import numpy as np
from pyscf import fci, scf
from ipie.analysis.autocorr import reblock_by_autocorr
import pandas as pd
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

def get_fci(mol):
    mf = scf.RHF(mol)
    mf.kernel()  # This will run the Hartree-Fock calculation

    # Step 3: Set up the FCI calculation
    cisolver = fci.FCI(mf)

    # Step 4: Run the FCI calculation
    energy_fci, fcivec = cisolver.kernel()
    return energy_fci


def reblock(x):
    start_index = len(x) - len(x) // 6 * 5
    print(reblock_by_autocorr(x[start_index:]))