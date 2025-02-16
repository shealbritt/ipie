
import numpy as np
from pyscf import gto, scf
from ipie.qmc.afqmc import AFQMC
from ipie.utils.from_pyscf import generate_hamiltonian, generate_wavefunction_from_mo_coeff
from ipie.trial_wavefunction.single_det import SingleDet
from ipie.analysis.extraction import extract_hdf5_data, extract_observable

r0 = 1.6
natoms = 10
mol = gto.M(
    atom=[("H", i*r0, 0, 0) for i in range(natoms)],
    basis='sto-6g',
    unit='Bohr',
    verbose=5
)
def ipierun(mol):
    mf = scf.RHF(mol)
    mf.chkfile = "scf.chk"
    mf.kernel()

    # Now let's build our custom AFQMC algorithm
    num_walkers = 1000
    num_steps_per_block = 1
    num_blocks = 5000
    timestep = 0.01


    # 1. Build Hamiltonian
    ham = generate_hamiltonian(
        mol,
        mf.mo_coeff,
        mf.get_hcore(),
        mf.mo_coeff,  # should be optional
    )

    # 2. Build trial wavefunction
    orbs = generate_wavefunction_from_mo_coeff(
        mf.mo_coeff,
        mf.mo_occ,
        mf.mo_coeff,  # Make optional argument
        mol.nelec,
    )
    num_basis = mf.mo_coeff[0].shape[-1]
    trial = SingleDet(
        np.hstack([orbs, orbs]),
        mol.nelec,
        num_basis,
    )
    trial.half_rotate(ham)

    afqmc = AFQMC.build(
        mol.nelec,
        ham,
        trial,
        num_walkers=num_walkers,
        num_steps_per_block=num_steps_per_block,
        num_blocks=num_blocks,
        timestep=timestep,
        pop_control_freq=5,
        seed=59306159,
    )
    afqmc.run()


    # Open the HDF5 file
    h5_file = 'estimates.0.h5'
    df = extract_observable(h5_file)

    energylist = df["ETotal"].values.tolist()
    return energylist


