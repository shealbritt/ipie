import cProfile
import pstats
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
from pyscf import gto, scf, fci  # Assuming you use PySCF for molecule definition
from hmc_vafqmc_lbfgs import Propagator  # Replace with the actual module where your Propagator class is
import sys
sys.path.append('../afqmc/')
from trial import Trial 
from keymanager import KeyManager

mol = gto.M(atom='H 0 0 0; H 0 0 1.6', basis='sto-3g', unit='bohr')
mf = scf.RHF(mol)
hf_energy = mf.kernel()
cisolver = fci.FCI(mf)
fci_energy = cisolver.kernel()[0]
nsteps = 10
dt = 0.1
prop = Propagator(mol, dt=dt, nsteps=nsteps, nwalkers=1000) # Example parameters
prop.trial = Trial(prop.mol)
prop.trial.get_trial()
prop.trial.tensora = jnp.array(prop.trial.tensora, dtype=jnp.complex128)
prop.trial.tensorb = jnp.array(prop.trial.tensorb, dtype=jnp.complex128)
h1e, v2e, nuc, l_tensor = prop.hamiltonian_integral()
prop.h1e = jnp.array(h1e)
prop.v2e = jnp.array(v2e)
prop.nuc = nuc
prop.l_tensor = jnp.array(l_tensor)
h1e_repeated = jnp.tile(h1e, (prop.nsteps, 1, 1))  # Repeat h1e nsteps times
t = jnp.array([prop.dt] * prop.nsteps)
s = t.copy()
params = np.load("optimal_params_adams.npy", allow_pickle=True)    
        
profiler = cProfile.Profile()
profiler.enable()
# Run the function you want to profile
prop.trial = Trial(prop.mol)
prop.trial.get_trial()
prop.trial.tensora = jnp.array(prop.trial.tensora, dtype=jnp.complex128)
prop.trial.tensorb = jnp.array(prop.trial.tensorb, dtype=jnp.complex128)
h1e, v2e, nuc, l_tensor = prop.hamiltonian_integral()
prop.h1e = jnp.array(h1e)
prop.v2e = jnp.array(v2e)
prop.nuc = nuc
prop.l_tensor = jnp.array(l_tensor)
h1e_repeated = jnp.tile(h1e, (prop.nsteps, 1, 1))  # Repeat h1e nsteps times
t = jnp.array([prop.dt] * prop.nsteps)
s = t.copy()
params = np.load("optimal_params_adams.npy", allow_pickle=True)    
prop.objective_func(params)
profiler.disable()
stats = pstats.Stats(profiler)
stats.sort_stats("cumulative").print_stats(20)  # Show top 20 slowest calls



