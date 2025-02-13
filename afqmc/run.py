
from ipie.analysis.autocorr import reblock_by_autocorr
from utils import read_fcidump, get_fci
import matplotlib.pyplot as plt
from pyscf import tools, lo, scf, fci, gto, cc
from afqmc import Propagator

def reblock(x):
    print(reblock_by_autocorr(x))

def run_h2o():
    # Define the H2O molecule with PySCF
    mol = gto.M(atom='''
        O  0.0  0.0  0.0
        H  0.757  0.586  0.0
        H  -0.757  0.586  0.0
        ''', basis='sto-3g', unit='angstrom')

    # Instantiate the Propagator class for the H2O molecule
    print("Running H2O")
    prop = Propagator(mol, dt=0.01, total_t=10, nwalkers=1000, trial_type="uhf")

    # Run the simulation to get time and energy lists
    time_list, energy_list = prop.run()
    energy_fci = get_fci(mol)

    # Plot time vs energy
    plt.figure(figsize=(8, 6))
    plt.plot(time_list, energy_list, label='AFQMC Energy (H2O)', color='g', marker='o')
    plt.hlines(energy_fci, xmin=0, xmax=15)
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Energy (Hartree)')
    plt.title('AFQMC Simulation: Time vs Energy for H2O')
    plt.grid(True)
    plt.legend()
    plt.show()

    reblock(energy_list)


def run_co2():
    # Define the CO2 molecule with PySCF
    mol = gto.M(atom='O 0 0 0; C 0 0 1.16; O 0 0 2.32', basis='sto-3g', unit='angstrom')

    # Instantiate the Propagator class for the CO2 molecule
    print("Running CO2")
    prop = Propagator(mol, dt=0.01, total_t=10, nwalkers=1000, trial_type="uhf")

    # Run the simulation to get time and energy lists
    time_list, energy_list = prop.run()
    energy_fci = get_fci(mol)

    # Plot time vs energy
    plt.figure(figsize=(8, 6))
    plt.plot(time_list, energy_list, label='AFQMC Energy (CO2)', color='r', marker='o')
    plt.hlines(energy_fci, xmin=0, xmax=15)
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Energy (Hartree)')
    plt.title('AFQMC Simulation: Time vs Energy for CO2')
    plt.grid(True)
    plt.legend()
    plt.show()
    reblock(energy_list[400:])

def run_n2():
    # Define the N2 molecule with PySCF
    mol = gto.M(atom='N 0 0 0; N 0 0 1.1', basis='sto-3g', unit='angstrom')

    # Instantiate the Propagator class for the N2 molecule
    print("Running N2")
    prop = Propagator(mol, dt=0.01, total_t=10, nwalkers=1000, trial_type="uhf")

    # Run the simulation to get time and energy lists
    time_list, energy_list = prop.run()
    energy_fci = get_fci(mol)

    # Plot time vs energy
    plt.figure(figsize=(8, 6))
    plt.plot(time_list, energy_list, label='AFQMC Energy (N2)', color='c', marker='o')
    plt.hlines(energy_fci, xmin=0, xmax=15)
    plt.xlabel('Time (a.u.)')
    plt.ylabel('Energy (Hartree)')
    plt.title('AFQMC Simulation: Time vs Energy for N2')
    plt.grid(True)
    plt.legend()
    plt.show()

if __name__ == '__main__':
    # Call the functions for each molecule you want to simulate
    
    #run_h2o()  # Run the H2O simulation
    run_co2()  # Uncomment to run CO2 simulation
    #run_n2()   # Uncomment to run N2 simulation
