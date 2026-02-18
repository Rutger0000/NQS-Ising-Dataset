from benchmark_helpers import evaluate, boltzmann_energy
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from os import makedirs

def load_J_and_h(n_visible_spins, alpha):
    J = np.genfromtxt(f'weights/W_ising_{n_visible_spins}_{alpha}_ti_W.csv', delimiter=',')
    h = np.genfromtxt(f'weights/W_ising_{n_visible_spins}_{alpha}_ti_b.csv', delimiter=',')
    return J, h

def make_bins(values, stddevs=3, num_bins=50):
    mean = np.mean(values)
    std = np.std(values)
    min_val = mean - stddevs * std
    max_val = mean + stddevs * std
    bins = np.linspace(min_val, max_val, num_bins + 1)
    return bins

##########################################################################
####################### Define the system size ###########################
##########################################################################

n_visible_spins = 64
alpha = 2
total_spins = n_visible_spins * (alpha + 1)

# Load the J connectivity matrix and h bias vector for the Ising Hamiltonian
J, h = load_J_and_h(n_visible_spins, alpha)

# Load included reference data
df_reference = pd.read_parquet(f'data/observables_{n_visible_spins}_{alpha}.parquet')

##########################################################################
######### Load raw sampled states in -1 and 1 states #####################
##########################################################################

df_raw_states = pd.read_parquet(f'data/states_{n_visible_spins}_{alpha}.parquet') # THIS IS AN EXAMPLE FILE, REPLACE WITH YOUR OWN DATA
raw_states = df_raw_states.values[:4000, :total_spins]

# Evaluate observables for the given states using the evaluate function
boltzmann_energies, quantum_energies, log_psi, sum_visible_spins, variational_energy, variational_energy_sem = evaluate(n_visible_spins=n_visible_spins, alpha=alpha, states=raw_states, J=J, h=h)


############################################################################################################
# No need to edit below this line, this is just for plotting the results and comparing with reference data #
############################################################################################################

##########################################################################
########################### Plot the results #############################
##########################################################################

makedirs('plots', exist_ok=True)

##### Compare Boltzmann energy/Ising Hamiltonian with reference data #####

bins_boltzmann_energy = make_bins(df_reference['boltzmann_energy'], num_bins=50)

print("Boltzmann energies from reference data:", df_reference['boltzmann_energy'])

hist_reference, _ = np.histogram(df_reference['boltzmann_energy'], bins=bins_boltzmann_energy, density=True)
hist_samples, _ = np.histogram(boltzmann_energies, bins=bins_boltzmann_energy, density=True)

plt.figure(figsize=(12, 6))
plt.bar(bins_boltzmann_energy[:-1], hist_reference, width=np.diff(bins_boltzmann_energy), alpha=0.5, label='Reference Boltzmann Energy')
plt.step(bins_boltzmann_energy[:-1]-np.diff(bins_boltzmann_energy)/2, hist_samples, where='post', label='Sampled Boltzmann Energy', color='orange')

plt.xlabel('Boltzmann Energy/$H_\\mathrm{ising}(s)$')
plt.ylabel('Frequency')
plt.title('Boltzmann energy')
plt.legend()
plt.savefig(f'plots/boltzmann_energy_comparison_{n_visible_spins}_{alpha}.pdf')
plt.show()

##### Compare quantum energies with reference data ##### 
bins_quantum_energy = make_bins(df_reference['quantum_energy'], num_bins=50)
hist_reference_quantum, _ = np.histogram(df_reference['quantum_energy'], bins=bins_quantum_energy, density=True)
hist_samples_quantum, _ = np.histogram(quantum_energies, bins=bins_quantum_energy, density=True)

plt.figure(figsize=(12, 6))
plt.bar(bins_quantum_energy[:-1], hist_reference_quantum, width=np.diff(bins_quantum_energy), alpha=0.5, label='Reference quantum Energy')
plt.step(bins_quantum_energy[:-1]-np.diff(bins_quantum_energy)/2, hist_samples_quantum, where='post', label='Sampled quantum Energy', color='orange')

plt.xlabel('Quantum Energy / $\\langle E \\rangle$')
plt.ylabel('Frequency')
plt.title('Quantum energy')
plt.legend()
plt.savefig(f'plots/quantum_energy_comparison_{n_visible_spins}_{alpha}.pdf')
plt.show()

##### Comparison score #####
reference_quantum_energy = np.mean(df_reference['quantum_energy'][df_reference['sum_visible_spins']==0])

relative_error_quantum_energy = np.abs(variational_energy - reference_quantum_energy) / np.abs(reference_quantum_energy)
print(f"Relative error in variational energy: {relative_error_quantum_energy:e}")