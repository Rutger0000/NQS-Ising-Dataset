
from functools import partial
import numpy as np
from numba import njit
import os
import json

@njit 
def boltzmann_energy(J, h, s) -> float:
    """
    Calculates the Boltzmann energy of a given state s for a Boltzmann machine/Ising model defined by the connectivity matrix J and field vector h. The energy is calculated using the formula E(s) = -((s.T @ (J @ s)) + s@h), where s is the state vector, J is the connectivity matrix, and h is the field vector.
    
    :param J: Connectivity matrix of the Boltzmann machine/Ising model (shape [all_spins, all_spins])
    :param h: Field/bias vector of the Boltzmann machine/Ising model (shape [all_spins])
    :param s: State vector of the Boltzmann machine/Ising model (shape [batch, all_spins])
    :return: Boltzmann energy of the state s
    :rtype: float
    """
    return -((s.T @ (J @ s)) + s@h)

def log_psi_batched(w, b, s, gamma=1.0) -> float:
    """
    Calculates log_psi for a given state s (only visible spins of RBM) and parameters w and b of the RBM.
    
    :param w: Weight matrix of the RBM (shape [n_visible, n_hidden])
    :param b: Bias vector of the RBM (shape [n_hidden])
    :param s: State vector (visible spins) of the RBM (shape [batch, n_visible])
    :param gamma: Gamma parameter for the Boltzmann machine
    :return: log_psi value for the given state and parameters
    :rtype: float
    """
    if s.ndim == 2:
        return gamma*np.sum(np.log(2*np.cosh(s @ w + b)), axis=1)
    else:
        return gamma*np.sum(np.log(2*np.cosh(b + w.T @ s)))

# By Dominique Kösters
def e_local(
        rbm,
        psi_log: np.ndarray,
        states: np.ndarray,
        bonds: np.ndarray,
        marshall_sign_rule: bool = True
    ) -> np.ndarray:
        """Computes the local energy corresponding to the states states 

        Args:
            lattice_bonds (list[tuple[int]]): The bonds of the 1D heisenberg chain 
            states (np.ndarray): set of staits states 

        Returns:
            np.ndarray: return a numpy 1D array corresponding to the local energy of the given states 
        """
        # initializes e local with zeros and set dtype to the dtype of the model (thus can be complex)
        e_loc = np.zeros(states.shape[0])

        sign = -1 if marshall_sign_rule else 1
        
        # loop over the bonds
        for bond in bonds:
            # add +/- one depending on the spins in a bond 
            e_loc += states[:, bond[0]] * states[:, bond[1]]
            
            # create a mask True if the two spins in a bond are opposite 
            mask = (states[:, bond[0]] != states[:, bond[1]]) 
            
            # if the bonds are not equal pick the whole state and flip the two spins in the bond 
            states_flipped = states[np.array(mask)]
            states_flipped[:, bond[0]] *= -1
            states_flipped[:, bond[1]] *= -1
            
            # compute the contribution of this spin flip 
            e_loc[mask] += 2 * sign * np.exp(rbm(states_flipped) - psi_log[np.array(mask)]) 
            
        # return the accumulated e_loc per state 
        return e_loc


def load_bonds(nspins):
    """
    Load a JSON file with specified base path and filename format `all_bonds_{nspins}.json`.
    
    Args:
    - base_path (str): The base directory where the JSON file is located.
    - nspins (int): The number of spins to be included in the filename.
    
    Returns:
    - data: The contents of the JSON file.
    """
    filename = os.path.join("data/bonds", f"all_bonds_{nspins}.json")
    try:
        with open(filename, 'r') as file:
            data = json.load(file)
        return np.array(data)-1
    except FileNotFoundError:
        print(f"File '{filename}' not found.")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON in file '{filename}': {e}")
        return None
    
def local_energies(states, w_rbm, b_rbm, bonds):
    """
    Calculates the local energies for a set of states given the parameters of an RBM and the bonds of the system.
    
    :param states: Array of shape (batch, n_visible) representing the states for which to calculate local energies
    :param w_rbm: Weight matrix of the RBM (shape [n_visible, n_hidden])
    :param b_rbm: Bias vector of the RBM (shape [n_hidden])
    :param bonds: Array of shape (n_bonds, 2) representing the pairs of spins that are connected by bonds in the system
    :return: Array of local energies per spin for the given states
    :rtype: np.ndarray
    """
    rbm_wavefunction = partial(log_psi_batched, w_rbm, b_rbm, gamma=0.5)

    nspins = w_rbm.shape[0]

    psi_logs = rbm_wavefunction(states)
    
    return e_local(rbm_wavefunction, psi_logs, states, bonds) / (nspins * 4)

def evaluate(n_visible_spins, alpha, states, J, h, bonds):
    n_visible_spins = n_visible_spins
    n_hidden_spins = alpha * n_visible_spins
    n_total_spins = n_visible_spins + n_hidden_spins

    assert states.shape[1] == n_total_spins, f"Expected states to have shape (batch, {n_total_spins}), but got {states.shape}"
    assert J.shape == (n_total_spins, n_total_spins), f"Expected J to have shape ({n_total_spins}, {n_total_spins}), but got {J.shape}"
    assert h.shape == (n_total_spins,), f"Expected h to have shape ({n_total_spins},), but got {h.shape}"
    assert bonds.shape[1] == 2, f"Expected bonds to have shape (n_bonds, 2), but got {bonds.shape}"

    # calculate the energy by Ising Hamiltonian/Boltzmann machine
    boltzmann_energies = np.array([boltzmann_energy(J, h, s) for s in states])

    # For Neural Quantum States we only need the visible part of the state
    visible = states[:,n_hidden_spins:] # [batch, visible spins]
    # Extract the parameters of the RBM from the J and h of the Boltzmann machine
    w_rbm = J[n_hidden_spins:, :n_hidden_spins] * 2
    assert w_rbm.shape == (n_visible_spins, n_hidden_spins), f"Expected w_rbm to have shape ({n_visible_spins}, {n_hidden_spins}), but got {w_rbm.shape}"
    b_rbm = h[:n_hidden_spins] * 2
    assert b_rbm.shape == (n_hidden_spins,), f"Expected b_rbm to have shape ({n_hidden_spins},), but got {b_rbm.shape}"

    # Calculates the log psi(s) for the given states
    log_psi = log_psi_batched(w_rbm, b_rbm, visible, gamma=0.5)

    # Calculate the quantum energies, the average corresponds to the variational energy of the quantum state
    quantum_energies = local_energies(visible, w_rbm, b_rbm, bonds)

    # Calculate the total spin for the visible part of the state
    sum_visible_spins = np.sum(visible, axis=1)

    # Calculate average variational energy over the states with magnetization zero (sum of visible spins is zero)
    variational_energy = np.mean(quantum_energies[sum_visible_spins==0])
    print(f"Variational energy for alpha={alpha} and nspins={n_visible_spins}: {variational_energy}")
    variational_energy_sem = np.std(quantum_energies[sum_visible_spins==0])/np.sqrt(len(quantum_energies[sum_visible_spins==0]))
    print(f"Standard error of the mean for the variational energy: {variational_energy_sem}")

    return boltzmann_energies, quantum_energies, log_psi, sum_visible_spins, variational_energy, variational_energy_sem
