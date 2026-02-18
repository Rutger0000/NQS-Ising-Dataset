# Dataset for mapping Neural Quantum States to Ising Model/QUBO

Neural Quantum States (NQS) is a technique for representing quantum many-body
wavefunctions using neural networks. Here, we use a Restricted Boltzmann Machine
(RBM) to represent the wavefunction of a quantum system. A restricted Boltzmann
machine can be mapped to an Ising model or QUBO (Quadratic Unconstrained Binary
Optimization) problem. This dataset provides pre-trained RBMs/Ising models which
sample the ground state of a 2D Quantum Heisenberg model trained using NQS as
described in
[https://arxiv.org/abs/2504.18359](https://arxiv.org/abs/2504.18359).

There are many different conventions for the Ising model/QUBO representation. Here, we
use the following Hamiltonian for the Ising model:

$$
H(s) = - \sum_{i=1,j=1}^{n+\alpha n} J_{ij} s_i s_j - \sum_{i=1}^{n+\alpha n} h_i s_i
$$

where `s_i` are spin variables taking values in {-1, 1}, $J_{ij}$ (`J`)
 is the connectivity matrix, and $h_i$ (`h`) is the bias vector. $n$ is the number of spins in the original quantum system and the number of visible spins of the RBM. $\alpha$ is the hidden unit density
of the RBM, defined as the ratio of the number of hidden units to the number of visible units. The probability of a spin configuration `s` is given by the Boltzmann distribution:

$$
P(s) = \frac{1}{Z} e^{-H(s)}
$$

where `Z` is the partition function.

The connectivity matrix `J` and bias vector `h` can be found in the `weights/`
folder. They are stored as CSV files named `W_ising_{visible nspins}_{alpha}_ti_W.csv`
and `W_ising_{visible nspins}_{alpha}_ti_b.csv`, respectively. 

## Example energies

In order to verify the convention how we calculate the Ising energy, we provide some
sample spin configurations and their corresponding energies below for `weights/W_ising_16_2_ti_W.csv` and `weights/W_ising_16_2_ti_b.csv`.

```
========== Sample ==========
s = [-1 -1 -1  1 -1  1  1 -1  1 -1 -1  1 -1  1  1 -1 -1 -1 -1  1  1  1  1 -1 -1 -1  1  1  1 -1  1  1 -1  1 -1 -1 -1  1  1 -1 -1  1 -1 -1 -1 -1  1 -1]
H(s) =  -30.5037869817115
========== Sample ==========
s = [-1  1 -1  1 -1  1 -1  1 -1  1 -1  1 -1  1 -1  1 -1  1 -1  1 -1  1 -1  1 -1  1 -1  1 -1  1 -1  1 -1  1 -1  1 -1  1 -1  1 -1  1 -1  1 -1  1 -1  1]
H(s) =  87.6942182447498
========== Sample ==========
s = [1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1]
H(s) =  0.18833603906623805
========== Sample ==========
s = [-1  1 -1  1  1 -1  1 -1 -1  1 -1  1  1 -1  1 -1  1  1 -1 -1 -1  1 -1  1  1 -1  1 -1 -1  1 -1  1  1 -1  1 -1 -1  1 -1  1  1 -1  1 -1 -1  1 -1  1]
H(s) =  -93.41480334399733
========== Sample ==========
s = [-1  1 -1  1  1 -1  1 -1 -1  1 -1  1  1 -1  1 -1  1 -1  1 -1 -1  1 -1  1  1 -1  1 -1 -1  1 -1 -1  1 -1  1 -1 -1  1 -1  1  1 -1  1 -1 -1  1 -1  1]
H(s) =  -98.96928063623803
```

## How to test the accuracy of sampled configurations
To test the accuracy, `benchmark.ipynb` can be used to load the sampled states and
calculate the `H(s)` (Energy of Ising Model) and the quantum energy $\langle E
\rangle$. The energy of the Ising model and the quantum energy are then compared
to earlier experiments which are stored in
`data/observables_{nspins}_{alpha}.parquet`. 

To load other samples and test their accuracy, use `benchmark.py` and modify lines 38/39:

```python
df_raw_states = pd.read_parquet(f'data/states_{n_visible_spins}_{alpha}.parquet') # THIS IS AN EXAMPLE FILE, REPLACE WITH YOUR OWN DATA
raw_states = df_raw_states.values[:4000, :total_spins]
```

