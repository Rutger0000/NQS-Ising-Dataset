import numpy as np

def boltzmann_energy(w, th, s) -> float:
    return -((s.T @ (w @ s)) + s@th)

def main():

    nspins = 16
    alpha = 2

    W_weights = np.genfromtxt(f'weights/W_ising_{nspins}_{alpha}_ti_W.csv', delimiter=',')
    b_biases = np.genfromtxt(f'weights/W_ising_{nspins}_{alpha}_ti_b.csv', delimiter=',')

    all_ising_spins = nspins * alpha + nspins

    s_random = np.random.choice([-1, 1], size=(nspins*alpha + nspins))
    s_updown = np.array([-1,1]*(all_ising_spins//2))
    s_allup = np.ones(nspins*alpha + nspins)

    np.set_printoptions(linewidth=400)

    energy_random = boltzmann_energy(W_weights, b_biases, s_random)
    energy_updown = boltzmann_energy(W_weights, b_biases, s_updown)
    energy_allup = boltzmann_energy(W_weights, b_biases, s_allup)

    print("========== Sample ==========")
    print("s =", s_random)
    print("H(s) = ", energy_random)
    print("========== Sample ==========")
    print("s =", s_updown)
    print("H(s) = ", energy_updown)
    print("========== Sample ==========")
    print("s =", s_allup)
    print("H(s) = ", energy_allup)

if __name__ == "__main__":
    main()
