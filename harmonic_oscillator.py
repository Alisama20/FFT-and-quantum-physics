"""Ground state of the 1D quantum harmonic oscillator via FFT.

We use the split-operator (Trotter) method in imaginary time:

    exp(-H tau) ~ exp(-T dt/2) exp(-V dt) exp(-T dt/2)

with T = p^2 / 2 diagonal in momentum space and V = x^2 / 2 diagonal in
position space.  The kinetic kick is applied via FFT:

    psi_k  = FFT(psi_x)
    psi_k *= exp(-k^2 dt / 4)           # half step (applied twice)
    psi_x  = IFFT(psi_k)
    psi_x *= exp(-V(x) dt)

After each step the wavefunction is renormalised, so imaginary-time
evolution converges to the ground state.  In natural units
(hbar = m = omega = 1) the exact ground-state energy is E_0 = 1/2 and
psi_0(x) = pi^(-1/4) exp(-x^2 / 2).
"""

import numpy as np
import matplotlib.pyplot as plt
from fft_core import fft_it

# ------------------------------
# Parameters (natural units: hbar = m = omega = 1)
# ------------------------------

N = 1024                  # must be a power of 2 for the iterative FFT
L = 20.0                  # box [-L/2, L/2]
dx = L / N
x = np.linspace(-L / 2, L / 2 - dx, N)

# Momentum grid compatible with numpy-style FFT ordering
k = 2 * np.pi * np.fft.fftfreq(N, d=dx)

dt = 0.01                 # imaginary-time step
n_steps = 4000            # total number of Trotter steps

# ------------------------------
# Potential and kinetic propagators
# ------------------------------

V = 0.5 * x ** 2
expV = np.exp(-V * dt)
expT_half = np.exp(-0.5 * k ** 2 * dt / 2)   # half-step kinetic

# ------------------------------
# Initial trial wavefunction (off-centre Gaussian to avoid symmetry lock)
# ------------------------------

psi = np.exp(-(x - 0.7) ** 2).astype(complex)
psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)

# ------------------------------
# Imaginary-time propagation
# ------------------------------

energies = []

for step in range(n_steps):

    # Half kinetic kick in momentum space
    psi_k = fft_it(psi)
    psi_k *= expT_half
    psi = fft_it(psi_k, inverse=True)

    # Full potential kick in position space
    psi *= expV

    # Half kinetic kick in momentum space
    psi_k = fft_it(psi)
    psi_k *= expT_half
    psi = fft_it(psi_k, inverse=True)

    # Renormalise
    norm = np.sqrt(np.sum(np.abs(psi) ** 2) * dx)
    psi /= norm

    # Energy estimator every 50 steps
    if step % 50 == 0:
        psi_k = fft_it(psi)
        T_psi = fft_it(0.5 * k ** 2 * psi_k, inverse=True)
        E_kin = np.real(np.sum(np.conj(psi) * T_psi) * dx)
        E_pot = np.real(np.sum(np.conj(psi) * V * psi) * dx)
        energies.append(E_kin + E_pot)

# ------------------------------
# Final energy and comparison
# ------------------------------

psi_k = fft_it(psi)
T_psi = fft_it(0.5 * k ** 2 * psi_k, inverse=True)
E_kin = np.real(np.sum(np.conj(psi) * T_psi) * dx)
E_pot = np.real(np.sum(np.conj(psi) * V * psi) * dx)
E0 = E_kin + E_pot

print(f"FFT split-operator ground-state energy : {E0:.8f}")
print(f"Exact ground-state energy              : 0.5")
print(f"Absolute error                         : {abs(E0 - 0.5):.2e}")

# ------------------------------
# Plot |psi|^2 against the analytic Gaussian ground state
# ------------------------------

psi_exact = np.pi ** (-0.25) * np.exp(-x ** 2 / 2)

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Quantum harmonic oscillator ground state via FFT", fontsize=14)

axs[0].plot(x, np.abs(psi) ** 2, label=r"FFT $|\psi_0(x)|^2$")
axs[0].plot(x, psi_exact ** 2, "--", label=r"Exact $|\psi_0(x)|^2$")
axs[0].set_xlim(-5, 5)
axs[0].set_xlabel(r"$x$")
axs[0].set_ylabel(r"$|\psi_0(x)|^2$")
axs[0].set_title("Ground-state density")
axs[0].grid(True)
axs[0].legend()

axs[1].plot(np.arange(len(energies)) * 50, energies, "o-", markersize=3,
            label="Energy estimator")
axs[1].axhline(0.5, color="k", ls="--", label=r"Exact $E_0 = 1/2$")
axs[1].set_xlabel("Imaginary-time step")
axs[1].set_ylabel(r"$\langle H \rangle$")
axs[1].set_title("Convergence to the ground-state energy")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/harmonic_oscillator.png", dpi=150)
plt.show()
