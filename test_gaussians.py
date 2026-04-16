"""DFT of Gaussian signals x[n] = exp(-(n-mu)^2 / (2 sigma^2))."""

import numpy as np
import matplotlib.pyplot as plt
from fft_core import fft_rec, fft_it


# ------------------------------
# Analytical reference: periodic-Gaussian DFT
# (sum over Brillouin replicas)
# ------------------------------

def gaussian_dft_analytic(k, sigma, mu, N, M=7):
    f = k / N
    X = np.zeros_like(f, dtype=complex)
    for m in range(-M, M + 1):
        term = (
            np.exp(-2 * np.pi**2 * sigma**2 * (f + m) ** 2)
            * np.exp(-2j * np.pi * (f + m) * mu)
        )
        X += term
    X *= sigma * np.sqrt(2 * np.pi)
    return X


# ------------------------------
# Configuration
# ------------------------------

N = 1024
mu = N / 2
n = np.arange(N)
k_indices = np.arange(-N // 2, N // 2)

sigmas = [10, 20, 40]

fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(r"$f[n] = e^{-(n-\mu)^2 / 2\sigma^2}$", fontsize=16)

# ------------------------------
# Loop over different widths
# ------------------------------

for ax, sigma in zip(axes, sigmas):

    x = np.exp(-(n - mu) ** 2 / (2 * sigma ** 2))

    X_rec_s = np.fft.fftshift(fft_rec(x))
    X_it_s = np.fft.fftshift(fft_it(x))
    X_np_s = np.fft.fftshift(np.fft.fft(x))
    X_ana_s = gaussian_dft_analytic(k_indices, sigma, mu, N)

    # Normalise analytical to match the discrete magnitude
    X_ana_s *= np.max(np.abs(X_np_s)) / np.max(np.abs(X_ana_s))

    err_rec = np.linalg.norm(X_rec_s - X_ana_s)
    err_it = np.linalg.norm(X_it_s - X_ana_s)
    err_np = np.linalg.norm(X_np_s - X_ana_s)

    ax.plot(k_indices, np.abs(X_ana_s), 'k-', linewidth=3, alpha=0.3,
            label='Analytical')
    ax.plot(k_indices, np.abs(X_np_s), 'go', markersize=2, alpha=0.6,
            label=f'NumPy (Err = {err_np:.1e})')
    ax.plot(k_indices, np.abs(X_rec_s), 'r--', linewidth=1.5,
            label=f'Recursive (Err = {err_rec:.1e})')
    ax.plot(k_indices, np.abs(X_it_s), 'b:', linewidth=2,
            label=f'Iterative (Err = {err_it:.1e})')

    ax.set_title(rf"$\sigma = {sigma}$", fontsize=12)
    ax.set_xlabel(r"Frequency $k$")
    ax.set_ylabel(r"Magnitude $|X[k]|$")
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(-120, 120)

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/gaussians.png", dpi=150)
plt.show()
