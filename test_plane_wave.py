"""DFT of a complex plane wave x[n] = exp(j 2 pi k0 n / N).

The exact DFT is a Kronecker delta in frequency: X[k] = N * delta_{k, k0}.
"""

import numpy as np
import matplotlib.pyplot as plt
from fft_core import fft_rec, fft_it

# ------------------------------
# Parameters
# ------------------------------

N = 1024
k = np.arange(-N // 2, N // 2)
n = np.arange(N)

k0_values = [25, 120]

# ------------------------------
# Loop over frequencies
# ------------------------------

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle(r"$x[n]=e^{\,j 2\pi k_0 n / N}$", fontsize=14)

for ax, k0 in zip(axes, k0_values):

    x = np.exp(2j * np.pi * k0 * n / N)

    Xr = fft_rec(x)
    Xi = fft_it(x)
    Xn = np.fft.fft(x)

    # Analytical DFT: N at index k0, zero elsewhere
    Xa = np.zeros(N, dtype=complex)
    Xa[k0] = N

    Xr_s = np.fft.fftshift(Xr)
    Xi_s = np.fft.fftshift(Xi)
    Xn_s = np.fft.fftshift(Xn)
    Xa_s = np.fft.fftshift(Xa)

    err_rec = np.linalg.norm(Xr - Xa)
    err_it = np.linalg.norm(Xi - Xa)
    err_np = np.linalg.norm(Xn - Xa)

    ax.plot(k, np.abs(Xa_s), label="Analytical", linewidth=2)
    ax.plot(k, np.abs(Xr_s), "--", label=f"Recursive (Err = {err_rec:.2e})")
    ax.plot(k, np.abs(Xi_s), ":", label=f"Iterative (Err = {err_it:.2e})")
    ax.plot(k, np.abs(Xn_s), "-.", label=f"NumPy (Err = {err_np:.2e})")

    ax.set_title(rf"$k_0 = {k0}$")
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$|X[k]|$")
    ax.set_xlim(k0 - 40, k0 + 40)
    ax.grid(True)
    ax.legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/plane_wave.png", dpi=150)
plt.show()
