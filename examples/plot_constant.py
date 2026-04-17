"""DFT of the constant signal x[n] = 1 — numerical vs analytical."""

import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fft import fft_rec, fft_it

# ------------------------------
# Parameters
# ------------------------------

N = 1024
k = np.arange(-N // 2, N // 2)

# ------------------------------
# Constant signal
# ------------------------------

x = np.ones(N)

# ------------------------------
# FFTs
# ------------------------------

Xr = fft_rec(x)
Xi = fft_it(x)
Xn = np.fft.fft(x)

# ------------------------------
# Exact analytical DFT: N*delta[k]
# ------------------------------

Xa = np.zeros(N, dtype=complex)
Xa[0] = N

# ------------------------------
# Center (fftshift)
# ------------------------------

Xr_s = np.fft.fftshift(Xr)
Xi_s = np.fft.fftshift(Xi)
Xn_s = np.fft.fftshift(Xn)
Xa_s = np.fft.fftshift(Xa)

# ------------------------------
# L2 errors
# ------------------------------

err_rec = np.linalg.norm(Xr_s - Xa_s)
err_it = np.linalg.norm(Xi_s - Xa_s)
err_np = np.linalg.norm(Xn_s - Xa_s)

# ------------------------------
# Real/imaginary parts
# ------------------------------

fig, axs = plt.subplots(1, 2, figsize=(10, 4))
fig.suptitle(r"$x[n]=1$", fontsize=14)

axs[0].plot(k, Xa_s.real, label="Analytical", linewidth=2)
axs[0].plot(k, Xr_s.real, "--", label="Recursive")
axs[0].plot(k, Xi_s.real, ":", label="Iterative")
axs[0].plot(k, Xn_s.real, "-.", label="NumPy")
axs[0].set_title("Real part")
axs[0].set_xlabel(r"$k$")
axs[0].set_ylabel(r"Re$\{X[k]\}$")
axs[0].grid(True)
axs[0].legend()

axs[1].plot(k, Xa_s.imag, label="Analytical", linewidth=2)
axs[1].plot(k, Xr_s.imag, "--", label="Recursive")
axs[1].plot(k, Xi_s.imag, ":", label="Iterative")
axs[1].plot(k, Xn_s.imag, "-.", label="NumPy")
axs[1].set_title("Imaginary part")
axs[1].set_xlabel(r"$k$")
axs[1].set_ylabel(r"Im$\{X[k]\}$")
axs[1].grid(True)

fig.text(
    0.5, 0.02,
    f"L2 Rec = {err_rec:.2e}   |   L2 It = {err_it:.2e}   |   L2 NumPy = {err_np:.2e}",
    ha="center",
)

plt.tight_layout(rect=[0, 0.05, 1, 0.95])
plt.savefig("figures/constant.png", dpi=150)
plt.show()
