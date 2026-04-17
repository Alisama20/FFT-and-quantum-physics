"""DFT of Kronecker deltas delta_{n,j0} — numerical vs analytical."""

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

# Positions j0 of the non-zero sample
j0_values = [0, 50]

# ------------------------------
# Main loop: one figure per delta
# ------------------------------

for j0 in j0_values:

    # Delta signal
    x = np.zeros(N)
    x[j0] = 1

    # FFTs
    Xr = fft_rec(x)
    Xi = fft_it(x)
    Xn = np.fft.fft(x)

    # Analytical DFT: X[k] = exp(-j 2 pi k j0 / N)
    k0 = np.arange(N)
    Xa = np.exp(-2j * np.pi * k0 * j0 / N)

    # Center
    Xr_s = np.fft.fftshift(Xr)
    Xi_s = np.fft.fftshift(Xi)
    Xn_s = np.fft.fftshift(Xn)
    Xa_s = np.fft.fftshift(Xa)

    # L2 errors
    err_rec = np.linalg.norm(Xr_s - Xa_s)
    err_it = np.linalg.norm(Xi_s - Xa_s)

    # 2 x 2 plot: Re, Im, |X|, arg(X)
    fig, axs = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle(rf"$x[n]=\delta_{{n,{j0}}}$", fontsize=14)

    axs[0, 0].plot(k, Xa_s.real, label="Analytical", linewidth=2)
    axs[0, 0].plot(k, Xr_s.real, "--", label="Recursive")
    axs[0, 0].plot(k, Xi_s.real, ":", label="Iterative")
    axs[0, 0].plot(k, Xn_s.real, "-.", label="NumPy")
    axs[0, 0].set_title("Real part")
    axs[0, 0].grid(True)
    axs[0, 0].legend()

    axs[0, 1].plot(k, Xa_s.imag, label="Analytical", linewidth=2)
    axs[0, 1].plot(k, Xr_s.imag, "--", label="Recursive")
    axs[0, 1].plot(k, Xi_s.imag, ":", label="Iterative")
    axs[0, 1].plot(k, Xn_s.imag, "-.", label="NumPy")
    axs[0, 1].set_title("Imaginary part")
    axs[0, 1].grid(True)

    axs[1, 0].plot(k, np.abs(Xa_s), label="Analytical", linewidth=2)
    axs[1, 0].plot(k, np.abs(Xr_s), "--", label="Recursive")
    axs[1, 0].plot(k, np.abs(Xi_s), ":", label="Iterative")
    axs[1, 0].plot(k, np.abs(Xn_s), "-.", label="NumPy")
    axs[1, 0].set_title("Magnitude")
    axs[1, 0].grid(True)

    axs[1, 1].plot(k, np.angle(Xa_s), label="Analytical", linewidth=2)
    axs[1, 1].plot(k, np.angle(Xr_s), "--", label="Recursive")
    axs[1, 1].plot(k, np.angle(Xi_s), ":", label="Iterative")
    axs[1, 1].plot(k, np.angle(Xn_s), "-.", label="NumPy")
    axs[1, 1].set_title("Phase")
    axs[1, 1].grid(True)

    for ax in axs.flat:
        ax.set_xlim(-N // 2, N // 2)
        ax.set_xlabel(r"$k$")

    fig.text(
        0.5, 0.01,
        f"L2 Rec = {err_rec:.2e}   |   L2 It = {err_it:.2e}",
        ha="center",
    )

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"figures/delta{j0}.png", dpi=150)
    plt.show()
