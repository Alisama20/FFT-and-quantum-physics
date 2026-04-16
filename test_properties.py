"""Numerical verification of DFT theoretical properties:
  * F^-1 F = I
  * F F^-1 = I
  * F^4 = N^2 I
  * Norm conservation (Parseval)

Tested on random complex Gaussian vectors and averaged over many runs.
"""

import numpy as np
import matplotlib.pyplot as plt
from fft_core import fft_rec, fft_it

# ------------------------------
# Parameters
# ------------------------------

Ns = np.array([2**2, 2**3, 2**4, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384])
num_tests = 100


def norm2(v):
    return np.linalg.norm(v)


def F(x):
    return fft_it(x, inverse=False)

# ------------------------------
# Storage
# ------------------------------

err_FinvF_rec, err_FinvF_it = [], []
err_FFinv_rec, err_FFinv_it = [], []
err_F4_rec, err_F4_it = [], []
err_norm_rec, err_norm_it = [], []

# ------------------------------
# Main loop
# ------------------------------

for N in Ns:
    print(f"Processing N = {N}")

    e1r, e1i = [], []
    e2r, e2i = [], []
    e3r, e3i = [], []
    e4r, e4i = [], []

    for _ in range(num_tests):

        x = np.random.randn(N) + 1j * np.random.randn(N)

        Xr = fft_rec(x)
        Xi = fft_it(x)

        # 1) F^{-1} F = I
        e1r.append(norm2(fft_rec(Xr, inverse=True) - x))
        e1i.append(norm2(fft_it(Xi, inverse=True) - x))

        # 2) F F^{-1} = I
        e2r.append(norm2(F(fft_rec(x, inverse=True)) - x))
        e2i.append(norm2(F(fft_it(x, inverse=True)) - x))

        # 3) F^4 = N^2 I
        y = F(F(F(F(x))))
        e3r.append(norm2(y / (N * N) - x))
        e3i.append(norm2(y / (N * N) - x))

        # 4) Parseval: ||x|| = ||X|| / sqrt(N)
        e4r.append(abs(norm2(x) - norm2(Xr) / np.sqrt(N)))
        e4i.append(abs(norm2(x) - norm2(Xi) / np.sqrt(N)))

    err_FinvF_rec.append(np.mean(e1r))
    err_FinvF_it.append(np.mean(e1i))
    err_FFinv_rec.append(np.mean(e2r))
    err_FFinv_it.append(np.mean(e2i))
    err_F4_rec.append(np.mean(e3r))
    err_F4_it.append(np.mean(e3i))
    err_norm_rec.append(np.mean(e4r))
    err_norm_it.append(np.mean(e4i))

# ------------------------------
# 2 x 2 plot
# ------------------------------

log2N = np.log2(Ns)
fig, axs = plt.subplots(2, 2, figsize=(10, 8))
fig.suptitle("Numerical verification of DFT properties", fontsize=14)

axs[0, 0].plot(log2N, err_FinvF_rec, "o--", label="Recursive")
axs[0, 0].plot(log2N, err_FinvF_it, "s--", label="Iterative")
axs[0, 0].set_title(r"Error $F^{-1}F=I$")
axs[0, 0].set_ylabel("L2 error")
axs[0, 0].grid(True)
axs[0, 0].legend()

axs[0, 1].plot(log2N, err_FFinv_rec, "o--", label="Recursive")
axs[0, 1].plot(log2N, err_FFinv_it, "s--", label="Iterative")
axs[0, 1].set_title(r"Error $FF^{-1}=I$")
axs[0, 1].grid(True)
axs[0, 1].legend()

axs[1, 0].plot(log2N, err_F4_rec, "o--", label="Recursive")
axs[1, 0].plot(log2N, err_F4_it, "s--", label="Iterative")
axs[1, 0].set_title(r"Error $F^{4}=N^{2}I$")
axs[1, 0].set_xlabel(r"$\log_{2} N$")
axs[1, 0].set_ylabel("L2 error")
axs[1, 0].grid(True)
axs[1, 0].legend()

axs[1, 1].plot(log2N, err_norm_rec, "o--", label="Recursive")
axs[1, 1].plot(log2N, err_norm_it, "s--", label="Iterative")
axs[1, 1].set_title("Norm conservation error")
axs[1, 1].set_xlabel(r"$\log_{2} N$")
axs[1, 1].grid(True)
axs[1, 1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/properties.png", dpi=150)
plt.show()
