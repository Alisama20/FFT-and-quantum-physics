"""DFT of a square pulse — x[n] = 1 for N/4 <= n < 3N/4, 0 elsewhere."""

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

a = N // 4
b = 3 * N // 4
L = b - a

# ------------------------------
# Signal
# ------------------------------

x = np.zeros(N)
x[a:b] = 1

# ------------------------------
# FFTs
# ------------------------------

Xr = fft_rec(x)
Xi = fft_it(x)
Xn = np.fft.fft(x)

# ------------------------------
# Analytical DFT (geometric sum)
# ------------------------------

k0 = np.arange(N)
Xa = np.zeros(N, dtype=complex)
for i, ki in enumerate(k0):
    if ki == 0:
        Xa[i] = L
    else:
        num = 1 - np.exp(-2j * np.pi * ki * L / N)
        den = 1 - np.exp(-2j * np.pi * ki / N)
        Xa[i] = np.exp(-2j * np.pi * ki * a / N) * num / den

# Center
Xr_s = np.fft.fftshift(Xr)
Xi_s = np.fft.fftshift(Xi)
Xn_s = np.fft.fftshift(Xn)
Xa_s = np.fft.fftshift(Xa)

# L2 errors vs analytical
err_rec = np.linalg.norm(Xr_s - Xa_s)
err_it = np.linalg.norm(Xi_s - Xa_s)
err_np = np.linalg.norm(Xn_s - Xa_s)

# ------------------------------
# Magnitude plot
# ------------------------------

plt.figure(figsize=(9, 5))

plt.plot(k, np.abs(Xa_s), label="Analytical", linewidth=2)
plt.plot(k, np.abs(Xr_s), "--", label=f"Recursive (Err = {err_rec:.2e})")
plt.plot(k, np.abs(Xi_s), ":", label=f"Iterative (Err = {err_it:.2e})")
plt.plot(k, np.abs(Xn_s), "-.", label=f"NumPy (Err = {err_np:.2e})")

# Zoom to active region
mask = np.abs(Xa_s) > 0.01 * np.max(np.abs(Xa_s))
plt.xlim(k[mask][0] - 5, k[mask][-1] + 5)

plt.title(r"$x[n]=1\ \mathrm{for}\ N/4\leq n<3N/4,\ 0\ \mathrm{else}$",
          fontsize=14)
plt.xlabel(r"$k$")
plt.ylabel(r"$|X[k]|$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/square_pulse.png", dpi=150)
plt.show()
