"""DFT of a sum of sines and cosines — numerical vs analytical line spectrum."""

import numpy as np
import matplotlib.pyplot as plt
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from fft import fft_rec, fft_it

# ------------------------------
# Parameters
# ------------------------------

N = 1024
n = np.arange(N)

freqs_sin = [15, 40, 90]
amps_sin = [1.0, 0.6, 0.4]

freqs_cos = [25, 70]
amps_cos = [0.8, 0.5]

# ------------------------------
# Signal
# ------------------------------

x = np.zeros(N)
for f, amp in zip(freqs_sin, amps_sin):
    x += amp * np.sin(2 * np.pi * f * n / N)
for f, amp in zip(freqs_cos, amps_cos):
    x += amp * np.cos(2 * np.pi * f * n / N)

# ------------------------------
# FFTs
# ------------------------------

Xr = fft_rec(x)
Xi = fft_it(x)
Xn = np.fft.fft(x)

# ------------------------------
# Analytical spectrum (line spectrum at integer frequencies)
# sin contribution -> -jN/2 at +f,  +jN/2 at -f
# cos contribution ->  N/2 at +/- f
# ------------------------------

X_analytic = np.zeros(N, dtype=complex)
for f, amp in zip(freqs_sin, amps_sin):
    X_analytic[f] += -1j * amp * N / 2
    X_analytic[N - f] += 1j * amp * N / 2
for f, amp in zip(freqs_cos, amps_cos):
    X_analytic[f] += amp * N / 2
    X_analytic[N - f] += amp * N / 2

# Center
k = np.arange(-N // 2, N // 2)
Xr_s = np.fft.fftshift(Xr)
Xi_s = np.fft.fftshift(Xi)
Xn_s = np.fft.fftshift(Xn)
Xa_s = np.fft.fftshift(X_analytic)

# L2 errors
err_rec = np.linalg.norm(Xr - X_analytic)
err_it = np.linalg.norm(Xi - X_analytic)
err_np = np.linalg.norm(Xn - X_analytic)

# ------------------------------
# Magnitude plot
# ------------------------------

plt.figure(figsize=(9, 5))
plt.plot(k, np.abs(Xa_s), label="Analytical", linewidth=2)
plt.plot(k, np.abs(Xr_s), "--", label=f"Recursive (Err = {err_rec:.2e})")
plt.plot(k, np.abs(Xi_s), ":", label=f"Iterative (Err = {err_it:.2e})")
plt.plot(k, np.abs(Xn_s), "-.", label=f"NumPy (Err = {err_np:.2e})")

mask = np.abs(Xa_s) > 1e-6
plt.xlim(np.min(k[mask]) - 5, np.max(k[mask]) + 5)

plt.title(
    r"$x[n]=\sum_{f}a_f\sin(2\pi fn/N)+\sum_{f}b_f\cos(2\pi fn/N)$",
    fontsize=13,
)
plt.xlabel(r"$k$")
plt.ylabel(r"$|X[k]|$")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig("figures/sine_cosine.png", dpi=150)
plt.show()
