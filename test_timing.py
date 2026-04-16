"""Timing comparison: recursive FFT, iterative FFT, NumPy FFT.

Checks the O(N log N) scaling by normalising t(N) by N log2(N).
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from fft_core import fft_rec, fft_it

# ------------------------------
# Parameters
# ------------------------------

Ns = np.array([2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048,
               4096, 8192, 16384, 32768, 65536, 131072])
num_tests = 1

# ------------------------------
# Storage
# ------------------------------

time_rec = []
time_it = []
time_np = []

# ------------------------------
# Measure
# ------------------------------

for N in Ns:
    print(f"Processing N = {N}")
    t_rec, t_it, t_np = [], [], []

    for _ in range(num_tests):
        x = np.random.randn(N) + 1j * np.random.randn(N)

        t0 = time.perf_counter()
        fft_rec(x)
        t_rec.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        fft_it(x)
        t_it.append(time.perf_counter() - t0)

        t0 = time.perf_counter()
        np.fft.fft(x)
        t_np.append(time.perf_counter() - t0)

    time_rec.append(np.mean(t_rec))
    time_it.append(np.mean(t_it))
    time_np.append(np.mean(t_np))

time_rec = np.array(time_rec)
time_it = np.array(time_it)
time_np = np.array(time_np)
log2N = np.log2(Ns)
theory = Ns * np.log2(Ns)

# ------------------------------
# Plot
# ------------------------------

fig, axs = plt.subplots(1, 2, figsize=(11, 4))
fig.suptitle("Complexity: O(N log N)", fontsize=14)

axs[0].plot(log2N, time_rec, "o--", label="Recursive")
axs[0].plot(log2N, time_it, "s--", label="Iterative")
axs[0].plot(log2N, time_np, "^-", label="NumPy")
axs[0].set_yscale("log")
axs[0].set_xlabel(r"$\log_{2} N$")
axs[0].set_ylabel("Time (s)")
axs[0].set_title("Execution time")
axs[0].grid(True, which="both", alpha=0.4)
axs[0].legend()

axs[1].plot(log2N, time_rec / theory, "o--", label="Recursive")
axs[1].plot(log2N, time_it / theory, "s--", label="Iterative")
axs[1].plot(log2N, time_np / theory, "^-", label="NumPy")
axs[1].set_xlabel(r"$\log_{2} N$")
axs[1].set_ylabel(r"$T(N)/(N\log_{2}N)$")
axs[1].set_title(r"Normalised by $N\log_{2}N$")
axs[1].grid(True)
axs[1].legend()

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.savefig("figures/timings.png", dpi=150)
plt.show()
