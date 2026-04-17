"""DFT of a rectangular pulse x[n] = 1 for a <= n < b, 0 elsewhere.

Analytical result (closed form geometric sum)::

    X[k] = exp(-j 2 pi k a / N) * (1 - exp(-j 2 pi k L / N)) /
                                   (1 - exp(-j 2 pi k   / N)),

with L = b - a, and X[0] = L.
"""

import numpy as np

from fft import fft_rec, fft_it

TOL = 1e-10


def analytical_pulse_dft(N, a, b):
    L = b - a
    Xa = np.zeros(N, dtype=complex)
    k = np.arange(N)
    for i, ki in enumerate(k):
        if ki == 0:
            Xa[i] = L
        else:
            num = 1.0 - np.exp(-2j * np.pi * ki * L / N)
            den = 1.0 - np.exp(-2j * np.pi * ki / N)
            Xa[i] = np.exp(-2j * np.pi * ki * a / N) * num / den
    return Xa


def test_square_pulse():
    N = 1024
    a, b = N // 4, 3 * N // 4

    x = np.zeros(N)
    x[a:b] = 1.0

    Xa = analytical_pulse_dft(N, a, b)
    Xr = fft_rec(x)
    Xi = fft_it(x)

    assert np.linalg.norm(Xr - Xa) < TOL
    assert np.linalg.norm(Xi - Xa) < TOL
