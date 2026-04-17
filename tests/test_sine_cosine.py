"""DFT of a superposition of sines and cosines at integer frequencies.

For integer frequencies 0 < f < N/2::

    sin(2 pi f n / N)  ->  (-j N/2) at k = f, (+j N/2) at k = N - f
    cos(2 pi f n / N)  ->  ( N/2 ) at k = f, ( N/2 ) at k = N - f
"""

import numpy as np

from fft import fft_rec, fft_it

TOL = 1e-9


def test_sines_and_cosines():
    N = 1024
    n = np.arange(N)

    freqs_sin = [15, 40, 90]
    amps_sin = [1.0, 0.6, 0.4]
    freqs_cos = [25, 70]
    amps_cos = [0.8, 0.5]

    x = np.zeros(N)
    for f, a in zip(freqs_sin, amps_sin):
        x += a * np.sin(2 * np.pi * f * n / N)
    for f, a in zip(freqs_cos, amps_cos):
        x += a * np.cos(2 * np.pi * f * n / N)

    Xa = np.zeros(N, dtype=complex)
    for f, a in zip(freqs_sin, amps_sin):
        Xa[f] += -1j * a * N / 2
        Xa[N - f] += 1j * a * N / 2
    for f, a in zip(freqs_cos, amps_cos):
        Xa[f] += a * N / 2
        Xa[N - f] += a * N / 2

    Xr = fft_rec(x)
    Xi = fft_it(x)

    assert np.linalg.norm(Xr - Xa) < TOL
    assert np.linalg.norm(Xi - Xa) < TOL
