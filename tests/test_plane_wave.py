"""DFT of a complex plane wave x[n] = exp(j 2 pi k0 n / N).

Analytical result: X[k] = N * delta_{k, k0}.
"""

import numpy as np
import pytest

from fft import fft_rec, fft_it

TOL = 1e-9


@pytest.mark.parametrize("k0", [0, 1, 25, 120, 511])
def test_plane_wave(k0):
    N = 1024
    n = np.arange(N)
    x = np.exp(2j * np.pi * k0 * n / N)

    Xa = np.zeros(N, dtype=complex)
    Xa[k0] = N

    Xr = fft_rec(x)
    Xi = fft_it(x)

    assert np.linalg.norm(Xr - Xa) < TOL
    assert np.linalg.norm(Xi - Xa) < TOL
