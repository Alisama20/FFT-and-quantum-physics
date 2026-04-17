"""DFT of a Kronecker delta delta_{n, j0}.

Analytical result: X[k] = exp(-j 2 pi k j0 / N).
"""

import numpy as np
import pytest

from fft import fft_rec, fft_it

TOL = 1e-10


@pytest.mark.parametrize("j0", [0, 1, 50, 123])
def test_delta_signal(j0):
    N = 256
    x = np.zeros(N)
    x[j0] = 1.0

    k = np.arange(N)
    Xa = np.exp(-2j * np.pi * k * j0 / N)

    Xr = fft_rec(x)
    Xi = fft_it(x)

    assert np.linalg.norm(Xr - Xa) < TOL
    assert np.linalg.norm(Xi - Xa) < TOL
