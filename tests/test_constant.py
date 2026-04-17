"""DFT of the constant signal x[n] = 1.

Analytical result: X[k] = N * delta[k].
"""

import numpy as np
import pytest

from fft import fft_rec, fft_it

TOL = 1e-10


@pytest.mark.parametrize("N", [16, 64, 1024])
def test_constant_signal(N):
    x = np.ones(N)
    Xa = np.zeros(N, dtype=complex)
    Xa[0] = N

    Xr = fft_rec(x)
    Xi = fft_it(x)

    assert np.linalg.norm(Xr - Xa) < TOL
    assert np.linalg.norm(Xi - Xa) < TOL
