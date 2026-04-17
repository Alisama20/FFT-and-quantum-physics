"""Black-box agreement with numpy.fft on random complex signals."""

import numpy as np
import pytest

from fft import fft_rec, fft_it

TOL = 1e-9


@pytest.mark.parametrize("N", [2, 4, 8, 16, 64, 256, 1024])
def test_fft_matches_numpy(N):
    rng = np.random.default_rng(42)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    Xn = np.fft.fft(x)

    assert np.linalg.norm(fft_rec(x) - Xn) < TOL * max(N, 1)
    assert np.linalg.norm(fft_it(x) - Xn) < TOL * max(N, 1)


@pytest.mark.parametrize("N", [2, 4, 16, 128, 1024])
def test_ifft_matches_numpy(N):
    rng = np.random.default_rng(7)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    Yn = np.fft.ifft(x)

    assert np.linalg.norm(fft_rec(x, inverse=True) - Yn) < TOL * max(N, 1)
    assert np.linalg.norm(fft_it(x, inverse=True) - Yn) < TOL * max(N, 1)
