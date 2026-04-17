"""Numerical verification of fundamental DFT properties:

  * F^{-1} F = I          (inverse on the right)
  * F F^{-1} = I          (inverse on the left)
  * F^4 = N^2 I           (DFT to the fourth power)
  * Parseval: ||F x||_2 = sqrt(N) * ||x||_2
"""

import numpy as np
import pytest

from fft import fft_rec, fft_it

TOL = 1e-9


@pytest.mark.parametrize("impl", [fft_rec, fft_it])
@pytest.mark.parametrize("N", [4, 16, 64, 256, 1024])
def test_inverse_on_the_right(impl, N):
    rng = np.random.default_rng(0)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    y = impl(impl(x), inverse=True)
    assert np.linalg.norm(y - x) < TOL * np.linalg.norm(x)


@pytest.mark.parametrize("impl", [fft_rec, fft_it])
@pytest.mark.parametrize("N", [4, 16, 64, 256, 1024])
def test_inverse_on_the_left(impl, N):
    rng = np.random.default_rng(1)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    y = impl(impl(x, inverse=True))
    assert np.linalg.norm(y - x) < TOL * np.linalg.norm(x)


@pytest.mark.parametrize("impl", [fft_rec, fft_it])
@pytest.mark.parametrize("N", [4, 16, 64, 256])
def test_F_to_the_fourth(impl, N):
    rng = np.random.default_rng(2)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    y = impl(impl(impl(impl(x))))
    assert np.linalg.norm(y - (N ** 2) * x) < TOL * (N ** 2) * np.linalg.norm(x)


@pytest.mark.parametrize("impl", [fft_rec, fft_it])
@pytest.mark.parametrize("N", [4, 16, 64, 256, 1024])
def test_parseval(impl, N):
    rng = np.random.default_rng(3)
    x = rng.standard_normal(N) + 1j * rng.standard_normal(N)
    X = impl(x)
    lhs = np.linalg.norm(X)
    rhs = np.sqrt(N) * np.linalg.norm(x)
    assert abs(lhs - rhs) < TOL * rhs
