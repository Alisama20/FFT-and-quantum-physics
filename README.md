# FFT and Quantum Physics

Self-contained **recursive** and **iterative** implementations of the
Fast Fourier Transform in Python, validated against analytical
solutions and NumPy's `np.fft`, and applied to a concrete
quantum-mechanics problem — the ground state of the quantum harmonic
oscillator.

This repository is the companion code and written report for the final
project of the course *Complementos Matemáticos y Numéricos* of the
M.Sc. in Physics (Radiation, Nanotechnology, Particles and
Astrophysics) at the University of Granada.

---

## Abstract

We study the discrete Fourier transform (DFT) and its efficient
$\mathcal{O}(N\log_2 N)$ implementation through the Fast Fourier
Transform (FFT) algorithm. Two independent implementations are
provided — a recursive Cooley–Tukey split and an in-place iterative
version based on bit-reversal permutation — and their accuracy,
theoretical properties and scaling are verified numerically.
The FFT is then applied to quantum mechanics: the kinetic operator
becomes diagonal in momentum space, which enables a spectral
evaluation of the Hamiltonian. As a concrete application we compute
the ground-state energy of the 1D quantum harmonic oscillator with
an FFT-based split-operator method in imaginary time and recover
$E_0 = 1/2$ to better than $10^{-10}$.

---

## Methods

The core package [`fft/`](fft) exposes two functions with the same
interface,

```python
from fft import fft_rec, fft_it

X = fft_rec(x, inverse=False)   # recursive Cooley–Tukey
X = fft_it (x, inverse=False)   # iterative / bit-reversal
```

Both expect a 1-D array whose length is a power of two and perform the
direct DFT for `inverse=False` and the inverse DFT for `inverse=True`.
Everything else in the repository imports these two functions so that
the algorithm lives in a single place.

Numerical validation is organised in two parallel trees:

- [`tests/`](tests) — `pytest` test suite that asserts $L^2$ error
  bounds against closed-form DFTs and against `numpy.fft`,
  covering random complex signals, constant / delta / plane-wave /
  pulse / sine–cosine inputs, and the DFT theoretical properties
  ($F^{-1}F=I$, $F^{4}=N^{2}I$, Parseval).
- [`examples/`](examples) — plain scripts that reproduce the figures
  embedded in this README (one script per signal family).

The quantum-mechanics application lives in
[`quantum/harmonic_oscillator.py`](quantum/harmonic_oscillator.py),
which propagates a trial wavefunction in imaginary time with a
second-order split-operator (Trotter) scheme and uses the FFT to apply
the kinetic kicks in momentum space.

---

## Results

### Particular cases

Every custom implementation tracks the analytical spectrum to
$L^2$ error $\lesssim 10^{-10}$, indistinguishable from NumPy's
`np.fft`.

| Sines and cosines | Gaussians |
|:--:|:--:|
| ![](figures/sine_cosine.png) | ![](figures/gaussians.png) |

| Square pulse | Plane wave |
|:--:|:--:|
| ![](figures/square_pulse.png) | ![](figures/plane_wave.png) |

| Kronecker delta $\delta_{n,0}$ | Kronecker delta $\delta_{n,50}$ |
|:--:|:--:|
| ![](figures/delta0.png) | ![](figures/delta50.png) |

| Constant signal |
|:--:|
| ![](figures/constant.png) |

### Theoretical properties

Averaging over 100 complex Gaussian random vectors, the identities
$F^{-1}F=I$, $F F^{-1}=I$, $F^{4}=N^{2}I$ and Parseval's norm
conservation all hold to $L^2$ error $\lesssim 10^{-11}$, with a mild
$\sqrt{N}$-like drift due to rounding.

![](figures/properties.png)

### Complexity

Normalising the measured execution time by $N\log_2 N$ produces
essentially flat curves for both the recursive and iterative
implementations, confirming the expected $\mathcal{O}(N\log_2 N)$
scaling. NumPy's FFTPACK is ~2 orders of magnitude faster in absolute
terms, but the scaling is the same.

![](figures/timings.png)

---

## Quantum harmonic oscillator via FFT

As a concrete quantum-mechanics example we compute the ground state
of the 1D quantum harmonic oscillator with the FFT. In natural units
($\hbar = m = \omega = 1$) the Hamiltonian is

$$
\hat H = -\frac{1}{2}\frac{d^2}{dx^2} + \frac{1}{2}x^{2},
\qquad
E_0 = \tfrac{1}{2},
\qquad
\psi_0(x) = \pi^{-1/4}\,e^{-x^{2}/2}.
$$

The kinetic operator is diagonal in momentum space, so it can be
evaluated efficiently with the FFT:

$$
\hat T \psi = \mathcal{F}^{-1}\!\left(\tfrac{1}{2}k^{2}\,\mathcal{F}\psi\right).
$$

Evolving in **imaginary time** $t \to -i\tau$ converts the unitary
propagator $e^{-i\hat H t}$ into an exponentially contracting filter
$e^{-\hat H \tau}$ that damps every excited state relative to the
ground state. A second-order Trotter split

$$
e^{-\hat H\,dt} \;\approx\;
e^{-\hat T\,dt/2}\,
e^{-\hat V\,dt}\,
e^{-\hat T\,dt/2}
$$

and a renormalisation at every step converge to $|\psi_0\rangle$ for
any initial state with non-zero overlap. The relevant lines of the
[`harmonic_oscillator.py`](harmonic_oscillator.py) script are:

```python
import numpy as np
from fft import fft_it

# Natural units, power-of-two grid for the FFT
N, L, dt, n_steps = 1024, 20.0, 0.01, 4000
dx = L / N
x  = np.linspace(-L / 2, L / 2 - dx, N)
k  = 2 * np.pi * np.fft.fftfreq(N, d=dx)

V         = 0.5 * x ** 2
expV      = np.exp(-V * dt)
expT_half = np.exp(-0.5 * k ** 2 * dt / 2)

# Initial (off-centre) trial wavefunction
psi = np.exp(-(x - 0.7) ** 2).astype(complex)
psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)

for _ in range(n_steps):
    # kinetic half-kick in momentum space
    psi = fft_it(fft_it(psi) * expT_half, inverse=True)
    # potential full-kick in position space
    psi *= expV
    # kinetic half-kick in momentum space
    psi = fft_it(fft_it(psi) * expT_half, inverse=True)
    # renormalise
    psi /= np.sqrt(np.sum(np.abs(psi) ** 2) * dx)

# Final energy estimator <psi|H|psi>
T_psi = fft_it(0.5 * k ** 2 * fft_it(psi), inverse=True)
E0    = np.real(np.sum(np.conj(psi) * T_psi) * dx) \
      + np.real(np.sum(np.conj(psi) * V * psi) * dx)

print(f"E_0 (FFT split-operator) = {E0:.10f}")
print(f"E_0 (exact)              = 0.5")
```

Running the full script yields

```
FFT split-operator ground-state energy : 0.50000000
Exact ground-state energy              : 0.5
Absolute error                         : 3.91e-11
```

and the probability density matches the analytical Gaussian ground
state to plotting accuracy:

![](figures/harmonic_oscillator.png)

---

## Reports

Fully worked-out reports (problem statement, derivations and
discussion of all the figures above) are provided in both languages:

| Language | File |
|----------|------|
| English  | [`MemoryEN.pdf`](MemoryEN.pdf) |
| Español  | [`MemoriaES.pdf`](MemoriaES.pdf) |

The LaTeX sources live in [`latex/`](latex).

---

## Repository structure

```
FFT-and-quantum-physics/
├── fft/                         # core package
│   ├── __init__.py              # re-exports fft_rec, fft_it
│   └── core.py                  # recursive & iterative Cooley–Tukey FFT
├── tests/                       # pytest test suite
│   ├── conftest.py              # puts the repo root on sys.path
│   ├── test_constant.py         # DFT of x[n] = 1
│   ├── test_deltas.py           # DFT of Kronecker deltas
│   ├── test_plane_wave.py       # DFT of a complex plane wave
│   ├── test_pulse.py            # DFT of a square pulse
│   ├── test_sine_cosine.py      # DFT of sines and cosines
│   ├── test_properties.py       # F^-1 F = I, F^4 = N^2 I, Parseval
│   └── test_vs_numpy.py         # agreement with numpy.fft on random signals
├── examples/                    # scripts that produce the figures
│   ├── plot_constant.py
│   ├── plot_deltas.py
│   ├── plot_gaussians.py
│   ├── plot_plane_wave.py
│   ├── plot_pulse.py
│   ├── plot_sine_cosine.py
│   ├── plot_properties.py
│   └── plot_timing.py
├── quantum/                     # physics application
│   └── harmonic_oscillator.py   # QHO ground state via FFT + imag. time
├── figures/                     # PNGs produced by the examples
├── latex/
│   ├── main_EN.tex
│   ├── main_ES.tex
│   ├── bibliography.bib
│   └── escudoUGRmonocromo.png
├── MemoryEN.pdf
└── MemoriaES.pdf
```

---

## Usage

Requires Python 3.10+, `numpy`, `matplotlib` and `pytest`.

```bash
# Run the unit test suite
pytest

# Reproduce the figures (each script writes its PNG to figures/)
python examples/plot_constant.py
python examples/plot_deltas.py
python examples/plot_gaussians.py
python examples/plot_pulse.py
python examples/plot_sine_cosine.py
python examples/plot_plane_wave.py
python examples/plot_properties.py     # slow: 100 runs × 13 sizes
python examples/plot_timing.py         # slow: up to N = 131 072

# Quantum harmonic oscillator
python quantum/harmonic_oscillator.py
```

---

## Author

**A. S. Amari Rabah**

Developed as part of the coursework for *Mathematical and Numerical Complements* —
Master's Degree in Physics: Radiation, Nanotechnology, Particles and Astrophysics,
University of Granada.


