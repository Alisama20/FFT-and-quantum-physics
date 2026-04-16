"""
Core FFT implementations (Cooley–Tukey radix-2).

Two reference implementations are provided:
  * fft_rec — recursive (decimation-in-time)
  * fft_it  — iterative with bit-reversal permutation

Both require the input length N to be a power of two. Setting
``inverse=True`` computes the inverse DFT with the standard 1/N scaling.
"""

import numpy as np


# ==========================================
# Recursive implementation
# ==========================================

def fft_rec(x, inverse=False):
    """Recursive Cooley–Tukey FFT of a 1-D array of length N = 2^m."""
    x = np.asarray(x, dtype=complex)
    N = x.shape[0]

    if N == 1:
        return x.copy()
    if N % 2 != 0:
        raise ValueError("N must be a power of 2")

    sign = 1 if inverse else -1

    X_even = fft_rec(x[::2], inverse)
    X_odd = fft_rec(x[1::2], inverse)

    X = np.zeros(N, dtype=complex)
    W1 = np.exp(sign * 2j * np.pi / N)   # W_N
    W = 1.0 + 0j

    for k in range(N // 2):
        t = W * X_odd[k]
        X[k] = X_even[k] + t
        X[k + N // 2] = X_even[k] - t    # periodicity
        W *= W1

    if inverse:
        X /= 2
    return X


# ==========================================
# Iterative implementation (bit-reversal)
# ==========================================

def fft_it(x, inverse=False):
    """Iterative Cooley–Tukey FFT of a 1-D array of length N = 2^m."""
    x = np.asarray(x, dtype=complex).copy()
    N = x.shape[0]
    if (N & (N - 1)) != 0:
        raise ValueError("N must be a power of 2")

    nbits = int(np.log2(N))

    def bit_reverse(n, bits):
        r = 0
        for _ in range(bits):
            r = (r << 1) | (n & 1)       # append LSB of n to r
            n >>= 1
        return r

    # 1) bit-reversal permutation
    for i in range(N):
        j = bit_reverse(i, nbits)
        if j > i:
            x[i], x[j] = x[j], x[i]

    # 2) log2(N) butterfly stages
    sign = 1 if inverse else -1
    m = 2
    while m <= N:
        half = m // 2
        Wm = np.exp(sign * 2j * np.pi / m)
        for k in range(0, N, m):
            W = 1.0 + 0j
            for j in range(half):
                t = W * x[k + j + half]
                u = x[k + j]
                x[k + j] = u + t
                x[k + j + half] = u - t
                W *= Wm
        m *= 2

    if inverse:
        x /= N
    return x


# ==========================================
# Sanity check
# ==========================================

if __name__ == "__main__":
    x = np.array([1.0, 1.0, 1.0, 1.0])
    X = fft_it(x)
    print("x =", x)
    print("X = fft(x) =", X)
    print("ifft(X) =", fft_it(X, inverse=True))
