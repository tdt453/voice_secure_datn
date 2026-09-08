"""
polar/utils.py
==========================================================
Polar Utilities for 3GPP TS 38.212

Version : Polar Package v1.0
Target  : Raspberry Pi 4 / Raspberry Pi 3

Functions
---------
1. Bit-reversal permutation.
2. BPSK modulation.
3. BPSK hard demodulation.
4. AWGN LLR computation.
5. SC decoder f() function.
6. SC decoder g() function.
7. Hard decision from LLR.
==========================================================
"""

import numpy as np


# ==========================================================
# Bit-Reversal
# ==========================================================

def bit_reverse_indices(N: int) -> np.ndarray:
    """
    Generate bit-reversal permutation indices.

    Example
    -------
    N = 8

    Original indices
    0 1 2 3 4 5 6 7

    Bit reversed
    0 4 2 6 1 5 3 7
    """

    n_bits = int(np.log2(N))

    indices = np.zeros(N, dtype=np.int32)

    for i in range(N):
        binary = format(i, f"0{n_bits}b")
        reversed_binary = binary[::-1]
        indices[i] = int(reversed_binary, 2)

    return indices


def bit_reverse(x: np.ndarray) -> np.ndarray:
    """
    Apply bit-reversal permutation.
    """

    idx = bit_reverse_indices(len(x))
    return x[idx]


# ==========================================================
# BPSK Modulation
# ==========================================================

def bpsk_modulate(bits: np.ndarray) -> np.ndarray:
    """
    BPSK mapping

    Bit 0 -> +1
    Bit 1 -> -1
    """

    bits = np.asarray(bits, dtype=np.uint8)

    return 1.0 - 2.0 * bits.astype(np.float32)


def bpsk_demodulate(symbols: np.ndarray) -> np.ndarray:
    """
    Hard decision BPSK.

    symbol >= 0 -> 0
    symbol < 0  -> 1
    """

    symbols = np.asarray(symbols)

    return (symbols < 0).astype(np.uint8)


# ==========================================================
# LLR Computation
# ==========================================================

def llr_awgn(received: np.ndarray, sigma2: float) -> np.ndarray:
    """
    Compute Log-Likelihood Ratio for AWGN.

    LLR = 2*y / sigma²
    """

    received = np.asarray(received, dtype=np.float32)

    return (2.0 * received) / sigma2


# ==========================================================
# SC Decoder Functions
# ==========================================================

def f_function(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    Min-Sum approximation.

    f(a,b) = sign(a) sign(b) min(|a|,|b|)
    """

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)

    return np.sign(a) * np.sign(b) * np.minimum(np.abs(a), np.abs(b))


def g_function(a: np.ndarray,
               b: np.ndarray,
               u: np.ndarray) -> np.ndarray:
    """
    g(a,b,u)

    u must contain decoded bits (0/1).
    """

    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    u = np.asarray(u, dtype=np.uint8)

    return b + (1.0 - 2.0 * u.astype(np.float32)) * a


# ==========================================================
# Hard Decision
# ==========================================================

def hard_decision(llr: np.ndarray) -> np.ndarray:
    """
    LLR >= 0 -> bit 0
    LLR < 0  -> bit 1
    """

    llr = np.asarray(llr)

    return (llr < 0).astype(np.uint8)


# ==========================================================
# AWGN Noise Generator
# ==========================================================

def add_awgn(signal: np.ndarray,
             ebno_db: float) -> tuple[np.ndarray, float]:
    """
    Add AWGN noise.

    Returns
    -------
    noisy_signal
    sigma²
    """

    signal = np.asarray(signal, dtype=np.float32)

    ebno = 10 ** (ebno_db / 10)

    sigma2 = 1.0 / (2 * ebno)

    noise = np.sqrt(sigma2) * np.random.randn(*signal.shape)

    noisy = signal + noise.astype(np.float32)

    return noisy, sigma2


# ==========================================================
# Unit Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 50)
    print("POLAR UTILS TEST")
    print("=" * 50)

    idx = bit_reverse_indices(8)
    print("Bit-Reversal N=8")
    print(idx)

    bits = np.array([0,1,1,0,0,1], dtype=np.uint8)
    symbols = bpsk_modulate(bits)
    bits_hat = bpsk_demodulate(symbols)

    print("\nBPSK Test")
    print("Bits    :", bits)
    print("Symbols :", symbols)
    print("Recover :", bits_hat)

    noisy, sigma2 = add_awgn(symbols, ebno_db=5)

    llr = llr_awgn(noisy, sigma2)

    print("\nLLR")
    print(llr)

    print("\nHard Decision")
    print(hard_decision(llr))

    a = np.array([2,-3,1,-4], dtype=np.float32)
    b = np.array([1,5,-2,-1], dtype=np.float32)
    u = np.array([0,1,0,1], dtype=np.uint8)

    print("\nf()")
    print(f_function(a,b))

    print("\ng()")
    print(g_function(a,b,u))

    print("\nUTILS VALIDATION: PASS")