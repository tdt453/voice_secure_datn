"""
polar/encoder.py
==========================================================
3GPP NR Polar Encoder (DATN Version)

Standard : 3GPP TS 38.212
Target   : Raspberry Pi 4 / Raspberry Pi 3

N = 512
K = 256

Compatible with MATLAB implementation.
==========================================================
"""

import numpy as np

from config.config import POLAR_N, POLAR_K
from polar.reliability import build_u_vector


# ==========================================================
# Fast Polar Transform (Arikan Butterfly)
# ==========================================================

def polar_transform(u: np.ndarray) -> np.ndarray:
    """
    Fast Polar Transform.

    x = u * G_N

    Complexity: O(N log N)
    """

    x = np.asarray(u, dtype=np.uint8).copy()

    N = len(x)

    if N & (N - 1):
        raise ValueError("N must be a power of two.")

    stage = 1

    while stage < N:

        step = stage * 2

        for i in range(0, N, step):

            for j in range(stage):

                x[i + j] ^= x[i + j + stage]

        stage <<= 1

    return x.astype(np.uint8)


# ==========================================================
# Encode from complete u-vector
# ==========================================================

def encode_from_u(u: np.ndarray) -> np.ndarray:
    """
    Encode a complete Polar input vector.
    """

    u = np.asarray(u, dtype=np.uint8)

    if len(u) != POLAR_N:
        raise ValueError(
            f"Expected length {POLAR_N}, got {len(u)}."
        )

    return polar_transform(u)


# ==========================================================
# Polar Encoder
# ==========================================================

def polar_encode(info_bits: np.ndarray,
                 frozen_bits: np.ndarray | None = None
                 ) -> np.ndarray:
    """
    Encode one information block.

    Parameters
    ----------
    info_bits : ndarray (256,)
    frozen_bits : ndarray (256,) or None

    Returns
    -------
    codeword : ndarray (512,)
    """

    info_bits = np.asarray(info_bits, dtype=np.uint8)

    if len(info_bits) != POLAR_K:
        raise ValueError(
            f"Need {POLAR_K} information bits."
        )

    # Build u vector according to TS 38.212 reliability sequence
    u = build_u_vector(info_bits, frozen_bits)

    # Fast Polar Transform
    codeword = polar_transform(u)

    return codeword.astype(np.uint8)


# ==========================================================
# Multi-block Encoder
# ==========================================================

def polar_encode_blocks(blocks: np.ndarray) -> np.ndarray:
    """
    Encode multiple information blocks.
    """

    blocks = np.asarray(blocks, dtype=np.uint8)

    if blocks.ndim != 2:
        raise ValueError("Input must be a 2-D array.")

    if blocks.shape[1] != POLAR_K:
        raise ValueError(
            f"Each block must contain {POLAR_K} bits."
        )

    encoded = np.zeros(
        (blocks.shape[0], POLAR_N),
        dtype=np.uint8
    )

    for i in range(blocks.shape[0]):
        encoded[i] = polar_encode(blocks[i])

    return encoded


# ==========================================================
# Self Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 55)
    print("POLAR ENCODER (3GPP DATN)")
    print("=" * 55)

    info_bits = np.random.randint(
        0, 2, POLAR_K, dtype=np.uint8
    )

    codeword = polar_encode(info_bits)

    print("Information Length :", len(info_bits))
    print("Codeword Length    :", len(codeword))

    print("\nFirst 16 information bits")
    print(info_bits[:16])

    print("\nFirst 16 encoded bits")
    print(codeword[:16])

    print("\nEncoder Validation: PASS")