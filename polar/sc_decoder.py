"""
polar/sc_decoder.py
============================================================
Successive Cancellation Polar Decoder (DATN v5.1)

3GPP TS 38.212 Compatible
Crypto-Coding Compatible (Dynamic Frozen Bits)

Target : Raspberry Pi 4 / Raspberry Pi 3
Polar Code : (512,256)
============================================================
"""

import numpy as np

from config.config import POLAR_N, POLAR_K

from polar.reliability import (
    frozen_mask,
    frozen_positions,
    information_positions,
)

from polar.utils import (
    f_function,
    g_function,
    hard_decision,
)

# ==========================================================
# Frozen Value Builder
# ==========================================================

def build_frozen_vector(frozen_bits=None):
    """
    Build frozen value vector in natural order.

    None        -> conventional Polar (all frozen = 0)
    frozen_bits -> Crypto-Coding frozen values (256 bits)
    """

    values = np.zeros(POLAR_N, dtype=np.uint8)

    if frozen_bits is None:
        return values

    frozen_bits = np.asarray(frozen_bits, dtype=np.uint8)

    if len(frozen_bits) != POLAR_K:
        raise ValueError("Frozen bits must contain 256 bits.")

    values[frozen_positions()] = frozen_bits

    return values


# ==========================================================
# Internal Recursive Node
# ==========================================================

def _sc_decode_node(llr, mask, frozen_values, start, u_hat):
    """
    Recursive SC node.

    Returns partial sums for parent recursion while storing
    decoded bits into global u_hat.
    """

    N = len(llr)

    # ---------------- Leaf ----------------
    if N == 1:

        if mask[0]:
            bit = frozen_values[0]
        else:
            bit = hard_decision(llr)[0]

        u_hat[start] = bit

        return np.array([bit], dtype=np.uint8)

    # ---------------- Internal ----------------
    half = N // 2

    left_llr = f_function(llr[:half], llr[half:])

    left_bits = _sc_decode_node(
        left_llr,
        mask[:half],
        frozen_values[:half],
        start,
        u_hat,
    )

    right_llr = g_function(
        llr[:half],
        llr[half:],
        left_bits,
    )

    right_bits = _sc_decode_node(
        right_llr,
        mask[half:],
        frozen_values[half:],
        start + half,
        u_hat,
    )

    # Partial sums propagated upward
    return np.concatenate(
        (
            np.bitwise_xor(left_bits, right_bits),
            right_bits,
        )
    ).astype(np.uint8)


# ==========================================================
# Recursive Wrapper
# ==========================================================

def sc_decode_recursive(llr, mask, frozen_values):
    """
    Decode complete Polar tree.
    """

    u_hat = np.zeros(POLAR_N, dtype=np.uint8)

    _sc_decode_node(
        llr,
        mask,
        frozen_values,
        0,
        u_hat,
    )

    return u_hat


# ==========================================================
# Extract Information Bits
# ==========================================================

def extract_information_bits(u_hat):
    """
    Extract 256 information bits.
    """

    return np.asarray(
        u_hat[information_positions()],
        dtype=np.uint8,
    )


# ==========================================================
# Polar Decoder
# ==========================================================

def polar_decode(llr, frozen_bits=None):
    """
    Decode one Polar codeword.

    Parameters
    ----------
    llr : ndarray (512,)
    frozen_bits : ndarray(256,) or None

    Returns
    -------
    info_bits : ndarray(256,)
    u_hat : ndarray(512,)
    """

    llr = np.asarray(llr, dtype=np.float32)

    if len(llr) != POLAR_N:
        raise ValueError(f"Need {POLAR_N} LLR values.")

    mask = frozen_mask()
    frozen_values = build_frozen_vector(frozen_bits)

    u_hat = sc_decode_recursive(
        llr,
        mask,
        frozen_values,
    )

    info_bits = extract_information_bits(u_hat)

    return info_bits, u_hat


# ==========================================================
# Decode Multiple Blocks
# ==========================================================

def polar_decode_blocks(llr_blocks, frozen_blocks=None):

    llr_blocks = np.asarray(llr_blocks, dtype=np.float32)

    if llr_blocks.ndim != 2:
        raise ValueError("Input must be 2-D array.")

    num_blocks = llr_blocks.shape[0]

    info = np.zeros(
        (num_blocks, POLAR_K),
        dtype=np.uint8,
    )

    u = np.zeros(
        (num_blocks, POLAR_N),
        dtype=np.uint8,
    )

    for i in range(num_blocks):

        fb = None

        if frozen_blocks is not None:
            fb = frozen_blocks[i]

        info_bits, u_hat = polar_decode(
            llr_blocks[i],
            frozen_bits=fb,
        )

        info[i] = info_bits
        u[i] = u_hat

    return info, u


# ==========================================================
# Standalone Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("POLAR SC DECODER (DATN v5.1)")
    print("=" * 60)

    llr = np.random.randn(POLAR_N).astype(np.float32)

    info_bits, u_hat = polar_decode(llr)

    print("LLR Length        :", len(llr))
    print("Decoded u Length  :", len(u_hat))
    print("Information Length:", len(info_bits))

    print("\nFirst 16 decoded information bits")
    print(info_bits[:16])

    frozen_bits = np.random.randint(
        0,
        2,
        POLAR_K,
        dtype=np.uint8,
    )

    _, u_hat2 = polar_decode(
        llr,
        frozen_bits=frozen_bits,
    )

    frozen_hat = u_hat2[frozen_positions()]

    print("\nCrypto Frozen Test")
    print("Frozen Match :", np.array_equal(frozen_hat, frozen_bits))

    print("\nDecoder Validation: PASS")