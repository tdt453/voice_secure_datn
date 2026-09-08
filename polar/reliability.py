"""
polar/reliability.py
==========================================================
3GPP TS 38.212 Polar Reliability Sequence

Version : Polar Package v1.0
Target  : Raspberry Pi 4 / Raspberry Pi 3
Standard: 3GPP TS 38.212 (Section 5.3.1.2)

Polar Parameters
----------------
N = 512
K = 256
Rate = 1/2

This module provides:
    • Reliability sequence Q_N.
    • Information positions.
    • Frozen positions.
    • Frozen / Information masks.
    • Build u-vector for Polar encoder.
    • Validation utilities.
==========================================================
"""

import numpy as np

from config.config import POLAR_N, POLAR_K
from polar.reliability_table import Q1024


# ==========================================================
# Reliability Sequence
# ==========================================================

def reliability_sequence() -> np.ndarray:
    """
    Return reliability sequence Q_N.

    Theo TS 38.212:
    Q_N được tạo bằng cách lấy tất cả Qi < N
    từ Polar sequence Q1024.
    """

    seq = Q1024[Q1024 < POLAR_N]

    if len(seq) != POLAR_N:
        raise RuntimeError(
            f"Expected {POLAR_N} reliability entries, "
            f"but got {len(seq)}."
        )

    return seq.astype(np.int16)


# ==========================================================
# Information / Frozen Sets
# ==========================================================

def information_positions() -> np.ndarray:
    """
    Information bit positions.

    Lấy K bit-channel đáng tin cậy nhất.
    """

    seq = reliability_sequence()

    info = seq[-POLAR_K:]

    return np.sort(info.astype(np.int16))


def frozen_positions() -> np.ndarray:
    """
    Frozen bit positions.
    """

    seq = reliability_sequence()

    frozen = seq[:-POLAR_K]

    return np.sort(frozen.astype(np.int16))


# ==========================================================
# Masks
# ==========================================================

def frozen_mask() -> np.ndarray:
    """
    Frozen mask.

    1 -> Frozen Bit
    0 -> Information Bit
    """

    mask = np.ones(POLAR_N, dtype=np.uint8)

    mask[information_positions()] = 0

    return mask


def info_mask() -> np.ndarray:
    """
    Information mask.

    1 -> Information Bit
    0 -> Frozen Bit
    """

    return 1 - frozen_mask()


# ==========================================================
# Build u Vector
# ==========================================================

def build_u_vector(
    info_bits: np.ndarray,
    frozen_bits: np.ndarray | None = None,
) -> np.ndarray:
    """
    Build Polar input vector u.

    Parameters
    ----------
    info_bits : ndarray
        256 information bits.

    frozen_bits : ndarray or None
        256 frozen bits.
        Nếu None -> Frozen bits = 0.

    Returns
    -------
    ndarray
        Vector u độ dài 512.
    """

    info_bits = np.asarray(info_bits, dtype=np.uint8)

    if info_bits.size != POLAR_K:
        raise ValueError(
            f"info_bits must contain {POLAR_K} bits."
        )

    if frozen_bits is None:
        frozen_bits = np.zeros(
            POLAR_N - POLAR_K,
            dtype=np.uint8
        )
    else:
        frozen_bits = np.asarray(
            frozen_bits,
            dtype=np.uint8
        )

    if frozen_bits.size != POLAR_N - POLAR_K:
        raise ValueError(
            f"frozen_bits must contain {POLAR_N-POLAR_K} bits."
        )

    u = np.zeros(POLAR_N, dtype=np.uint8)

    u[information_positions()] = info_bits
    u[frozen_positions()] = frozen_bits

    return u


# ==========================================================
# Validation
# ==========================================================

def validate_q1024() -> bool:
    """
    Validate Q1024 table.
    """

    assert len(Q1024) == 1024, \
        f"Q1024 length = {len(Q1024)}"

    assert len(np.unique(Q1024)) == 1024, \
        "Duplicate values found."

    assert Q1024.min() == 0
    assert Q1024.max() == 1023

    return True


def validate_sequence() -> bool:
    """
    Validate reliability sequence for N.
    """

    validate_q1024()

    seq = reliability_sequence()

    assert len(seq) == POLAR_N
    assert len(np.unique(seq)) == POLAR_N

    assert seq.min() >= 0
    assert seq.max() < POLAR_N

    info = information_positions()
    frozen = frozen_positions()

    assert len(info) == POLAR_K
    assert len(frozen) == POLAR_N - POLAR_K

    assert np.intersect1d(info, frozen).size == 0

    mask = frozen_mask()

    assert np.sum(mask == 0) == POLAR_K
    assert np.sum(mask == 1) == POLAR_N - POLAR_K

    return True


# ==========================================================
# Standalone Test
# ==========================================================

if __name__ == "__main__":

    validate_sequence()

    seq = reliability_sequence()
    info = information_positions()
    frozen = frozen_positions()
    mask = frozen_mask()

    print("=" * 50)
    print("        POLAR RELIABILITY (3GPP NR)")
    print("=" * 50)

    print(f"Q1024 Length      : {len(Q1024)}")
    print(f"Sequence Length   : {len(seq)}")
    print(f"Information Bits  : {len(info)}")
    print(f"Frozen Bits       : {len(frozen)}")

    print("\nFirst 20 Reliability Entries")
    print(seq[:20])

    print("\nFirst 20 Information Positions")
    print(info[:20])

    print("\nFirst 20 Frozen Positions")
    print(frozen[:20])

    print("\nFrozen Mask Summary")
    print("Mask == 0 :", np.sum(mask == 0))
    print("Mask == 1 :", np.sum(mask == 1))

    print("\nValidation: PASS")