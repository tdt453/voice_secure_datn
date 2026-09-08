"""
tests/test_encoder_identity.py
==========================================================
Unit Test for Polar Encoder (3GPP TS 38.212)

Target : Raspberry Pi 4
N = 512
K = 256
==========================================================
"""

import numpy as np

from config.config import POLAR_N, POLAR_K
from polar.encoder import polar_encode, polar_encode_blocks

print("=" * 60)
print("POLAR ENCODER IDENTITY TEST")
print("=" * 60)

# ==========================================================
# Test 1 : Single Block
# ==========================================================

info_bits = np.random.randint(
    0, 2, POLAR_K, dtype=np.uint8
)

codeword = polar_encode(info_bits)

assert codeword.shape == (POLAR_N,), \
    "Codeword length is incorrect."

assert codeword.dtype == np.uint8, \
    "Codeword dtype must be uint8."

assert np.all((codeword == 0) | (codeword == 1)), \
    "Codeword must contain only 0/1."

print("[PASS] Single-block encoding")
print(f"Codeword length : {len(codeword)}")

# ==========================================================
# Test 2 : Multiple Blocks
# ==========================================================

num_blocks = 20

blocks = np.random.randint(
    0, 2,
    (num_blocks, POLAR_K),
    dtype=np.uint8
)

encoded = polar_encode_blocks(blocks)

assert encoded.shape == (num_blocks, POLAR_N), \
    "Multi-block shape mismatch."

assert np.all((encoded == 0) | (encoded == 1)), \
    "Multi-block output is not binary."

print(f"[PASS] Multi-block encoding ({num_blocks} blocks)")

# ==========================================================
# Test 3 : Random Stress Test
# ==========================================================

for _ in range(100):

    bits = np.random.randint(
        0, 2, POLAR_K, dtype=np.uint8
    )

    cw = polar_encode(bits)

    assert len(cw) == POLAR_N
    assert np.all((cw == 0) | (cw == 1))

print("[PASS] Random stress test (100 blocks)")

print("\nALL ENCODER TESTS PASSED")