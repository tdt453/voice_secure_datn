"""
tests/test_encoder.py
Unit tests for Polar Encoder
"""

import numpy as np

from config.config import POLAR_K, POLAR_N
from polar.encoder import (
    polar_transform,
    polar_transform_matrix,
    polar_encode,
    polar_encode_blocks,
)

print("=" * 55)
print("RUNNING POLAR ENCODER TEST")
print("=" * 55)


# -------------------------------------------------------
# Test 1 : N=8 Butterfly = Matrix
# -------------------------------------------------------

for _ in range(50):

    u = np.random.randint(0,2,8,dtype=np.uint8)

    x1 = polar_transform(u)
    x2 = polar_transform_matrix(u)

    assert np.array_equal(x1,x2)

print("[PASS] N=8 Butterfly Transform")


# -------------------------------------------------------
# Test 2 : N=16 Butterfly = Matrix
# -------------------------------------------------------

for _ in range(20):

    u = np.random.randint(0,2,16,dtype=np.uint8)

    x1 = polar_transform(u)
    x2 = polar_transform_matrix(u)

    assert np.array_equal(x1,x2)

print("[PASS] N=16 Butterfly Transform")


# -------------------------------------------------------
# Test 3 : Polar Encode
# -------------------------------------------------------

info = np.random.randint(0,2,POLAR_K,dtype=np.uint8)

codeword = polar_encode(info)

assert len(codeword)==POLAR_N

assert codeword.dtype==np.uint8

print("[PASS] Polar Encode (512 bits)")


# -------------------------------------------------------
# Test 4 : Multiple Blocks
# -------------------------------------------------------

blocks = np.random.randint(
    0,
    2,
    (10,POLAR_K),
    dtype=np.uint8
)

encoded = polar_encode_blocks(blocks)

assert encoded.shape==(10,POLAR_N)

print("[PASS] Multi-block Encoding")


# -------------------------------------------------------
# Test 5 : Binary Output
# -------------------------------------------------------

assert np.all(np.logical_or(codeword==0,codeword==1))

print("[PASS] Binary Codeword")


print("\nALL ENCODER TESTS PASSED")