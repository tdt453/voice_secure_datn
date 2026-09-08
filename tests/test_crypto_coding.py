"""
tests/test_crypto_coding.py
==========================================================
Crypto-Coding Unit Test
==========================================================
"""

import numpy as np

from config.config import POLAR_K, POLAR_N

from polar.crypto_coding import *

print("=" * 60)
print("CRYPTO CODING TEST")
print("=" * 60)

info = np.random.randint(
    0, 2, POLAR_K,
    dtype=np.uint8
)

key = np.random.randint(
    0, 2, POLAR_K,
    dtype=np.uint8
)

cipher = encrypt_information_bits(info, key)

recover = decrypt_information_bits(cipher, key)

assert np.array_equal(info, recover)

print("[PASS] XOR Encryption / Decryption")

u = build_crypto_u_vector(info, key)

assert len(u) == POLAR_N

cipher2, frozen = extract_crypto_bits(u)

assert np.array_equal(cipher, cipher2)
assert np.array_equal(frozen, key)

print("[PASS] Build u Vector")
print("[PASS] Extract Crypto Bits")

ok, ber = verify_frozen_bits(frozen, key)

assert ok
assert ber == 0

print("[PASS] Frozen Verification")

wrong_key = key.copy()
wrong_key[10] ^= 1

ok2, ber2 = verify_frozen_bits(frozen, wrong_key)

assert not ok2
assert ber2 > 0

print("[PASS] Wrong Key Detection")
print("Frozen BER =", ber2)

print("\nALL CRYPTO CODING TESTS PASSED")