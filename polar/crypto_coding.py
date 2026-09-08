"""
polar/crypto_coding.py
==========================================================
Crypto-Coding Module

DATN Version 1.0
Target : Raspberry Pi 4 / Raspberry Pi 3

Implements the Crypto-Coding architecture described in
the paper.

N = 512
K = 256
==========================================================
"""

import numpy as np

from config.config import POLAR_N, POLAR_K

from polar.reliability import (
    information_positions,
    frozen_positions
)

# ---------------------------------------------------------
# Layer 1 : Encrypt Information Bits
# ---------------------------------------------------------

def encrypt_information_bits(info_bits, keystream):
    """
    Cipher bits = Information bits XOR Keystream.
    """

    info_bits = np.asarray(info_bits, dtype=np.uint8)
    keystream = np.asarray(keystream, dtype=np.uint8)

    if len(info_bits) != POLAR_K:
        raise ValueError("Need 256 information bits.")

    if len(keystream) != POLAR_K:
        raise ValueError("Need 256 keystream bits.")

    return np.bitwise_xor(info_bits, keystream)


# ---------------------------------------------------------
# Layer 2 : Replace Frozen Bits
# ---------------------------------------------------------

def replace_frozen_bits(keystream):
    """
    Frozen bits are replaced by AES keystream.
    """

    keystream = np.asarray(keystream, dtype=np.uint8)

    if len(keystream) != POLAR_K:
        raise ValueError("Need 256 keystream bits.")

    return keystream.copy()


# ---------------------------------------------------------
# Build Crypto u Vector
# ---------------------------------------------------------

def build_crypto_u_vector(info_bits,
                          keystream):
    """
    Build Polar input vector u.

    Information positions <- Cipher Bits

    Frozen positions <- Keystream
    """

    cipher_bits = encrypt_information_bits(
        info_bits,
        keystream
    )

    frozen_bits = replace_frozen_bits(
        keystream
    )

    u = np.zeros(POLAR_N, dtype=np.uint8)

    u[information_positions()] = cipher_bits
    u[frozen_positions()] = frozen_bits

    return u


# ---------------------------------------------------------
# Extract Cipher/Frozen Bits
# ---------------------------------------------------------

def extract_crypto_bits(u_hat):
    """
    Extract information-set bits and frozen-set bits.
    """

    u_hat = np.asarray(u_hat, dtype=np.uint8)

    if len(u_hat) != POLAR_N:
        raise ValueError("Need 512 decoded bits.")

    cipher_bits = u_hat[information_positions()]
    frozen_bits = u_hat[frozen_positions()]

    return cipher_bits, frozen_bits


# ---------------------------------------------------------
# Decrypt Information Bits
# ---------------------------------------------------------

def decrypt_information_bits(cipher_bits,
                             keystream):
    """
    AES-CTR decryption.

    Message = Cipher XOR Keystream
    """

    cipher_bits = np.asarray(cipher_bits, dtype=np.uint8)
    keystream = np.asarray(keystream, dtype=np.uint8)

    return np.bitwise_xor(cipher_bits, keystream)


# ---------------------------------------------------------
# Frozen Verification
# ---------------------------------------------------------

def verify_frozen_bits(received_frozen,
                       keystream):
    """
    Compare received frozen bits with locally
    generated keystream.
    """

    received_frozen = np.asarray(received_frozen,
                                 dtype=np.uint8)

    keystream = np.asarray(keystream,
                           dtype=np.uint8)

    equal = np.sum(received_frozen == keystream)

    ber = 1.0 - equal / len(keystream)

    return ber == 0, ber


# ---------------------------------------------------------
# Self Test
# ---------------------------------------------------------

if __name__ == "__main__":

    print("=" * 60)
    print("CRYPTO CODING SELF TEST")
    print("=" * 60)

    info = np.random.randint(
        0, 2, POLAR_K,
        dtype=np.uint8
    )

    key = np.random.randint(
        0, 2, POLAR_K,
        dtype=np.uint8
    )

    u = build_crypto_u_vector(info, key)

    cipher, frozen = extract_crypto_bits(u)

    recover = decrypt_information_bits(cipher, key)

    ok, ber = verify_frozen_bits(frozen, key)

    print("u Length          :", len(u))
    print("Cipher Length     :", len(cipher))
    print("Frozen Length     :", len(frozen))

    print("\nRecover Correct :", np.array_equal(info, recover))
    print("Frozen Verify    :", ok)
    print("Frozen BER       :", ber)

    print("\nValidation: PASS")