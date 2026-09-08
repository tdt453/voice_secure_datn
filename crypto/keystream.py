"""
crypto/keystream.py
==========================================================
AES-CTR Keystream Generator

DATN Version 1.0
Target : Raspberry Pi 4 / Raspberry Pi 3

Generate 256-bit keystream for Crypto-Coding.
==========================================================
"""

import numpy as np

from Crypto.Cipher import AES
from Crypto.Util import Counter


# ==========================================================
# AES CTR Keystream Generator
# ==========================================================

def generate_keystream(master_key: bytes,
                       iv: bytes,
                       counter: int = 0,
                       n_bits: int = 256) -> np.ndarray:
    """
    Generate AES-CTR keystream.

    Parameters
    ----------
    master_key : bytes (32 bytes)

    iv : bytes (16 bytes)

    counter : block counter

    n_bits : output length

    Returns
    -------
    ndarray (uint8)
        Binary keystream.
    """

    if len(master_key) != 32:
        raise ValueError("AES-256 key must contain 32 bytes.")

    if len(iv) != 16:
        raise ValueError("IV must contain 16 bytes.")

    n_bytes = (n_bits + 7) // 8

    init = int.from_bytes(iv, "big") + counter

    ctr = Counter.new(
        128,
        initial_value=init
    )

    cipher = AES.new(
        master_key,
        AES.MODE_CTR,
        counter=ctr
    )

    stream = cipher.encrypt(
        b"\x00" * n_bytes
    )

    bits = np.unpackbits(
        np.frombuffer(stream, dtype=np.uint8)
    )

    return bits[:n_bits].astype(np.uint8)


# ==========================================================
# Hex helper
# ==========================================================

def keystream_hex(bits: np.ndarray) -> str:

    packed = np.packbits(bits)

    return packed.tobytes().hex()


# ==========================================================
# Self Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AES KEYSTREAM TEST")
    print("=" * 60)

    key = bytes(range(32))
    iv = bytes(range(16))

    ks = generate_keystream(
        key,
        iv,
        counter=0
    )

    print("Length :", len(ks))
    print("First 32 bits")
    print(ks[:32])

    print("\nHEX (first 16 bytes)")
    print(keystream_hex(ks)[:32])

    print("\nValidation: PASS")