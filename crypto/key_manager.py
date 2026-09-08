"""
crypto/key_manager.py
==========================================================
Crypto Key Manager (DATN Version 2.0)

Target : Raspberry Pi 4 / Raspberry Pi 3

- Fixed AES-256 Master Key.
- Fixed 128-bit IV.
- AES-CTR Counter generator.
- Hex conversion utilities.

The Master Key will later be replaced by the
Channel Fingerprint module (LAB 14).
==========================================================
"""

from __future__ import annotations

# ==========================================================
# DATN DEFAULT MASTER KEY (256 bits)
# ==========================================================

MASTER_KEY_HEX = (
    "00112233445566778899AABBCCDDEEFF"
    "102132435465768798A9BACBDCEDFE0F"
)

IV_HEX = "0123456789ABCDEFFEDCBA9876543210"

BLOCKS_PER_FRAME = 20

# ==========================================================
# Key / IV
# ==========================================================

def get_master_key() -> bytes:
    """
    Return AES-256 Master Key (32 bytes).
    """
    key = bytes.fromhex(MASTER_KEY_HEX)

    if len(key) != 32:
        raise RuntimeError("Master Key must be 32 bytes.")

    return key


def get_iv() -> bytes:
    """
    Return 128-bit IV (16 bytes).
    """
    iv = bytes.fromhex(IV_HEX)

    if len(iv) != 16:
        raise RuntimeError("IV must be 16 bytes.")

    return iv


# ==========================================================
# Counter
# ==========================================================

def block_counter(frame_id: int,
                  block_id: int) -> int:
    """
    AES-CTR counter for one Polar block.

    Counter increases continuously across frames.
    """

    if block_id < 0 or block_id >= BLOCKS_PER_FRAME:
        raise ValueError("block_id must be between 0 and 19.")

    if frame_id < 0:
        raise ValueError("frame_id must be >= 0.")

    return frame_id * BLOCKS_PER_FRAME + block_id


# ==========================================================
# Conversion Utilities
# ==========================================================

def key_to_hex(key: bytes) -> str:
    return key.hex().upper()


def iv_to_hex(iv: bytes) -> str:
    return iv.hex().upper()


def hex_to_key(text: str) -> bytes:
    key = bytes.fromhex(text)

    if len(key) != 32:
        raise ValueError("Invalid AES-256 key.")

    return key


def hex_to_iv(text: str) -> bytes:
    iv = bytes.fromhex(text)

    if len(iv) != 16:
        raise ValueError("Invalid IV.")

    return iv


# ==========================================================
# Generate counters for one audio frame
# ==========================================================

def frame_counters(frame_id: int):
    """
    Return list of 20 counters for one frame.
    """

    return [
        block_counter(frame_id, b)
        for b in range(BLOCKS_PER_FRAME)
    ]


# ==========================================================
# Self Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("CRYPTO KEY MANAGER TEST")
    print("=" * 60)

    key = get_master_key()
    iv = get_iv()

    print("Master Key Length :", len(key), "bytes")
    print("IV Length         :", len(iv), "bytes")

    print("\nMaster Key")
    print(key_to_hex(key))

    print("\nIV")
    print(iv_to_hex(iv))

    print("\nFrame 0 Counters")
    print(frame_counters(0))

    print("\nFrame 3 Counters")
    print(frame_counters(3))

    print("\nValidation: PASS")