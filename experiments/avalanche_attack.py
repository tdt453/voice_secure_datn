"""
experiments/avalanche_attack.py

Avalanche Effect Test
DATN - Raspberry Pi
"""

import csv
import hashlib
from pathlib import Path

import numpy as np
import soundfile as sf

from crypto.key_manager import get_master_key, get_iv
from crypto.keystream import generate_keystream
from crypto.key_manager import block_counter
from polar.crypto_coding import build_crypto_u_vector, extract_crypto_bits
from audio.audio_blocks import split_audio_blocks

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

AUDIO = "data/voice.wav"
POLAR_K = 256


# ======================================================
# WAV -> Bitstream
# ======================================================
pcm, _ = sf.read(AUDIO, dtype="int16")
bits = np.unpackbits(
    np.frombuffer(pcm.tobytes(), dtype=np.uint8)
).astype(np.uint8)

blocks, pad_bits = split_audio_blocks(bits)

master_key = get_master_key()
iv = get_iv()


# ======================================================
# Encrypt using supplied key
# ======================================================
def encrypt_with_key(key_bytes):

    cipher_bits_all = []

    for blk_id, info_bits in enumerate(blocks):

        frame_id = blk_id // 20
        block_id = blk_id % 20

        counter = block_counter(frame_id, block_id)

        ks = generate_keystream(
            key_bytes,
            iv,
            counter
        )

        u = build_crypto_u_vector(
            info_bits,
            ks
        )

        cipher_bits, _ = extract_crypto_bits(u)

        cipher_bits_all.extend(cipher_bits.tolist())

    return np.asarray(cipher_bits_all, dtype=np.uint8)


# ======================================================
# Original ciphertext
# ======================================================
cipher_ref = encrypt_with_key(master_key)


# ======================================================
# Flip each key bit
# ======================================================
rows = []

for bit in range(256):

    key2 = bytearray(master_key)

    byte = bit // 8
    b = bit % 8

    key2[byte] ^= (1 << b)

    cipher_test = encrypt_with_key(bytes(key2))

    diff = np.sum(cipher_ref != cipher_test)

    avalanche = diff / len(cipher_ref) * 100

    rows.append([bit, diff, avalanche])

rows = np.asarray(rows)

mean_avalanche = rows[:,2].mean()
std_avalanche = rows[:,2].std()
min_avalanche = rows[:,2].min()
max_avalanche = rows[:,2].max()

with open(
    RESULT_DIR / "avalanche_results.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow([
        "KeyBit",
        "ChangedBits",
        "AvalanchePercent"
    ])

    writer.writerows(rows.tolist())

print("="*60)
print("AVALANCHE EFFECT TEST")
print("="*60)
print(f"Total Cipher Bits : {len(cipher_ref)}")
print(f"Mean Avalanche    : {mean_avalanche:.4f}%")
print(f"Std               : {std_avalanche:.4f}%")
print(f"Min               : {min_avalanche:.4f}%")
print(f"Max               : {max_avalanche:.4f}%")
print("Saved -> results/avalanche_results.csv")