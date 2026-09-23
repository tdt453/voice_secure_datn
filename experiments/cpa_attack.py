import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

from audio.audio_blocks import split_audio_blocks
from crypto.key_manager import get_master_key, get_iv, block_counter
from crypto.keystream import generate_keystream
from polar.crypto_coding import build_crypto_u_vector
from polar.encoder import polar_encode

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

# ======================================================
# WAV -> Bitstream
# ======================================================
pcm, _ = sf.read("data/voice.wav", dtype="int16")
bits = np.unpackbits(
    np.frombuffer(pcm.tobytes(), dtype=np.uint8)
).astype(np.uint8)

master_key = get_master_key()
iv = get_iv()


from polar.crypto_coding import (
    build_crypto_u_vector,
    extract_crypto_bits,
)

from polar.encoder import polar_encode


def encrypt_codeword(audio_bits):

    blocks, _ = split_audio_blocks(audio_bits)

    codeword_bits = []

    for blk_id, info_bits in enumerate(blocks):

        frame_id = blk_id // 20
        block_id = blk_id % 20

        ks = generate_keystream(
            master_key,
            iv,
            block_counter(frame_id, block_id)
        )

        # Tao u-vector
        u = build_crypto_u_vector(info_bits, ks)

        # Lay 256 cipher bits
        cipher_bits, _ = extract_crypto_bits(u)

        # Polar encoder nhan 256 bits
        codeword = polar_encode(
            cipher_bits,
            frozen_bits=ks
        )

        codeword_bits.extend(codeword.tolist())

    return np.asarray(codeword_bits, dtype=np.uint8)

# ======================================================
# Original ciphertext
# ======================================================
cipher1 = encrypt_codeword(bits)

# ======================================================
# Flip ONE plaintext bit
# ======================================================
bits2 = bits.copy()
bits2[1000] ^= 1

cipher2 = encrypt_codeword(bits2)

# ======================================================
# Metrics
# ======================================================
npcr = np.mean(cipher1 != cipher2) * 100

uaci = (
    np.mean(np.abs(cipher1.astype(np.int16) - cipher2.astype(np.int16)))
    * 100
)

corr = np.corrcoef(cipher1, cipher2)[0, 1]

with open(RESULT_DIR / "cpa_metrics.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["NPCR_percent", npcr])
    writer.writerow(["UACI_percent", uaci])
    writer.writerow(["Correlation", corr])

idx = np.random.choice(len(cipher1), 6000, replace=False)

plt.figure(figsize=(6, 6))
plt.scatter(cipher1[idx], cipher2[idx], s=2, alpha=0.25)
plt.xlabel("Ciphertext A")
plt.ylabel("Ciphertext B")
plt.tight_layout()
plt.savefig(RESULT_DIR / "cpa_scatter.png", dpi=600)
plt.close()

print("=" * 60)
print("CPA ATTACK (POLAR CODEWORD)")
print("=" * 60)
print(f"Cipher bits  : {len(cipher1)}")
print(f"NPCR         : {npcr:.6f}%")
print(f"UACI         : {uaci:.6f}%")
print(f"Correlation  : {corr:.6f}")
print("Saved -> results/cpa_metrics.csv")