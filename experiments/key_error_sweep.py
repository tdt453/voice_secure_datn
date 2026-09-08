"""
experiments/key_error_sweep.py
============================================================
Key Error Sweep Experiment (DATN v2.0)

Crypto-Coding Security Evaluation

Outputs
-------
results/csv/key_error_sweep.csv
results/figures/key_error_sweep.png
results/logs/key_error_sweep.txt
============================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from config.config import POLAR_K

from crypto.key_manager import (
    get_master_key,
    get_iv,
    block_counter,
)

from crypto.keystream import generate_keystream

from polar.crypto_coding import (
    build_crypto_u_vector,
    extract_crypto_bits,
    decrypt_information_bits,
    verify_frozen_bits,
)

from polar.encoder import polar_encode
from polar.sc_decoder import polar_decode

from polar.utils import (
    bpsk_modulate,
    add_awgn,
    llr_awgn,
)

# ==========================================================
# Configuration
# ==========================================================

NUM_BLOCKS = 500
FRAME_ID = 0
EBNO_DB = 20

ERROR_LEVELS = [0, 1, 25, 50, 75, 100]

CSV_PATH = "results/csv/key_error_sweep.csv"
FIG_PATH = "results/figures/key_error_sweep.png"
LOG_PATH = "results/logs/key_error_sweep.txt"

os.makedirs("results/csv", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

np.random.seed(1234)

# ==========================================================
# Fixed Monte-Carlo Dataset
# ==========================================================

dataset = []

master_key = get_master_key()
iv = get_iv()

for block in range(NUM_BLOCKS):

    info_bits = np.random.randint(
        0, 2, POLAR_K,
        dtype=np.uint8
    )

    frame_id = FRAME_ID + block // 20
    block_id = block % 20

    counter = block_counter(frame_id, block_id)

    keystream = generate_keystream(
        master_key,
        iv,
        counter
    )

    dataset.append((info_bits, keystream))

# ==========================================================
# Corrupt Keystream
# ==========================================================

def corrupt_keystream(bits, error_percent):
    """
    Corrupt the keystream by flipping a percentage of bits.

    0%   : original keystream.
    1-99%: flip exactly that percentage of bits.
    100% : generate a completely independent random keystream
            (not simply the bitwise complement).
    """

    bits = bits.copy()

    # Correct key
    if error_percent == 0:
        return bits

    # Completely wrong key (independent key)
    if error_percent == 100:
        rng = np.random.default_rng()
        return rng.integers(
            0, 2, len(bits), dtype=np.uint8
        )

    # Partial key mismatch
    n_flip = max(
        1,
        int(round(len(bits) * error_percent / 100))
    )

    index = np.random.choice(
        len(bits),
        n_flip,
        replace=False
    )

    bits[index] ^= 1

    return bits

# ==========================================================
# Sweep
# ==========================================================

records = []

print("=" * 60)
print("KEY ERROR SWEEP EXPERIMENT")
print("=" * 60)

for error in ERROR_LEVELS:

    bob_errors = 0
    eve_errors = 0
    frozen_errors = 0

    total_bits = NUM_BLOCKS * POLAR_K

    for block, (info_bits, keystream) in enumerate(dataset):

        eve_keystream = corrupt_keystream(
            keystream,
            error
        )

        # ---------------- Encryption ----------------

        u = build_crypto_u_vector(
            info_bits,
            keystream
        )

        cipher_bits, _ = extract_crypto_bits(u)

        codeword = polar_encode(
            cipher_bits,
            frozen_bits=keystream
        )

        tx = bpsk_modulate(codeword)

        rx, sigma2 = add_awgn(
            tx,
            EBNO_DB
        )

        llr = llr_awgn(rx, sigma2)

        # ======================================================
        # Bob
        # ======================================================

        _, u_hat_bob = polar_decode(
            llr,
            frozen_bits=keystream
        )

        cipher_hat_bob, frozen_hat_bob = extract_crypto_bits(
            u_hat_bob
        )

        info_hat_bob = decrypt_information_bits(
            cipher_hat_bob,
            keystream
        )

        bob_errors += np.sum(
            info_hat_bob != info_bits
        )

        # ======================================================
        # Eve
        # ======================================================

        _, u_hat_eve = polar_decode(
            llr,
            frozen_bits=eve_keystream
        )

        cipher_hat_eve, frozen_hat_eve = extract_crypto_bits(
            u_hat_eve
        )

        info_hat_eve = decrypt_information_bits(
            cipher_hat_eve,
            eve_keystream
        )

        eve_errors += np.sum(
            info_hat_eve != info_bits
        )

        _, frozen_ber = verify_frozen_bits(
            frozen_hat_eve,
            keystream
        )

        frozen_errors += frozen_ber

    bob_ber = bob_errors / total_bits
    eve_ber = eve_errors / total_bits
    frozen_ber = frozen_errors / NUM_BLOCKS

    records.append({
        "KeyError(%)": error,
        "BobBER": bob_ber,
        "EveBER": eve_ber,
        "FrozenBER": frozen_ber,
    })

    print(
        f"{error:>3}% | Bob BER={bob_ber:.6f} | "
        f"Eve BER={eve_ber:.6f} | Frozen BER={frozen_ber:.6f}"
    )

# ==========================================================
# CSV
# ==========================================================

df = pd.DataFrame(records)
df.to_csv(CSV_PATH, index=False)

# ==========================================================
# IEEE Figure
# ==========================================================

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]

plt.figure(figsize=(6.4, 4.2), facecolor="white")

plt.plot(
    df["KeyError(%)"],
    df["EveBER"],
    marker="o",
    linewidth=2.2,
    markersize=6,
    label="Eavesdropper BER"
)

plt.plot(
    df["KeyError(%)"],
    df["BobBER"],
    marker="s",
    linewidth=2,
    markersize=5,
    label="Legitimate BER"
)

plt.axhline(
    0.5,
    linestyle="--",
    linewidth=1.2,
    label="Random Guess BER = 0.5"
)

plt.xticks(ERROR_LEVELS)
plt.ylim(-0.02, 0.55)

plt.xlabel("Keystream Error Probability (%)", fontsize=11)
plt.ylabel("Bit Error Rate (BER)", fontsize=11)

plt.grid(alpha=0.3)
plt.legend(frameon=False, loc="lower right")

plt.tight_layout()

plt.savefig(
    FIG_PATH,
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# LOG
# ==========================================================

with open(LOG_PATH, "w") as f:

    f.write("Key Error Sweep Experiment\n")
    f.write("=" * 40 + "\n\n")

    f.write(f"Blocks : {NUM_BLOCKS}\n")
    f.write(f"Eb/N0  : {EBNO_DB} dB\n\n")

    for row in records:
        f.write(
            f"{row['KeyError(%)']:>3}% | "
            f"Bob={row['BobBER']:.6f} | "
            f"Eve={row['EveBER']:.6f} | "
            f"Frozen={row['FrozenBER']:.6f}\n"
        )

print("\nFinished.")
print("CSV :", CSV_PATH)
print("PNG :", FIG_PATH)
print("LOG :", LOG_PATH)