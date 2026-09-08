"""
experiments/wrong_key_montecarlo.py
============================================================
Monte Carlo Wrong-Key Security Evaluation

Crypto-Coding Polar (512,256)

Outputs
-------
CSV
PNG (600 DPI)
LOG
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
)

from polar.encoder import polar_encode
from polar.sc_decoder import polar_decode

from polar.utils import (
    bpsk_modulate,
    add_awgn,
    llr_awgn,
)

# ======================================================
# Configuration
# ======================================================

NUM_BLOCKS = 1000
FRAME_ID = 0
EBNO_DB = 20

CSV_PATH = "results/csv/wrong_key_montecarlo.csv"
FIG_PATH = "results/figures/wrong_key_montecarlo.png"
LOG_PATH = "results/logs/wrong_key_montecarlo.txt"

os.makedirs("results/csv", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

np.random.seed(42)

# ======================================================
# Keys
# ======================================================

master_key = get_master_key()
iv = get_iv()

wrong_key = bytearray(master_key)
wrong_key[0] ^= 0x01
wrong_key = bytes(wrong_key)

# ======================================================
# Monte Carlo
# ======================================================

correct_history = []
wrong_history = []

records = []

total_correct_errors = 0
total_wrong_errors = 0

total_bits = NUM_BLOCKS * POLAR_K

print("=" * 60)
print("MONTE CARLO WRONG KEY EVALUATION")
print("=" * 60)

for block in range(NUM_BLOCKS):

    info_bits = np.random.randint(
        0,2,POLAR_K,dtype=np.uint8
    )
    frame_id = FRAME_ID + block // 20
    block_id = block % 20
    counter = block_counter(
    frame_id,
    block_id
)
    

    keystream = generate_keystream(
        master_key,
        iv,
        counter
    )

    wrong_keystream = generate_keystream(
        wrong_key,
        iv,
        counter
    )

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

    _, u_hat = polar_decode(
        llr,
        frozen_bits=keystream
    )

    cipher_hat, _ = extract_crypto_bits(u_hat)

    # Bob
    info_bob = decrypt_information_bits(
        cipher_hat,
        keystream
    )

    # Eve
    info_eve = decrypt_information_bits(
        cipher_hat,
        wrong_keystream
    )

    ber_bob = np.mean(info_bits != info_bob)
    ber_eve = np.mean(info_bits != info_eve)

    total_correct_errors += np.sum(info_bits != info_bob)
    total_wrong_errors += np.sum(info_bits != info_eve)

    correct_history.append(
        total_correct_errors /
        ((block+1)*POLAR_K)
    )

    wrong_history.append(
        total_wrong_errors /
        ((block+1)*POLAR_K)
    )

    records.append({
        "Block": block+1,
        "CorrectBER": ber_bob,
        "WrongBER": ber_eve,
        "RunningWrongBER": wrong_history[-1]
    })

# ======================================================
# Save CSV
# ======================================================

df = pd.DataFrame(records)
df.to_csv(CSV_PATH, index=False)

# ======================================================
# Figure
# ======================================================

plt.figure(figsize=(7,4))

plt.plot(
    wrong_history,
    linewidth=2,
    label="Wrong Key BER"
)

plt.plot(
    correct_history,
    linewidth=2,
    label="Correct Key BER"
)

plt.axhline(
    0.5,
    linestyle="--",
    linewidth=1,
    label="Random Guess BER = 0.5"
)

plt.xlabel("Monte Carlo Blocks")
plt.ylabel("Running BER")
plt.title("Wrong-Key BER Convergence")
plt.grid(True)
plt.legend()

plt.tight_layout()

plt.savefig(
    FIG_PATH,
    dpi=600
)

plt.close()

# ======================================================
# Log
# ======================================================

with open(LOG_PATH,"w") as f:

    f.write("Monte Carlo Wrong Key Evaluation\n")
    f.write("="*45+"\n\n")
    f.write(f"Blocks        : {NUM_BLOCKS}\n")
    f.write(f"Eb/N0         : {EBNO_DB} dB\n\n")

    f.write(
        f"Bob BER       : {correct_history[-1]:.6f}\n"
    )

    f.write(
        f"Eve BER       : {wrong_history[-1]:.6f}\n"
    )

print("\nFinished.")
print("CSV :", CSV_PATH)
print("PNG :", FIG_PATH)
print("LOG :", LOG_PATH)

print("\nFinal Bob BER :", correct_history[-1])
print("Final Eve BER :", wrong_history[-1])