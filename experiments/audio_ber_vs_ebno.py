"""
experiments/audio_ber_vs_ebno.py
============================================================
Audio BER vs Eb/N0 Experiment (DATN v1.2)

Evaluate Secure Audio Crypto-Coding over AWGN.

Target
------
Raspberry Pi 4 / Raspberry Pi 3

Polar Code : (512,256)

Input
-----
data/voice.wav

Outputs
-------
results/audio/recovered_voice.wav
results/csv/audio_ber_vs_ebno.csv
results/figures/audio_ber_vs_ebno.png
results/logs/audio_ber_vs_ebno.txt

Features
--------
✓ Optimized for Raspberry Pi 4
✓ FAST / FULL Mode
✓ Progress Bar
✓ IEEE Plot (600 DPI)
✓ Runtime Statistics
============================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm

from audio.audio_io import (
    read_wav,
    write_wav,
    audio_to_bits,
    bits_to_audio,
)

from audio.audio_blocks import (
    split_audio_blocks,
    merge_audio_blocks,
    remove_padding,
    frame_block_index,
)

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

# ==========================================================
# Configuration
# ==========================================================

INPUT_AUDIO = "data/voice.wav"
OUTPUT_AUDIO = "results/audio/recovered_voice.wav"

EBNO_LIST = np.arange(-5, 35, 5)

# ----------------------------------------------------------
# FAST MODE
# ----------------------------------------------------------
# True  : 2000 blocks (~512k bits) -> khoảng 2 phút trên Pi4.
# False : toàn bộ voice.wav -> kết quả chính thức luận văn.
# ----------------------------------------------------------

FAST_MODE = True
MAX_BLOCKS = 2000 if FAST_MODE else None

CSV_PATH = "results/csv/audio_ber_vs_ebno.csv"
FIG_PATH = "results/figures/audio_ber_vs_ebno.png"
LOG_PATH = "results/logs/audio_ber_vs_ebno.txt"

os.makedirs("results/audio", exist_ok=True)
os.makedirs("results/csv", exist_ok=True)
os.makedirs("results/figures", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

# ==========================================================
# Read Audio
# ==========================================================

samples, sample_rate = read_wav(INPUT_AUDIO)

audio_bits = audio_to_bits(samples)

blocks, pad_bits = split_audio_blocks(audio_bits)

if MAX_BLOCKS is not None:
    blocks = blocks[:MAX_BLOCKS]

audio_bits = merge_audio_blocks(blocks)
audio_bits = remove_padding(audio_bits, pad_bits)

NUM_BLOCKS = len(blocks)
NUM_FRAMES = int(np.ceil(NUM_BLOCKS / 20))

print("=" * 60)
print("AUDIO BER vs Eb/N0 EXPERIMENT (DATN v1.2)")
print("=" * 60)
print("Input Audio :", INPUT_AUDIO)
print("Sample Rate :", sample_rate, "Hz")
print("Audio Bits  :", len(audio_bits))
print("Blocks      :", NUM_BLOCKS)
print("Frames      :", NUM_FRAMES)
print("Mode        :", "FAST" if FAST_MODE else "FULL")
print()

# ==========================================================
# Cache Keys (Generated ONCE)
# ==========================================================

MASTER_KEY = get_master_key()
IV = get_iv()

# Precompute keystream for every block.
KEYSTREAM_CACHE = []

for block in range(NUM_BLOCKS):

    frame_id, block_id = frame_block_index(block)

    counter = block_counter(frame_id, block_id)

    KEYSTREAM_CACHE.append(
        generate_keystream(
            MASTER_KEY,
            IV,
            counter
        )
    )

print("Keystream Cache Built :", len(KEYSTREAM_CACHE), "blocks\n")

# ==========================================================
# Experiment
# ==========================================================

records = []

experiment_start = time.time()

for ebno in EBNO_LIST:

    start_time = time.time()

    recovered_blocks = []

    total_errors = 0
    total_bits = 0

    progress = tqdm(
        range(NUM_BLOCKS),
        desc=f"Eb/N0 {ebno:>3} dB",
        unit="blk",
        ncols=95,
    )

    for blk in progress:

        info_bits = blocks[blk]
        keystream = KEYSTREAM_CACHE[blk]

        # ---------------- Crypto Coding ----------------

        u = build_crypto_u_vector(
            info_bits,
            keystream
        )

        cipher_bits, _ = extract_crypto_bits(u)

        # ---------------- Polar Encoder ----------------

        codeword = polar_encode(
            cipher_bits,
            frozen_bits=keystream
        )

        # ---------------- AWGN Channel ----------------

        tx = bpsk_modulate(codeword)

        rx, sigma2 = add_awgn(
            tx,
            ebno_db=float(ebno)
        )

        llr = llr_awgn(rx, sigma2)

        # ---------------- Polar Decoder ----------------

        _, u_hat = polar_decode(
            llr,
            frozen_bits=keystream
        )

        cipher_hat, _ = extract_crypto_bits(u_hat)

        recover_bits = decrypt_information_bits(
            cipher_hat,
            keystream
        )

        recovered_blocks.append(recover_bits)

        errors = np.sum(recover_bits != info_bits)

        total_errors += errors
        total_bits += len(info_bits)

    # ======================================================
    # BER
    # ======================================================

    ber = total_errors / total_bits

    runtime = time.time() - start_time

    records.append({
        "EbN0(dB)": float(ebno),
        "BER": float(ber),
        "Errors": int(total_errors),
        "Bits": int(total_bits),
        "Blocks": NUM_BLOCKS,
        "Frames": NUM_FRAMES,
        "Runtime(s)": runtime,
    })

    print(
        f"Eb/N0={ebno:>3} dB | "
        f"BER={ber:.6f} | "
        f"Errors={total_errors} | "
        f"Runtime={runtime:.2f}s"
    )

    # Save recovered audio at 20 dB
    if ebno == 20:

        recovered_stream = merge_audio_blocks(recovered_blocks)
        recovered_stream = remove_padding(recovered_stream, pad_bits)

        recovered_audio = bits_to_audio(recovered_stream)

        write_wav(
            OUTPUT_AUDIO,
            recovered_audio,
            sample_rate
        )

# ==========================================================
# CSV
# ==========================================================

df = pd.DataFrame(records)

df.to_csv(
    CSV_PATH,
    index=False
)

# ==========================================================
# IEEE Figure
# ==========================================================

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 11

plot_ber = df["BER"].copy()

# Do not draw BER = 0 on logarithmic axis.
plot_ber[plot_ber == 0] = np.nan

plt.figure(figsize=(6.5, 4.2), facecolor="white")

plt.semilogy(
    df["EbN0(dB)"],
    plot_ber,
    marker="o",
    linewidth=2,
    markersize=6,
    label="Secure Audio Crypto-Coding"
)

plt.grid(True, which="both", linestyle="--", alpha=0.35)

plt.xlabel("Eb/N0 (dB)")
plt.ylabel("Bit Error Rate (BER)")

plt.xticks(EBNO_LIST)

plt.legend(frameon=False, loc="upper right")

plt.tight_layout()

plt.savefig(
    FIG_PATH,
    dpi=600,
    bbox_inches="tight",
)

plt.close()

# ==========================================================
# Log
# ==========================================================

total_runtime = time.time() - experiment_start

with open(LOG_PATH, "w") as f:

    f.write("Audio BER vs Eb/N0 Experiment (DATN v1.2)\n")
    f.write("=" * 50 + "\n\n")

    f.write(f"Input Audio : {INPUT_AUDIO}\n")
    f.write(f"Sample Rate : {sample_rate} Hz\n")
    f.write(f"Audio Bits  : {len(audio_bits)}\n")
    f.write(f"Blocks      : {NUM_BLOCKS}\n")
    f.write(f"Frames      : {NUM_FRAMES}\n")
    f.write(f"Mode        : {'FAST' if FAST_MODE else 'FULL'}\n\n")

    for row in records:

        f.write(
            f"Eb/N0={row['EbN0(dB)']:>3.0f} dB | "
            f"BER={row['BER']:.6f} | "
            f"Errors={row['Errors']} | "
            f"Bits={row['Bits']} | "
            f"Runtime={row['Runtime(s)']:.2f}s\n"
        )

    f.write("\n")
    f.write(f"Total Runtime : {total_runtime:.2f} s\n")

# ==========================================================
# Summary
# ==========================================================

print("\n" + "=" * 60)
print("EXPERIMENT FINISHED")
print("=" * 60)

print(f"Total Runtime : {total_runtime:.2f} s")

print("Recovered Audio :", OUTPUT_AUDIO)
print("CSV             :", CSV_PATH)
print("Figure          :", FIG_PATH)
print("Log             :", LOG_PATH)