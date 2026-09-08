"""
experiments/audio_wrong_key_analysis.py
============================================================
Wrong Master Key Audio Security Analysis (DATN v1.0)

Purpose
-------
Evaluate Bob vs Eve using the same transmitted Polar codeword.

Bob : Correct Master Key.
Eve : Wrong 256-bit Master Key.

Outputs
-------
results/audio/wrong_key_voice.wav
results/csv/wrong_key_audio_metrics.csv
results/logs/wrong_key_audio_analysis.txt
============================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import spectrogram
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

from metrics.audio_metrics import evaluate_audio

# ==========================================================
# Configuration
# ==========================================================

INPUT_AUDIO = "data/voice.wav"

BOB_AUDIO = "results/audio/recovered_voice.wav"
EVE_AUDIO = "results/audio/wrong_key_voice.wav"

CSV_FILE = "results/csv/wrong_key_audio_metrics.csv"
LOG_FILE = "results/logs/wrong_key_audio_analysis.txt"

EBNO_DB = 20

os.makedirs("results/audio", exist_ok=True)
os.makedirs("results/csv", exist_ok=True)
os.makedirs("results/logs", exist_ok=True)

# ==========================================================
# Read Audio
# ==========================================================

samples, sample_rate = read_wav(INPUT_AUDIO)

audio_bits = audio_to_bits(samples)

blocks, pad_bits = split_audio_blocks(audio_bits)

NUM_BLOCKS = len(blocks)

print("="*60)
print("WRONG MASTER KEY AUDIO ANALYSIS")
print("="*60)

print("Input Audio :", INPUT_AUDIO)
print("Blocks      :", NUM_BLOCKS)
print("Eb/N0       :", EBNO_DB, "dB")

# ==========================================================
# Keys
# ==========================================================

master_key = get_master_key()
iv = get_iv()

# Convert bytes -> mutable bytearray
wrong_master_key = bytearray(master_key)

# Flip 8 bits của byte đầu tiên để tạo Wrong Master Key
wrong_master_key[0] ^= 0xFF

# Đổi lại thành bytes để generate_keystream() sử dụng
wrong_master_key = bytes(wrong_master_key)
# ==========================================================
# Transmission
# ==========================================================

bob_blocks = []
eve_blocks = []

bob_errors = 0
eve_errors = 0
total_bits = 0

start = time.time()

for blk in tqdm(range(NUM_BLOCKS), ncols=90):

    info_bits = blocks[blk]

    frame_id, block_id = frame_block_index(blk)

    counter = block_counter(frame_id, block_id)

    # Correct keystream
    ks_tx = generate_keystream(master_key, iv, counter)

    # Eve keystream
    ks_wrong = generate_keystream(wrong_master_key, iv, counter)

    # Crypto Coding
    u = build_crypto_u_vector(info_bits, ks_tx)

    cipher_bits, _ = extract_crypto_bits(u)

    codeword = polar_encode(
        cipher_bits,
        frozen_bits=ks_tx
    )

    # Channel
    tx = bpsk_modulate(codeword)

    rx, sigma2 = add_awgn(
        tx,
        ebno_db=EBNO_DB
    )

    llr = llr_awgn(rx, sigma2)

    # Bob
    _, u_bob = polar_decode(
        llr,
        frozen_bits=ks_tx
    )

    cipher_bob, _ = extract_crypto_bits(u_bob)

    info_bob = decrypt_information_bits(
        cipher_bob,
        ks_tx
    )

    bob_blocks.append(info_bob)

    bob_errors += np.sum(info_bob != info_bits)

    # Eve
    _, u_eve = polar_decode(
        llr,
        frozen_bits=ks_wrong
    )

    cipher_eve, _ = extract_crypto_bits(u_eve)

    info_eve = decrypt_information_bits(
        cipher_eve,
        ks_wrong
    )

    eve_blocks.append(info_eve)

    eve_errors += np.sum(info_eve != info_bits)

    total_bits += len(info_bits)

runtime = time.time() - start

# ==========================================================
# BER
# ==========================================================

bob_ber = bob_errors / total_bits
eve_ber = eve_errors / total_bits

print("\nBER Results")
print("Bob BER :", bob_ber)
print("Eve BER :", eve_ber)

# ==========================================================
# Reconstruct Audio
# ==========================================================

bob_stream = merge_audio_blocks(bob_blocks)
bob_stream = remove_padding(bob_stream, pad_bits)

eve_stream = merge_audio_blocks(eve_blocks)
eve_stream = remove_padding(eve_stream, pad_bits)

bob_audio = bits_to_audio(bob_stream)
eve_audio = bits_to_audio(eve_stream)

write_wav(
    BOB_AUDIO,
    bob_audio,
    sample_rate
)

write_wav(
    EVE_AUDIO,
    eve_audio,
    sample_rate
)

# ==========================================================
# Metrics
# ==========================================================

bob_metrics = evaluate_audio(
    reference_audio=samples,
    encrypted_audio=eve_audio,
    recovered_audio=bob_audio,
    reference_bits=audio_bits,
    recovered_bits=bob_stream,
)

eve_metrics = evaluate_audio(
    reference_audio=samples,
    encrypted_audio=eve_audio,
    recovered_audio=eve_audio,
    reference_bits=audio_bits,
    recovered_bits=eve_stream,
)

# ==========================================================
# CSV
# ==========================================================

df = pd.DataFrame([
    {
        "Receiver":"Bob",
        "BER":bob_ber,
        "Entropy":bob_metrics["RecoveredEntropy"],
        "Correlation":bob_metrics["RecoveredCorrelation"],
        "MSE":bob_metrics["MSE"],
        "PSNR(dB)":bob_metrics["PSNR"],
        "SNR(dB)":bob_metrics["SNR"]
    },
    {
        "Receiver":"Eve",
        "BER":eve_ber,
        "Entropy":eve_metrics["RecoveredEntropy"],
        "Correlation":eve_metrics["RecoveredCorrelation"],
        "MSE":eve_metrics["MSE"],
        "PSNR(dB)":eve_metrics["PSNR"],
        "SNR(dB)":eve_metrics["SNR"]
    }
])

df.to_csv(CSV_FILE, index=False)

# ==========================================================
# LOG
# ==========================================================

with open(LOG_FILE,"w") as f:

    f.write("Wrong Master Key Audio Analysis\n")
    f.write("="*45+"\n\n")

    f.write(f"Audio File : {INPUT_AUDIO}\n")
    f.write(f"Blocks     : {NUM_BLOCKS}\n")
    f.write(f"Eb/N0      : {EBNO_DB} dB\n\n")

    f.write("Bob\n")
    f.write(f"BER         : {bob_ber:.8f}\n")
    f.write(f"Entropy     : {bob_metrics['RecoveredEntropy']:.6f}\n")
    f.write(f"Correlation : {bob_metrics['RecoveredCorrelation']:.6f}\n")
    f.write(f"PSNR        : {bob_metrics['PSNR']}\n")
    f.write(f"SNR         : {bob_metrics['SNR']}\n\n")

    f.write("Eve\n")
    f.write(f"BER         : {eve_ber:.8f}\n")
    f.write(f"Entropy     : {eve_metrics['RecoveredEntropy']:.6f}\n")
    f.write(f"Correlation : {eve_metrics['RecoveredCorrelation']:.6f}\n")
    f.write(f"PSNR        : {eve_metrics['PSNR']}\n")
    f.write(f"SNR         : {eve_metrics['SNR']}\n\n")

    f.write(f"Runtime : {runtime:.2f} seconds\n")

# ==========================================================
# Summary
# ==========================================================

print("\nAudio Saved")
print("Bob :", BOB_AUDIO)
print("Eve :", EVE_AUDIO)

print("\nCSV :", CSV_FILE)
print("LOG :", LOG_FILE)

print("\nRuntime :", round(runtime,2),"seconds")

print("\nLAB 11F-A COMPLETED")
# ==========================================================
# LAB 11F-B
# IEEE Security Visualization
# ==========================================================

# ==========================================================
# LAB 11F-B (DATN v1.2)
# IEEE Security Visualization
# ==========================================================

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman"],
    "font.size": 11,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "figure.facecolor": "white"
})

FIG_DIR = "results/figures"
os.makedirs(FIG_DIR, exist_ok=True)

COLOR_ORIGINAL = "#2563EB"   # Blue
COLOR_BOB      = "#16A34A"   # Green
COLOR_EVE      = "#DC2626"   # Red
COLOR_GRAY     = "#6B7280"
def save_ieee(fig, filename):

    fig.tight_layout()

    fig.savefig(
        os.path.join(FIG_DIR, filename),
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )

    plt.close(fig)
def plot_waveform_security():

    seconds = 3
    N = sample_rate * seconds
    t = np.arange(N) / sample_rate

    ylim = np.max(np.abs(samples[:N])) * 1.05

    fig, axs = plt.subplots(3,1, figsize=(6.4,4.8), sharex=True)

    data = [
        (samples, COLOR_ORIGINAL, "Original Audio"),
        (bob_audio, COLOR_BOB, "Bob (Correct Master Key)"),
        (eve_audio, COLOR_EVE, "Eve (Wrong Master Key)")
    ]

    for ax,(sig,color,title) in zip(axs,data):

        ax.plot(
            t,
            sig[:N],
            color=color,
            linewidth=0.6
        )

        ax.set_ylim(-ylim, ylim)
        ax.set_ylabel("Amplitude")
        ax.set_title(title, fontweight="normal")
        ax.grid(alpha=0.25)

    axs[-1].set_xlabel("Time (s)")

    save_ieee(fig, "waveform_wrong_key.png")
def draw_spec(ax, signal, title):

    signal = signal[:sample_rate*3]

    f,tt,S = spectrogram(
        signal.astype(np.float32),
        fs=sample_rate,
        window="hann",
        nperseg=512,
        noverlap=384,
        mode="magnitude"
    )

    S = 20*np.log10(S + 1e-12)

    pcm = ax.pcolormesh(
        tt,
        f/1000,
        S,
        shading="gouraud",
        cmap="jet",
        vmin=-80,
        vmax=50
    )

    ax.set_ylabel("kHz")
    ax.set_ylim(0,8)
    ax.set_title(title, fontweight="normal")
    ax.grid(False)

    return pcm


def plot_spectrogram_security():

    fig,axs = plt.subplots(
        3,1,
        figsize=(6.4,6.2),
        sharex=True
    )

    pcm = draw_spec(axs[0], samples, "Original Audio")
    draw_spec(axs[1], bob_audio, "Bob (Correct Master Key)")
    draw_spec(axs[2], eve_audio, "Eve (Wrong Master Key)")

    axs[-1].set_xlabel("Time (s)")

    cbar = fig.colorbar(
        pcm,
        ax=axs,
        fraction=0.03,
        pad=0.02
    )

    cbar.set_label("Magnitude (dB)")

    save_ieee(fig, "spectrogram_wrong_key.png")
def plot_entropy():

    labels = ["Original","Bob","Eve"]

    entropy = [
        bob_metrics["OriginalEntropy"],
        bob_metrics["RecoveredEntropy"],
        eve_metrics["RecoveredEntropy"]
    ]

    colors = [
        COLOR_ORIGINAL,
        COLOR_BOB,
        COLOR_EVE
    ]

    fig = plt.figure(figsize=(5.6,3.5))

    bars = plt.bar(
        labels,
        entropy,
        color=colors,
        edgecolor="black",
        linewidth=0.6
    )

    plt.ylabel("Entropy (bits/sample)")
    plt.ylim(0,8.2)

    plt.title(
        "Shannon Entropy Comparison",
        fontweight="normal"
    )

    plt.grid(axis="y", alpha=0.3)

    for bar,val in zip(bars,entropy):

        plt.text(
            bar.get_x()+bar.get_width()/2,
            val+0.05,
            f"{val:.3f}",
            ha="center",
            fontsize=9
        )

    save_ieee(fig,"entropy_security.png")
def scatter_corr(ax, signal, title, color):

    N = 6000

    ax.scatter(
        signal[:N],
        signal[1:N+1],
        s=2,
        alpha=0.25,
        color=color
    )

    ax.set_xlabel("x(n)")
    ax.set_ylabel("x(n+1)")
    ax.set_title(title,fontweight="normal")
    ax.grid(alpha=0.2)


def plot_correlation():

    fig,axs = plt.subplots(
        1,2,
        figsize=(6.4,3.2)
    )

    scatter_corr(
        axs[0],
        samples,
        "Original Audio",
        COLOR_ORIGINAL
    )

    scatter_corr(
        axs[1],
        eve_audio,
        "Wrong Master Key",
        COLOR_EVE
    )

    save_ieee(fig,"correlation_security.png")
def plot_quality():

    labels = ["Bob","Eve"]

    psnr_plot = [
        60 if np.isinf(bob_metrics["PSNR"]) else bob_metrics["PSNR"],
        eve_metrics["PSNR"]
    ]

    snr_plot = [
        60 if np.isinf(bob_metrics["SNR"]) else bob_metrics["SNR"],
        eve_metrics["SNR"]
    ]

    x = np.arange(len(labels))
    width = 0.32

    fig = plt.figure(figsize=(6.2,3.5))

    b1 = plt.bar(
        x-width/2,
        psnr_plot,
        width,
        color=COLOR_ORIGINAL,
        label="PSNR"
    )

    b2 = plt.bar(
        x+width/2,
        snr_plot,
        width,
        color="#EA580C",
        label="SNR"
    )

    plt.xticks(x,labels)
    plt.ylabel("dB")

    plt.ylim(-20,65)

    plt.title(
        "Audio Quality Metrics",
        fontweight="normal"
    )

    plt.grid(axis="y", alpha=0.3)
    plt.legend(frameon=False)

    plt.text(
        b1[0].get_x()+width/2,
        61,
        "∞",
        ha="center",
        fontsize=11
    )

    plt.text(
        b2[0].get_x()+width/2,
        61,
        "∞",
        ha="center",
        fontsize=11
    )

    plt.text(
        b1[1].get_x()+width/2,
        eve_metrics["PSNR"]+2,
        f"{eve_metrics['PSNR']:.2f}",
        ha="center",
        fontsize=9
    )

    plt.text(
        b2[1].get_x()+width/2,
        eve_metrics["SNR"]-3,
        f"{eve_metrics['SNR']:.2f}",
        ha="center",
        fontsize=9
    )

    save_ieee(fig,"quality_security.png")
def plot_ber():

    values = [bob_ber, eve_ber]

    fig = plt.figure(figsize=(5.5,3.5))

    bars = plt.bar(
        ["Bob","Eve"],
        values,
        color=[COLOR_BOB,COLOR_EVE],
        edgecolor="black",
        linewidth=0.6
    )

    plt.axhline(
        0.5,
        linestyle="--",
        linewidth=1.2,
        color="red",
        label="Random Guess BER = 0.5"
    )

    plt.ylabel("Bit Error Rate")
    plt.ylim(0,0.55)

    plt.title(
        "Recovered BER under Wrong Master Key Attack",
        fontweight="normal"
    )

    plt.grid(axis="y", alpha=0.3)

    for bar,val in zip(bars,values):

        plt.text(
            bar.get_x()+bar.get_width()/2,
            val+0.012,
            f"{val:.3f}",
            ha="center",
            fontsize=9
        )

    plt.legend(frameon=False)

    save_ieee(fig,"ber_bob_vs_eve.png")
def plot_security_summary():

    labels = ["BER","Entropy","Correlation"]

    bob = [
        bob_ber/0.5,
        bob_metrics["RecoveredEntropy"]/8,
        abs(bob_metrics["RecoveredCorrelation"])
    ]

    eve = [
        eve_ber/0.5,
        eve_metrics["RecoveredEntropy"]/8,
        abs(eve_metrics["RecoveredCorrelation"])
    ]

    x = np.arange(len(labels))
    width = 0.32

    fig = plt.figure(figsize=(6.0,3.6))

    plt.bar(
        x-width/2,
        bob,
        width,
        color=COLOR_BOB,
        label="Bob"
    )

    plt.bar(
        x+width/2,
        eve,
        width,
        color=COLOR_EVE,
        label="Eve"
    )

    plt.xticks(x,labels)

    plt.ylabel("Normalized Value")
    plt.ylim(0,1.05)

    plt.title(
        "Security Metric Summary",
        fontweight="normal"
    )

    plt.grid(axis="y", alpha=0.3)
    plt.legend(frameon=False)

    save_ieee(fig,"security_summary.png")
print("\nGenerating IEEE Security Figures...")

figure_start = time.time()

plot_waveform_security()
plot_spectrogram_security()
plot_entropy()
plot_correlation()
plot_quality()
plot_ber()
plot_security_summary()

figure_runtime = time.time() - figure_start

print(
    f"Security Figure Runtime : {figure_runtime:.2f} seconds"
)
print("Figures Saved -> results/figures/")