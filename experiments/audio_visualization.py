"""
experiments/audio_visualization.py
============================================================
Audio Visualization Experiment (DATN v1.0)

Chapter 4 Visualization for Secure Audio Crypto-Coding

Generate
--------
1. Original Audio
2. Encrypted Audio
3. Recovered Audio

Later sections generate IEEE figures and metrics.

Target
------
Raspberry Pi 4 / Raspberry Pi 3

Polar Code : (512,256)
AES Mode   : CTR (256-bit Keystream)

Input
-----
data/voice.wav

Outputs
-------
results/audio/encrypted_voice.wav
results/audio/recovered_voice.wav

results/figures/
results/csv/
results/logs/

============================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.signal import spectrogram
from tqdm import tqdm

# ==========================================================
# Audio Modules
# ==========================================================

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

# ==========================================================
# Crypto Modules
# ==========================================================

from crypto.key_manager import (
    get_master_key,
    get_iv,
    block_counter,
)

from crypto.keystream import generate_keystream

# ==========================================================
# Polar Modules
# ==========================================================

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
# Metrics
# ==========================================================

from metrics.audio_metrics import (
    evaluate_audio,
)

# ==========================================================
# Configuration
# ==========================================================

INPUT_AUDIO = "data/voice.wav"

ENCRYPTED_AUDIO = "results/audio/encrypted_voice.wav"
RECOVERED_AUDIO = "results/audio/recovered_voice.wav"

FIG_DIR = "results/figures"
CSV_DIR = "results/csv"
LOG_DIR = "results/logs"
AUDIO_DIR = "results/audio"

EBNO_DB = 20

FAST_MODE = False
MAX_BLOCKS = 2000 if FAST_MODE else None

os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(AUDIO_DIR, exist_ok=True)

# ==========================================================
# IEEE Figure Style
# ==========================================================

plt.rcParams["font.family"] = "serif"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["font.size"] = 11
plt.rcParams["figure.facecolor"] = "white"

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
print("AUDIO VISUALIZATION EXPERIMENT")
print("=" * 60)

print("Input Audio :", INPUT_AUDIO)
print("Sample Rate :", sample_rate)
print("Samples     :", len(samples))
print("Audio Bits  :", len(audio_bits))
print("Blocks      :", NUM_BLOCKS)
print("Frames      :", NUM_FRAMES)
print("Eb/N0       :", EBNO_DB, "dB")
print("Mode        :", "FAST" if FAST_MODE else "FULL")

# ==========================================================
# Generate AES Keystream Cache
# ==========================================================

print("\nGenerating AES Keystream Cache...")

MASTER_KEY = get_master_key()
IV = get_iv()

KEYSTREAM_CACHE = []

for block in tqdm(range(NUM_BLOCKS), ncols=90):

    frame_id, block_id = frame_block_index(block)

    counter = block_counter(frame_id, block_id)

    KEYSTREAM_CACHE.append(
        generate_keystream(
            MASTER_KEY,
            IV,
            counter
        )
    )

print("Keystream Cache Ready :", len(KEYSTREAM_CACHE))

# ==========================================================
# Secure Audio Transmission
# ==========================================================

print("\nRunning Secure Audio Pipeline...")

encrypted_blocks = []
recovered_blocks = []

total_errors = 0
total_bits = 0

pipeline_start = time.time()

for blk in tqdm(range(NUM_BLOCKS), ncols=90):

    info_bits = blocks[blk]
    keystream = KEYSTREAM_CACHE[blk]

    # ------------------------------------------------------
    # Crypto Coding
    # ------------------------------------------------------

    u = build_crypto_u_vector(
        info_bits,
        keystream
    )

    cipher_bits, _ = extract_crypto_bits(u)

    encrypted_blocks.append(cipher_bits.copy())

    # ------------------------------------------------------
    # Polar Encode
    # ------------------------------------------------------

    codeword = polar_encode(
        cipher_bits,
        frozen_bits=keystream
    )

    # ------------------------------------------------------
    # Channel
    # ------------------------------------------------------

    tx = bpsk_modulate(codeword)

    rx, sigma2 = add_awgn(
        tx,
        ebno_db=EBNO_DB
    )

    llr = llr_awgn(
        rx,
        sigma2
    )

    # ------------------------------------------------------
    # Polar Decode
    # ------------------------------------------------------

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

    total_errors += np.sum(
        recover_bits != info_bits
    )

    total_bits += len(info_bits)

pipeline_runtime = time.time() - pipeline_start

audio_ber = total_errors / total_bits

print("\nPipeline Finished.")
print("Recovered BER :", audio_ber)
print("Runtime       :", round(pipeline_runtime, 2), "seconds")

# ==========================================================
# Reconstruct Audio
# ==========================================================

encrypted_stream = merge_audio_blocks(encrypted_blocks)
encrypted_stream = remove_padding(encrypted_stream, pad_bits)

recovered_stream = merge_audio_blocks(recovered_blocks)
recovered_stream = remove_padding(recovered_stream, pad_bits)

encrypted_audio = bits_to_audio(encrypted_stream)
recovered_audio = bits_to_audio(recovered_stream)

# ==========================================================
# Save Audio
# ==========================================================

write_wav(
    ENCRYPTED_AUDIO,
    encrypted_audio,
    sample_rate
)

write_wav(
    RECOVERED_AUDIO,
    recovered_audio,
    sample_rate
)

print("\nAudio Saved")
print("Encrypted :", ENCRYPTED_AUDIO)
print("Recovered :", RECOVERED_AUDIO)

# ==========================================================
# Time Axis
# ==========================================================

time_axis = np.arange(len(samples)) / sample_rate

time_enc = np.arange(len(encrypted_audio)) / sample_rate

time_rec = np.arange(len(recovered_audio)) / sample_rate
# ==========================================================
# LAB 11E-2B
# IEEE Visualization (Waveform / Spectrogram / Histogram /
# Correlation)
# ==========================================================

from scipy.signal import spectrogram

# ----------------------------------------------------------
# Helper: Save Figure
# ----------------------------------------------------------

def save_ieee_figure(fig, filename):
    """
    Save IEEE-quality figure (600 DPI).
    """
    fig.tight_layout()
    fig.savefig(
        os.path.join(FIG_DIR, filename),
        dpi=600,
        bbox_inches="tight",
        facecolor="white"
    )
    plt.close(fig)


# ----------------------------------------------------------
# Waveform Plot
# ----------------------------------------------------------

def plot_waveform(signal,
                  sr,
                  title,
                  filename,
                  color="#1f77b4",
                  seconds=3):
    """
    Plot waveform (first few seconds).
    """

    samples = min(len(signal), sr * seconds)

    t = np.arange(samples) / sr

    fig = plt.figure(figsize=(6.4, 2.2))

    plt.plot(
        t,
        signal[:samples],
        linewidth=0.6,
        color=color
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")

    plt.title(title, fontsize=11, fontweight="normal")

    plt.grid(alpha=0.3)

    save_ieee_figure(fig, filename)


# ----------------------------------------------------------
# Spectrogram Plot
# ----------------------------------------------------------

def plot_spectrogram(signal,
                     sr,
                     title,
                     filename):
    """
    Generate speech spectrogram.
    """

    f, t, S = spectrogram(
        signal.astype(np.float32),
        fs=sr,
        window="hann",
        nperseg=512,
        noverlap=384,
        scaling="density",
        mode="magnitude"
    )

    S_db = 20 * np.log10(S + 1e-12)

    fig = plt.figure(figsize=(6.4, 3.2))

    plt.pcolormesh(
        t,
        f / 1000,
        S_db,
        shading="gouraud",
        cmap="jet"
    )

    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (kHz)")

    plt.title(title, fontsize=11, fontweight="normal")

    cbar = plt.colorbar()
    cbar.set_label("Magnitude (dB)")

    save_ieee_figure(fig, filename)


# ----------------------------------------------------------
# Histogram Comparison
# ----------------------------------------------------------

def plot_histogram(original,
                   encrypted,
                   filename):
    """
    Histogram comparison between original/encrypted audio.
    """

    fig = plt.figure(figsize=(6.4, 3.2))

    plt.hist(
        original,
        bins=256,
        density=True,
        alpha=0.55,
        label="Original Audio"
    )

    plt.hist(
        encrypted,
        bins=256,
        density=True,
        alpha=0.55,
        label="Encrypted Audio"
    )

    plt.xlabel("Amplitude")
    plt.ylabel("Probability Density")

    plt.title(
        "Amplitude Distribution Comparison",
        fontsize=11,
        fontweight="normal"
    )

    plt.legend(frameon=False)

    plt.grid(alpha=0.3)

    save_ieee_figure(fig, filename)


# ----------------------------------------------------------
# Correlation Comparison
# ----------------------------------------------------------

def plot_correlation(original,
                     encrypted,
                     filename):
    """
    Scatter plot of adjacent samples.
    """

    N = min(5000, len(original) - 1)

    fig = plt.figure(figsize=(6.4, 3.2))

    plt.subplot(1, 2, 1)

    plt.scatter(
        original[:N],
        original[1:N+1],
        s=2,
        alpha=0.35
    )

    plt.title("Original", fontsize=10, fontweight="normal")
    plt.xlabel("x(n)")
    plt.ylabel("x(n+1)")
    plt.grid(alpha=0.2)

    plt.subplot(1, 2, 2)

    plt.scatter(
        encrypted[:N],
        encrypted[1:N+1],
        s=2,
        alpha=0.35,
        color="tab:red"
    )

    plt.title("Encrypted", fontsize=10, fontweight="normal")
    plt.xlabel("x(n)")
    plt.ylabel("x(n+1)")
    plt.grid(alpha=0.2)

    save_ieee_figure(fig, filename)


# ==========================================================
# Generate All Figures
# ==========================================================

print("\nGenerating IEEE Figures...")

figure_start = time.time()

# ---------------- Waveforms ----------------

plot_waveform(
    samples,
    sample_rate,
    "Original Audio Waveform",
    "waveform_original.png"
)

plot_waveform(
    encrypted_audio,
    sample_rate,
    "Encrypted Audio Waveform",
    "waveform_encrypted.png",
    color="tab:red"
)

plot_waveform(
    recovered_audio,
    sample_rate,
    "Recovered Audio Waveform",
    "waveform_recovered.png",
    color="tab:green"
)

# ---------------- Spectrograms ----------------

plot_spectrogram(
    samples,
    sample_rate,
    "Original Audio Spectrogram",
    "spectrogram_original.png"
)

plot_spectrogram(
    encrypted_audio,
    sample_rate,
    "Encrypted Audio Spectrogram",
    "spectrogram_encrypted.png"
)

plot_spectrogram(
    recovered_audio,
    sample_rate,
    "Recovered Audio Spectrogram",
    "spectrogram_recovered.png"
)

# ---------------- Histogram ----------------

plot_histogram(
    samples,
    encrypted_audio,
    "histogram_comparison.png"
)

# ---------------- Correlation ----------------

plot_correlation(
    samples,
    encrypted_audio,
    "correlation_comparison.png"
)

figure_runtime = time.time() - figure_start

print("Figures Generated :", round(figure_runtime, 2), "seconds")
print("Saved to          :", FIG_DIR)
# ==========================================================
# LAB 11E-2C
# Audio Metrics / CSV / LOG
# ==========================================================

print("\nComputing Audio Security Metrics...")

metrics_start = time.time()

# ----------------------------------------------------------
# Metrics
# ----------------------------------------------------------

metrics = evaluate_audio(
    reference_audio=samples,
    encrypted_audio=encrypted_audio,
    recovered_audio=recovered_audio,
    reference_bits=audio_bits,
    recovered_bits=recovered_stream,
)

metrics_runtime = time.time() - metrics_start

# ----------------------------------------------------------
# Metrics Table
# ----------------------------------------------------------

metrics_df = pd.DataFrame(
    [
        {
            "Signal": "Original",
            "Entropy": metrics["OriginalEntropy"],
            "Correlation": metrics["OriginalCorrelation"],
            "MSE": np.nan,
            "PSNR(dB)": np.nan,
            "SNR(dB)": np.nan,
        },
        {
            "Signal": "Encrypted",
            "Entropy": metrics["EncryptedEntropy"],
            "Correlation": metrics["EncryptedCorrelation"],
            "MSE": np.nan,
            "PSNR(dB)": np.nan,
            "SNR(dB)": np.nan,
        },
        {
            "Signal": "Recovered",
            "Entropy": metrics["RecoveredEntropy"],
            "Correlation": metrics["RecoveredCorrelation"],
            "MSE": metrics["MSE"],
            "PSNR(dB)": metrics["PSNR"],
            "SNR(dB)": metrics["SNR"],
        },
    ]
)

CSV_FILE = os.path.join(CSV_DIR, "audio_metrics.csv")

metrics_df.to_csv(CSV_FILE, index=False)

# ----------------------------------------------------------
# Console Report
# ----------------------------------------------------------

print("\n================ AUDIO METRICS ================\n")

print(f"Original Entropy     : {metrics['OriginalEntropy']:.4f}")
print(f"Encrypted Entropy    : {metrics['EncryptedEntropy']:.4f}")
print(f"Recovered Entropy    : {metrics['RecoveredEntropy']:.4f}")

print()

print(f"Original Correlation : {metrics['OriginalCorrelation']:.4f}")
print(f"Encrypted Correlation: {metrics['EncryptedCorrelation']:.4f}")
print(f"Recovered Correlation: {metrics['RecoveredCorrelation']:.4f}")

print()

print(f"MSE                 : {metrics['MSE']:.6f}")
print(f"PSNR (dB)           : {metrics['PSNR']}")
print(f"SNR  (dB)           : {metrics['SNR']}")

print()

print(f"Recovered BER        : {metrics['BER']:.8f}")

# ----------------------------------------------------------
# Experiment Summary CSV
# ----------------------------------------------------------

summary_df = pd.DataFrame([{
    "SampleRate": sample_rate,
    "Samples": len(samples),
    "AudioBits": len(audio_bits),
    "Blocks": NUM_BLOCKS,
    "Frames": NUM_FRAMES,
    "EbN0(dB)": EBNO_DB,
    "RecoveredBER": metrics["BER"],
    "PipelineRuntime(s)": round(pipeline_runtime, 3),
    "FigureRuntime(s)": round(figure_runtime, 3),
    "MetricsRuntime(s)": round(metrics_runtime, 3),
}])

summary_df.to_csv(
    os.path.join(CSV_DIR, "audio_summary.csv"),
    index=False,
    float_format="%.6f"
)

# ----------------------------------------------------------
# LOG File
# ----------------------------------------------------------

LOG_FILE = os.path.join(LOG_DIR, "audio_visualization.txt")

with open(LOG_FILE, "w") as f:

    f.write("Audio Visualization Experiment (DATN v1.0)\n")
    f.write("=" * 55 + "\n\n")

    f.write("Input Audio\n")
    f.write(f"File        : {INPUT_AUDIO}\n")
    f.write(f"Sample Rate : {sample_rate} Hz\n")
    f.write(f"Samples     : {len(samples)}\n")
    f.write(f"Audio Bits  : {len(audio_bits)}\n")
    f.write(f"Blocks      : {NUM_BLOCKS}\n")
    f.write(f"Frames      : {NUM_FRAMES}\n")
    f.write(f"Eb/N0       : {EBNO_DB} dB\n\n")

    f.write("Security Metrics\n")
    f.write(f"Original Entropy      : {metrics['OriginalEntropy']:.6f}\n")
    f.write(f"Encrypted Entropy     : {metrics['EncryptedEntropy']:.6f}\n")
    f.write(f"Recovered Entropy     : {metrics['RecoveredEntropy']:.6f}\n\n")

    f.write(f"Original Correlation  : {metrics['OriginalCorrelation']:.6f}\n")
    f.write(f"Encrypted Correlation : {metrics['EncryptedCorrelation']:.6f}\n")
    f.write(f"Recovered Correlation : {metrics['RecoveredCorrelation']:.6f}\n\n")

    f.write(f"MSE                  : {metrics['MSE']:.10f}\n")
    f.write(f"PSNR (dB)            : {metrics['PSNR']}\n")
    f.write(f"SNR (dB)             : {metrics['SNR']}\n")
    f.write(f"Recovered BER         : {metrics['BER']:.10f}\n\n")

    f.write("Runtime\n")
    f.write(f"Pipeline Runtime : {pipeline_runtime:.2f} s\n")
    f.write(f"Figure Runtime   : {figure_runtime:.2f} s\n")
    f.write(f"Metrics Runtime  : {metrics_runtime:.2f} s\n")
    f.write(
        f"Total Runtime    : "
        f"{pipeline_runtime + figure_runtime + metrics_runtime:.2f} s\n"
    )

# ----------------------------------------------------------
# Final Report
# ----------------------------------------------------------

print("\nResults Saved Successfully\n")

print("Audio Files")
print("  ", ENCRYPTED_AUDIO)
print("  ", RECOVERED_AUDIO)

print("\nFigures")
for name in sorted(os.listdir(FIG_DIR)):
    print("  ", name)

print("\nCSV")
print("  ", CSV_FILE)
print("  ", os.path.join(CSV_DIR, "audio_summary.csv"))

print("\nLOG")
print("  ", LOG_FILE)

print("\n============================================================")
print("LAB 11E COMPLETED SUCCESSFULLY")
print("============================================================")