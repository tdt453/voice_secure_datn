"""
experiments/kpa_attack.py

Known-Plaintext Attack (KPA)
DATN - Raspberry Pi TX

Danh gia kha nang khang tan cong KPA cua he thong
SNOW-V + Crypto-Coding + Polar Code.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

ORIGINAL_WAV = "data/voice.wav"
ENCRYPTED_WAV = "output/encrypted_audio.wav"

# ==========================================================
# Doc audio
# ==========================================================
orig, sr = sf.read(ORIGINAL_WAV, dtype="int16")
cipher, _ = sf.read(ENCRYPTED_WAV, dtype="int16")

orig = orig.astype(np.int16)
cipher = cipher.astype(np.int16)

N = min(len(orig), len(cipher))
orig = orig[:N]
cipher = cipher[:N]

# ==========================================================
# Metrics
# ==========================================================
corr = np.corrcoef(
    orig.astype(np.float64),
    cipher.astype(np.float64)
)[0, 1]

mse = np.mean(
    (orig.astype(np.float64) - cipher.astype(np.float64)) ** 2
)

signal_power = np.mean(orig.astype(np.float64) ** 2)

snr_db = 10 * np.log10(signal_power / mse)

# ==========================================================
# Pearson theo tung cua so (512 samples)
# ==========================================================
WINDOW = 512

corr_window = []

for i in range(0, N - WINDOW + 1, WINDOW):

    x = orig[i:i + WINDOW].astype(np.float64)
    y = cipher[i:i + WINDOW].astype(np.float64)

    c = np.corrcoef(x, y)[0, 1]

    if np.isnan(c):
        c = 0

    corr_window.append(c)

corr_window = np.asarray(corr_window)

# ==========================================================
# CSV
# ==========================================================
with open(
    RESULT_DIR / "kpa_metrics.csv",
    "w",
    newline=""
) as f:

    writer = csv.writer(f)

    writer.writerow(["Metric", "Value"])

    writer.writerow(["Samples", N])
    writer.writerow(["Duration_sec", N / sr])
    writer.writerow(["Correlation", corr])
    writer.writerow(["MSE", mse])
    writer.writerow(["SNR_dB", snr_db])
    writer.writerow(["Mean_Window_Correlation", corr_window.mean()])
    writer.writerow(["Std_Window_Correlation", corr_window.std()])
    writer.writerow(["Max_Window_Correlation", corr_window.max()])
    writer.writerow(["Min_Window_Correlation", corr_window.min()])

# ==========================================================
# Scatter Plot
# ==========================================================
np.random.seed(42)

idx = np.random.choice(
    N,
    min(6000, N),
    replace=False
)

plt.figure(figsize=(6, 6))

plt.scatter(
    orig[idx],
    cipher[idx],
    s=2,
    alpha=0.25
)

plt.xlabel("Original PCM Sample")
plt.ylabel("Ciphertext PCM Sample")

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "kpa_scatter.png",
    dpi=600
)

plt.close()

# ==========================================================
# Correlation theo Frame
# ==========================================================
plt.figure(figsize=(8, 3.5))

plt.plot(
    corr_window,
    linewidth=1
)

plt.xlabel("Audio Window Index (512 samples)")
plt.ylabel("Pearson Correlation")

plt.ylim([-1, 1])

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "kpa_window_correlation.png",
    dpi=600
)

plt.close()

# ==========================================================
# Histogram so sanh
# ==========================================================
plt.figure(figsize=(8, 4))

plt.hist(
    orig,
    bins=256,
    density=True,
    alpha=0.45,
    label="Original"
)

plt.hist(
    cipher,
    bins=256,
    density=True,
    alpha=0.45,
    label="Ciphertext"
)

plt.xlabel("PCM Sample Value")
plt.ylabel("Probability Density")

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "kpa_histogram.png",
    dpi=600
)

plt.close()

# ==========================================================
# Power Spectrum Comparison
# ==========================================================
fft_orig = np.abs(np.fft.rfft(orig.astype(np.float64))) ** 2
fft_cipher = np.abs(np.fft.rfft(cipher.astype(np.float64))) ** 2

freq = np.fft.rfftfreq(N, d=1 / sr)

plt.figure(figsize=(8, 4))

plt.semilogy(
    freq,
    fft_orig,
    linewidth=1,
    label="Original"
)

plt.semilogy(
    freq,
    fft_cipher,
    linewidth=1,
    label="Ciphertext"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectrum")

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "kpa_power_spectrum.png",
    dpi=600
)

plt.close()

# ==========================================================
# Console Summary
# ==========================================================
print("=" * 60)
print("KNOWN PLAINTEXT ATTACK (KPA)")
print("=" * 60)

print(f"Samples                 : {N}")
print(f"Duration_sec            : {N/sr:.3f}")
print(f"Correlation             : {corr:.8f}")
print(f"MSE                     : {mse:.4f}")
print(f"SNR_dB                  : {snr_db:.4f}")
print(f"Mean Window Correlation : {corr_window.mean():.8f}")
print(f"Std Window Correlation  : {corr_window.std():.8f}")
print(f"Max Window Correlation  : {corr_window.max():.8f}")
print(f"Min Window Correlation  : {corr_window.min():.8f}")

print("\nSaved:")
print(" results/kpa_metrics.csv")
print(" results/kpa_scatter.png")
print(" results/kpa_window_correlation.png")
print(" results/kpa_histogram.png")
print(" results/kpa_power_spectrum.png")
print("=" * 60)