"""
experiments/audio_recovery_evaluation.py

Danh gia hieu nang khoi phuc tin hieu sau truyen thuc tren Raspberry Pi.

Sinh cac chi so:
- BER
- MSE
- SNR
- PSNR
- Correlation
- PESQ
- STOI

Sinh 9 hinh 600 DPI:
1. Waveform Original
2. Waveform Recovered
3. Waveform Error
4. Spectrogram Original
5. Spectrogram Recovered
6. FFT Original
7. FFT Recovered
8. PSD Original
9. PSD Recovered
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf
from pesq import pesq
from pystoi.stoi import stoi

# ==========================================================
# Paths
# ==========================================================

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

ORIGINAL = "data/voice.wav"
RECOVERED = "output/network_received.wav"

# ==========================================================
# Read Audio
# ==========================================================

orig, sr = sf.read(ORIGINAL, dtype="int16")
recv, sr2 = sf.read(RECOVERED, dtype="int16")

assert sr == sr2, "Sample rate mismatch."

N = min(len(orig), len(recv))

orig = orig[:N]
recv = recv[:N]

orig_f = orig.astype(np.float64)
recv_f = recv.astype(np.float64)

time_axis = np.arange(N) / sr

# ==========================================================
# Metrics
# ==========================================================

mse = np.mean((orig_f - recv_f) ** 2)

noise_power = np.mean((orig_f - recv_f) ** 2)
signal_power = np.mean(orig_f ** 2)

if noise_power == 0:
    snr = np.inf
    psnr = np.inf
else:
    snr = 10 * np.log10(signal_power / noise_power)
    psnr = 10 * np.log10((32767.0 ** 2) / mse)

ber = np.mean(
    np.unpackbits(orig.view(np.uint8))
    != np.unpackbits(recv.view(np.uint8))
)

corr = np.corrcoef(orig_f, recv_f)[0, 1]

pesq_score = pesq(sr, orig_f, recv_f, "wb")
stoi_score = stoi(orig_f, recv_f, sr, extended=False)

# ==========================================================
# Save Metrics CSV
# ==========================================================

metrics = pd.DataFrame({
    "Metric": [
        "Samples",
        "Duration_sec",
        "BER",
        "MSE",
        "PSNR_dB",
        "SNR_dB",
        "Correlation",
        "PESQ",
        "STOI"
    ],
    "Value": [
        N,
        N / sr,
        ber,
        mse,
        psnr,
        snr,
        corr,
        pesq_score,
        stoi_score
    ]
})

metrics.to_csv(
    RESULT_DIR / "audio_recovery_metrics.csv",
    index=False
)

# ==========================================================
# Figure 1 : Waveform Original
# ==========================================================

plt.figure(figsize=(12,3.5))

plt.plot(
    time_axis,
    orig_f,
    color="royalblue",
    linewidth=0.6
)

plt.title("Original Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "waveform_original.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# Figure 2 : Waveform Recovered
# ==========================================================

plt.figure(figsize=(12,3.5))

plt.plot(
    time_axis,
    recv_f,
    color="crimson",
    linewidth=0.6
)

plt.title("Recovered Audio Waveform")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "waveform_recovered.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# Figure 3 : Waveform Error
# ==========================================================

plt.figure(figsize=(12,3))

plt.plot(
    time_axis,
    orig_f - recv_f,
    color="black",
    linewidth=0.5
)

plt.title("Waveform Error (Original - Recovered)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude Error")
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "waveform_error.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# Figure 4 : Spectrogram Original
# ==========================================================

plt.figure(figsize=(10,4))

plt.specgram(
    orig_f,
    Fs=sr,
    NFFT=512,
    noverlap=256,
    cmap="viridis"
)

plt.title("Original Audio Spectrogram")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Power (dB)")

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "spectrogram_original.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# Figure 5 : Spectrogram Recovered
# ==========================================================

plt.figure(figsize=(10,4))

plt.specgram(
    recv_f,
    Fs=sr,
    NFFT=512,
    noverlap=256,
    cmap="viridis"
)

plt.title("Recovered Audio Spectrogram")
plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")
plt.colorbar(label="Power (dB)")

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "spectrogram_recovered.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# FFT
# ==========================================================

freq = np.fft.rfftfreq(N, d=1/sr)

fft_orig = np.abs(np.fft.rfft(orig_f))
fft_recv = np.abs(np.fft.rfft(recv_f))

# ==========================================================
# Figure 6 : FFT Original
# ==========================================================

plt.figure(figsize=(10,4))

plt.plot(
    freq,
    fft_orig,
    color="royalblue",
    linewidth=0.8
)

plt.title("Original Audio FFT Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, sr/2)
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "fft_original.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# Figure 7 : FFT Recovered
# ==========================================================

plt.figure(figsize=(10,4))

plt.plot(
    freq,
    fft_recv,
    color="crimson",
    linewidth=0.8
)

plt.title("Recovered Audio FFT Spectrum")
plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")
plt.xlim(0, sr/2)
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "fft_recovered.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# PSD
# ==========================================================

psd_orig = fft_orig ** 2
psd_recv = fft_recv ** 2

# ==========================================================
# Figure 8 : PSD Original
# ==========================================================

plt.figure(figsize=(10,4))

plt.semilogy(
    freq,
    psd_orig,
    color="royalblue",
    linewidth=0.8
)

plt.title("Original Audio Power Spectral Density")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.xlim(0, sr/2)
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "psd_original.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# Figure 9 : PSD Recovered
# ==========================================================

plt.figure(figsize=(10,4))

plt.semilogy(
    freq,
    psd_recv,
    color="crimson",
    linewidth=0.8
)

plt.title("Recovered Audio Power Spectral Density")
plt.xlabel("Frequency (Hz)")
plt.ylabel("PSD")
plt.xlim(0, sr/2)
plt.grid(alpha=0.3)

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "psd_recovered.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close()

# ==========================================================
# Console Summary
# ==========================================================

print("=" * 60)
print("AUDIO RECOVERY EVALUATION")
print("=" * 60)

for _, row in metrics.iterrows():

    value = row["Value"]

    if isinstance(value, float):
        print(f"{row['Metric']:18s}: {value:.8f}")
    else:
        print(f"{row['Metric']:18s}: {value}")

print("=" * 60)
print("Saved metrics : results/audio_recovery_metrics.csv")
print("Saved figures : results/")