"""
experiments/audio_security_visualization.py

DATN - Audio Security & Recovery Visualization
Raspberry Pi RX

Sinh cac chi so:
- BER
- MSE
- PSNR
- SNR
- Correlation
- Shannon Entropy
- Packet Loss

Sinh cac hinh 600 DPI:
- waveform_comparison.png
- spectrogram_original.png
- spectrogram_recovered.png
- histogram_comparison.png
- correlation_scatter.png
- psd_comparison.png
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

# ==========================================================
# PATH
# ==========================================================

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

ORIGINAL_FILE = "data/voice.wav"
RECOVERED_FILE = "output/network_received.wav"
PACKET_LOG = "logs/packet_log.csv"


# ==========================================================
# Metrics
# ==========================================================

def mse(original, recovered):
    return np.mean(
        (original.astype(np.float64) -
         recovered.astype(np.float64)) ** 2
    )


def psnr(original, recovered):
    err = mse(original, recovered)

    if err == 0:
        return np.inf

    return 20 * np.log10(32767 / np.sqrt(err))


def snr(original, recovered):
    signal_power = np.mean(original.astype(np.float64) ** 2)

    noise_power = np.mean(
        (original.astype(np.float64) -
         recovered.astype(np.float64)) ** 2
    )

    if noise_power == 0:
        return np.inf

    return 10 * np.log10(signal_power / noise_power)


def ber_pcm(original, recovered):
    """
    BER tren tung bit cua PCM16.
    """

    original_bytes = original.astype(np.int16).view(np.uint8)
    recovered_bytes = recovered.astype(np.int16).view(np.uint8)

    bits_original = np.unpackbits(original_bytes)
    bits_recovered = np.unpackbits(recovered_bytes)

    return np.mean(bits_original != bits_recovered)


def correlation(original, recovered):
    """
    Bo qua silence frame de tranh giam correlation.
    """

    valid = (original != 0) | (recovered != 0)

    if np.sum(valid) == 0:
        return 0.0

    return np.corrcoef(
        original[valid],
        recovered[valid]
    )[0, 1]


def shannon_entropy(signal):
    """
    Shannon entropy PCM16.
    """

    values = signal.astype(np.int16).view(np.uint16)

    histogram = np.bincount(
        values,
        minlength=65536
    ).astype(np.float64)

    probability = histogram / histogram.sum()
    probability = probability[probability > 0]

    return -np.sum(probability * np.log2(probability))


# ==========================================================
# Load Audio
# ==========================================================

orig, sr = sf.read(ORIGINAL_FILE, dtype="int16")
recv, _ = sf.read(RECOVERED_FILE, dtype="int16")

orig = orig.astype(np.int16)
recv = recv.astype(np.int16)

length = min(len(orig), len(recv))

orig = orig[:length]
recv = recv[:length]

time_axis = np.arange(length) / sr

print("=" * 60)
print("AUDIO FILE INFORMATION")
print("=" * 60)
print("Original Samples :", len(orig))
print("Recovered Samples:", len(recv))
print("Duration (s)     :", round(len(orig) / sr, 3))
print()


# ==========================================================
# Waveform Comparison
# ==========================================================

plt.figure(figsize=(12, 4))

plt.plot(time_axis, orig,
         linewidth=0.5,
         label="Original")

plt.plot(time_axis, recv,
         linewidth=0.5,
         alpha=0.8,
         label="Recovered")

plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "waveform_comparison.png",
    dpi=600
)

plt.close()


# ==========================================================
# Spectrogram Original
# ==========================================================

plt.figure(figsize=(12, 4))

plt.specgram(
    orig,
    Fs=sr,
    NFFT=512,
    noverlap=256
)

plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "spectrogram_original.png",
    dpi=600
)

plt.close()


# ==========================================================
# Spectrogram Recovered
# ==========================================================

plt.figure(figsize=(12, 4))

plt.specgram(
    recv,
    Fs=sr,
    NFFT=512,
    noverlap=256
)

plt.xlabel("Time (s)")
plt.ylabel("Frequency (Hz)")

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "spectrogram_recovered.png",
    dpi=600
)

plt.close()


# ==========================================================
# Histogram
# ==========================================================

plt.figure(figsize=(12, 4))

plt.hist(
    orig,
    bins=256,
    alpha=0.6,
    density=True,
    label="Original"
)

plt.hist(
    recv,
    bins=256,
    alpha=0.6,
    density=True,
    label="Recovered"
)

plt.xlabel("PCM Amplitude")
plt.ylabel("Probability Density")
plt.legend()

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "histogram_comparison.png",
    dpi=600
)

plt.close()


# ==========================================================
# Correlation Scatter
# ==========================================================

sample_count = min(12000, length)

indices = np.random.choice(
    length,
    sample_count,
    replace=False
)

plt.figure(figsize=(6, 6))

plt.scatter(
    orig[indices],
    recv[indices],
    s=2
)

plt.xlabel("Original Sample")
plt.ylabel("Recovered Sample")

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "correlation_scatter.png",
    dpi=600
)

plt.close()


# ==========================================================
# Power Spectral Density
# ==========================================================

frequency = np.fft.rfftfreq(length, d=1 / sr)

psd_original = np.abs(
    np.fft.rfft(orig.astype(np.float32))
) ** 2

psd_recovered = np.abs(
    np.fft.rfft(recv.astype(np.float32))
) ** 2

plt.figure(figsize=(12, 4))

plt.semilogy(
    frequency,
    psd_original,
    label="Original"
)

plt.semilogy(
    frequency,
    psd_recovered,
    label="Recovered"
)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Power Spectral Density")

plt.legend()

plt.tight_layout()

plt.savefig(
    RESULT_DIR / "psd_comparison.png",
    dpi=600
)

plt.close()


# ==========================================================
# Packet Loss
# ==========================================================

packet_loss = 0.0

if Path(PACKET_LOG).exists():

    packet_log = pd.read_csv(PACKET_LOG)

    crc_ok = packet_log["CRC_OK"].astype(bool)

    packet_loss = 100 * (
        1 - crc_ok.sum() / len(packet_log)
    )


# ==========================================================
# Statistics
# ==========================================================

statistics = {
    "Samples": length,
    "Duration_sec": length / sr,
    "Packet_Loss_percent": packet_loss,
    "BER": ber_pcm(orig, recv),
    "MSE": mse(orig, recv),
    "PSNR_dB": psnr(orig, recv),
    "SNR_dB": snr(orig, recv),
    "Correlation": correlation(orig, recv),
    "Entropy_Original": shannon_entropy(orig),
    "Entropy_Recovered": shannon_entropy(recv),
}

pd.DataFrame([statistics]).to_csv(
    RESULT_DIR / "audio_security_statistics.csv",
    index=False
)


# ==========================================================
# Print Summary
# ==========================================================

print("=" * 60)
print("AUDIO SECURITY STATISTICS")
print("=" * 60)

for key, value in statistics.items():

    if isinstance(value, float):
        print(f"{key:22s}: {value:.8f}")
    else:
        print(f"{key:22s}: {value}")

print()
print("Figures saved in results/")
print("Statistics saved : results/audio_security_statistics.csv")
print("=" * 60)