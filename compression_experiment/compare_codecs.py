"""
compression_experiment/compare_codecs.py

Comparison between:
1. PCM (No Compression)
2. Opus
3. AAC-LC
4. Speex

Output:
- codec_comparison_metrics.csv
- 10 figures (600 DPI)
"""

from pathlib import Path
import hashlib

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import soundfile as sf
from scipy.signal import stft

try:
    from pesq import pesq
except:
    pesq = None

try:
    from pystoi import stoi
except:
    stoi = None


# ==========================================================
# PATH
# ==========================================================

ROOT = Path("codec_results")
OUT = ROOT / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

PCM_FILE = Path("data/voice.wav")

FILES = {
    "PCM": PCM_FILE,
    "Opus": ROOT / "opus" / "opus_recovered.wav",
    "AAC": ROOT / "aac" / "voice_recovered.wav",
    "Speex": ROOT / "speex" / "voice_recovered.wav",
}

PAYLOAD = {
    "PCM": Path("data/voice.wav"),
    "Opus": ROOT / "opus" / "opus_payload.bin",
    "AAC": ROOT / "aac" / "voice.m4a",
    "Speex": ROOT / "speex" / "voice.spx",
}

BITRATE = {
    "PCM": 256,
    "Opus": 24,
    "AAC": 32,
    "Speex": 16,
}

SR = 16000

# ==========================================================
# METRICS
# ==========================================================

def mse(a, b):
    return np.mean((a.astype(np.float64)-b.astype(np.float64))**2)


def psnr(a, b):

    err = mse(a, b)

    if err == 0:
        return np.inf

    peak = 32767.0

    return 20*np.log10(peak/np.sqrt(err))


def snr(a, b):

    noise = a.astype(np.float64)-b.astype(np.float64)

    p_signal = np.mean(a.astype(np.float64)**2)
    p_noise = np.mean(noise**2)

    if p_noise == 0:
        return np.inf

    return 10*np.log10(p_signal/p_noise)


def ber(a, b):

    ba = np.unpackbits(a.view(np.uint8))
    bb = np.unpackbits(b.view(np.uint8))

    return np.mean(ba != bb)


def corr(a, b):
    return np.corrcoef(a.astype(float), b.astype(float))[0,1]


# ==========================================================
# LOAD ORIGINAL
# ==========================================================

orig, _ = sf.read(PCM_FILE, dtype="int16")
duration = len(orig)/SR
orig_size = PAYLOAD["PCM"].stat().st_size

results = []

# ==========================================================
# LOOP
# ==========================================================

for codec, wav in FILES.items():

    rec, _ = sf.read(wav, dtype="int16")

    rec = rec[:len(orig)]

    payload_size = PAYLOAD[codec].stat().st_size

    compression_ratio = orig_size/payload_size

    item = {
        "Codec": codec,
        "Payload_Bytes": payload_size,
        "Compression_Ratio": compression_ratio,
        "Bitrate_kbps": BITRATE[codec],
        "BER": ber(orig, rec),
        "MSE": mse(orig, rec),
        "PSNR_dB": psnr(orig, rec),
        "SNR_dB": snr(orig, rec),
        "Correlation": corr(orig, rec),
        "MD5": hashlib.md5(rec.tobytes()).hexdigest()
    }

    if pesq:
        item["PESQ"] = pesq(SR, orig.astype(np.float32), rec.astype(np.float32), "nb")
    else:
        item["PESQ"] = np.nan

    if stoi:
        item["STOI"] = stoi(orig.astype(np.float32), rec.astype(np.float32), SR)
    else:
        item["STOI"] = np.nan

    results.append(item)

df = pd.DataFrame(results)
df.to_csv(OUT/"codec_comparison_metrics.csv", index=False)

# ==========================================================
# BAR CHART FUNCTION
# ==========================================================

def bar(metric, ylabel, filename):

    plt.figure(figsize=(6,4))

    plt.bar(df["Codec"], df[metric])

    plt.ylabel(ylabel)

    plt.grid(True, axis="y")

    plt.tight_layout()

    plt.savefig(OUT/filename, dpi=600)

    plt.close()


bar("Compression_Ratio", "Compression Ratio", "compression_ratio.png")
bar("Bitrate_kbps", "Bitrate (kbps)", "bitrate.png")
bar("PESQ", "PESQ", "pesq.png")
bar("STOI", "STOI", "stoi.png")
bar("PSNR_dB", "PSNR (dB)", "psnr.png")
bar("SNR_dB", "SNR (dB)", "snr.png")
bar("BER", "BER", "ber.png")
bar("Payload_Bytes", "Payload Size (Bytes)", "payload_size.png")

# ==========================================================
# WAVEFORM COMPARISON
# ==========================================================

time_axis = np.arange(len(orig))/SR

for codec, wav in FILES.items():

    rec, _ = sf.read(wav, dtype="int16")
    rec = rec[:len(orig)]

    plt.figure(figsize=(10,3))
    plt.plot(time_axis, rec, linewidth=0.5)
    plt.title(f"{codec} Waveform")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT/f"{codec.lower()}_waveform.png", dpi=600)
    plt.close()

# ==========================================================
# SPECTROGRAM
# ==========================================================

for codec, wav in FILES.items():

    rec, _ = sf.read(wav, dtype="int16")
    rec = rec.astype(np.float32)

    f, t, Z = stft(rec, SR, nperseg=512)

    plt.figure(figsize=(8,4))

    plt.pcolormesh(
        t,
        f,
        20*np.log10(np.abs(Z)+1e-10),
        shading="gouraud"
    )

    plt.title(f"{codec} Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.colorbar(label="Magnitude (dB)")
    plt.tight_layout()
    plt.savefig(OUT/f"{codec.lower()}_spectrogram.png", dpi=600)
    plt.close()

# ==========================================================
# PRINT
# ==========================================================

print("="*70)
print("CODEC COMPARISON FINISHED")
print("="*70)

print(df[[
    "Codec",
    "Compression_Ratio",
    "Bitrate_kbps",
    "PESQ",
    "STOI",
    "PSNR_dB",
    "BER"
]])

print("="*70)
print("CSV  :", OUT/"codec_comparison_metrics.csv")
print("FIGS :", OUT)