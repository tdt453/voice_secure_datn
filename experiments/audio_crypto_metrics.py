"""
experiments/audio_crypto_metrics.py

Danh gia cac chi so mat ma cua audio da ma hoa.
"""

import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

ORIGINAL_WAV = "data/voice.wav"
ENCRYPTED_BIN = "output/encrypted_audio.bin"

orig, sr = sf.read(ORIGINAL_WAV, dtype="int16")
orig = orig.astype(np.int16)

if not Path(ENCRYPTED_BIN).exists():
    raise FileNotFoundError(
        "Khong tim thay output/encrypted_audio.bin.\n"
        "Can xuat ciphertext truoc khi tinh chi so mat ma."
    )

with open(ENCRYPTED_BIN, "rb") as f:
    enc_bytes = f.read()

enc = np.frombuffer(enc_bytes, dtype=np.int16)

N = min(len(orig), len(enc))
orig = orig[:N]
enc = enc[:N]


def entropy_pcm(x):
    hist, _ = np.histogram(
        x.astype(np.int32),
        bins=65536,
        range=(-32768, 32768),
    )
    p = hist / hist.sum()
    p = p[p > 0]
    return -(p * np.log2(p)).sum()


corr = np.corrcoef(
    orig.astype(np.float64),
    enc.astype(np.float64),
)[0, 1]

npcr = np.mean(orig != enc) * 100

uaci = (
    np.mean(np.abs(orig.astype(np.int32) - enc.astype(np.int32)))
    / 65535
    * 100
)

hist, _ = np.histogram(
    enc.astype(np.int32),
    bins=256,
    range=(-32768, 32768),
)
expected = hist.sum() / 256
chi_square = np.sum(((hist - expected) ** 2) / expected)

with open(
    RESULT_DIR / "security_metrics.csv",
    "w",
    newline="",
) as f:
    writer = csv.writer(f)
    writer.writerow(["Metric", "Value"])
    writer.writerow(["Entropy_Original", entropy_pcm(orig)])
    writer.writerow(["Entropy_Encrypted", entropy_pcm(enc)])
    writer.writerow(["Correlation", corr])
    writer.writerow(["NPCR_percent", npcr])
    writer.writerow(["UACI_percent", uaci])
    writer.writerow(["Chi_Square", chi_square])

idx = np.random.choice(len(orig), min(6000, len(orig)), replace=False)

plt.figure(figsize=(6, 6))
plt.scatter(orig[idx], enc[idx], s=2, alpha=0.25)
plt.xlabel("Original Sample")
plt.ylabel("Encrypted Sample")
plt.tight_layout()
plt.savefig(RESULT_DIR / "correlation_scatter.png", dpi=600)
plt.close()

plt.figure(figsize=(7, 4))
plt.hist(enc, bins=256, density=True)
plt.xlabel("Encrypted Sample Value")
plt.ylabel("Probability Density")
plt.tight_layout()
plt.savefig(RESULT_DIR / "encrypted_histogram.png", dpi=600)
plt.close()

print("=" * 60)
print("AUDIO CRYPTO METRICS")
print("=" * 60)
print(f"Entropy Original : {entropy_pcm(orig):.6f}")
print(f"Entropy Encrypted: {entropy_pcm(enc):.6f}")
print(f"Correlation       : {corr:.6f}")
print(f"NPCR (%)          : {npcr:.4f}")
print(f"UACI (%)          : {uaci:.4f}")
print(f"Chi-Square        : {chi_square:.4f}")
print("Saved -> results/security_metrics.csv")