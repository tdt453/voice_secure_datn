"""
tests/test_audio_metrics.py
============================================================
Audio Metrics Unit Test (DATN v1.1)
============================================================
"""

import numpy as np

from metrics.audio_metrics import (
    shannon_entropy,
    adjacent_correlation,
    mse,
    psnr,
    snr,
    bit_error_rate,
)

print("=" * 60)
print("AUDIO METRICS TEST")
print("=" * 60)

np.random.seed(0)

original = np.random.randint(
    -32768, 32767, 8000, dtype=np.int16
)

recovered = original.copy()

encrypted = np.random.randint(
    -32768, 32767, 8000, dtype=np.int16
)

# ----------------------------------------------------------
# Entropy
# ----------------------------------------------------------

ent_original = shannon_entropy(original)
ent_encrypted = shannon_entropy(encrypted)

print(f"Original Entropy  : {ent_original:.4f}")
print(f"Encrypted Entropy : {ent_encrypted:.4f}")

# Entropy phải nằm gần 8 bit/sample.
assert 7.8 <= ent_original <= 8.0
assert 7.8 <= ent_encrypted <= 8.0

print("[PASS] Entropy")

# ----------------------------------------------------------
# Correlation
# ----------------------------------------------------------

corr = adjacent_correlation(encrypted)

print(f"Encrypted Correlation : {corr:.4f}")

assert abs(corr) < 0.1

print("[PASS] Correlation")

# ----------------------------------------------------------
# MSE / PSNR / SNR
# ----------------------------------------------------------

assert mse(original, recovered) == 0
assert np.isinf(psnr(original, recovered))
assert np.isinf(snr(original, recovered))

print("[PASS] MSE")
print("[PASS] PSNR")
print("[PASS] SNR")

# ----------------------------------------------------------
# BER
# ----------------------------------------------------------

bits = np.random.randint(
    0, 2, 256, dtype=np.uint8
)

assert bit_error_rate(bits, bits) == 0

print("[PASS] BER")

print("\nALL AUDIO METRICS TESTS PASSED")