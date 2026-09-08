"""
tests/test_audio_io.py
============================================================
Unit Test : Audio I/O
============================================================
"""

import numpy as np

from audio.audio_io import (
    audio_to_bits,
    bits_to_audio,
)

print("=" * 60)
print("AUDIO IO TEST")
print("=" * 60)

SAMPLE_RATE = 16000

duration = 2

samples = np.random.randint(
    -32768,
    32767,
    SAMPLE_RATE * duration,
    dtype=np.int16,
)

bits = audio_to_bits(samples)

recover = bits_to_audio(bits)

assert np.array_equal(samples, recover)

print("[PASS] PCM -> Bitstream -> PCM")

assert bits.dtype == np.uint8

print("[PASS] Bitstream dtype")

assert len(bits) == len(samples) * 16

print("[PASS] Bitstream Length =", len(bits))

print("\nALL AUDIO IO TESTS PASSED")