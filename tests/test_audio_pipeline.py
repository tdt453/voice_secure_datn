"""
tests/test_audio_pipeline.py
============================================================
Secure Audio Pipeline Test
============================================================
"""

import numpy as np

from audio.audio_crypto_pipeline import secure_audio_transmission

print("="*60)
print("SECURE AUDIO PIPELINE TEST")
print("="*60)

audio_bits = np.random.randint(
    0,
    2,
    320000,
    dtype=np.uint8
)

recover_bits, llr_blocks, stats = secure_audio_transmission(
    audio_bits,
    ebno_db=20
)

assert np.array_equal(audio_bits, recover_bits)

print("[PASS] Audio Bitstream")
print("[PASS] Crypto-Coding")
print("[PASS] Polar Encode / Decode")
print("[PASS] Audio Reconstruction")

print("Recovered BER :", stats["ber"])

assert stats["ber"] == 0

print("\nALL AUDIO PIPELINE TESTS PASSED")