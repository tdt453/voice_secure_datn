"""
experiments/export_encrypted_audio.py

Xuat ciphertext thanh encrypted_audio.bin va encrypted_audio.wav.
"""

from pathlib import Path

import numpy as np
import soundfile as sf

from audio.audio_crypto_pipeline import encrypt_audio_blocks

INPUT_WAV = "data/voice.wav"
OUTPUT_BIN = "output/encrypted_audio.bin"
OUTPUT_WAV = "output/encrypted_audio.wav"

Path("output").mkdir(exist_ok=True)

# =====================================================
# WAV -> PCM -> Bitstream
# =====================================================
pcm, sr = sf.read(INPUT_WAV, dtype="int16")

pcm = pcm.astype(np.int16)

pcm_bytes = pcm.tobytes()

audio_bits = np.unpackbits(
    np.frombuffer(pcm_bytes, dtype=np.uint8)
).astype(np.uint8)

print(f"Input samples : {len(pcm)}")
print(f"Input bits    : {len(audio_bits)}")

# =====================================================
# Encrypt only
# =====================================================
llr_blocks, pad_bits, stats = encrypt_audio_blocks(
    audio_bits,
    ebno_db=100
)

# Bit sau Polar + Crypto
cipher_bits = (llr_blocks < 0).astype(np.uint8).reshape(-1)

cipher_bits = cipher_bits[:len(audio_bits)]

cipher_bytes = np.packbits(cipher_bits)

cipher_bytes = cipher_bytes[:len(pcm_bytes)]

# BIN
with open(OUTPUT_BIN, "wb") as f:
    f.write(cipher_bytes)

# WAV de ve histogram / spectrogram
cipher_pcm = np.frombuffer(cipher_bytes, dtype=np.int16)

sf.write(
    OUTPUT_WAV,
    cipher_pcm,
    sr,
    subtype="PCM_16"
)

print("=" * 60)
print("ENCRYPTED AUDIO EXPORT")
print("=" * 60)
print(f"Blocks         : {stats['num_blocks']}")
print(f"Cipher samples : {len(cipher_pcm)}")
print(f"Saved BIN      : {OUTPUT_BIN}")
print(f"Saved WAV      : {OUTPUT_WAV}")