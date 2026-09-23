"""
compression_experiment/test_aac.py

AAC + Crypto-Coding + Polar Code
"""

from pathlib import Path
import numpy as np

from compression_experiment.codec_aac import encode_file, decode_file
from audio.audio_crypto_pipeline import secure_audio_transmission

OUT = Path("codec_results/aac")
OUT.mkdir(parents=True, exist_ok=True)

INPUT_WAV = "data/voice.wav"

AAC_FILE = OUT / "voice.m4a"
AAC_RECOVER = OUT / "voice_recovered.m4a"

# --------------------------------------------------
# Encode AAC
# --------------------------------------------------
encode_file(INPUT_WAV, AAC_FILE)

# Decode de kiem tra codec
decode_file(AAC_FILE, OUT / "voice_decoded.wav")

# --------------------------------------------------
# Dua payload AAC vao Crypto Pipeline
# --------------------------------------------------
payload = AAC_FILE.read_bytes()

bits = np.unpackbits(
    np.frombuffer(payload, dtype=np.uint8)
)

recover_bits, llr, stats = secure_audio_transmission(
    bits,
    ebno_db=20
)

recover_payload = np.packbits(recover_bits).tobytes()

AAC_RECOVER.write_bytes(recover_payload)

# Decode payload sau Crypto
decode_file(AAC_RECOVER, OUT / "voice_recovered.wav")

print("=" * 60)
print("AAC TEST FINISHED")
print("=" * 60)
print(f"AAC size      : {len(payload)} bytes")
print(f"Recovered size: {len(recover_payload)} bytes")
print(stats)