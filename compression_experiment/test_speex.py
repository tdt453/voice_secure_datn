"""
compression_experiment/test_speex.py
Speex + Crypto-Coding + Polar Code
"""

from pathlib import Path
import subprocess
import numpy as np

from audio.audio_crypto_pipeline import secure_audio_transmission

OUT = Path("codec_results/speex")
OUT.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------
# Encode Speex
# --------------------------------------------------
subprocess.run([
    "ffmpeg","-y",
    "-i","data/voice.wav",
    "-acodec","libspeex",
    "-ar","16000",
    "-ac","1",
    "-b:a","16k",
    OUT/"voice.spx"
], check=True)

# Decode de kiem tra
subprocess.run([
    "ffmpeg","-y",
    "-i",OUT/"voice.spx",
    OUT/"voice_decoded.wav"
], check=True)

# --------------------------------------------------
# Crypto Pipeline
# --------------------------------------------------
payload = open(OUT/"voice.spx","rb").read()

bits = np.unpackbits(
    np.frombuffer(payload,dtype=np.uint8)
)

recover_bits,llr,stats = secure_audio_transmission(bits,20)

recover_payload = np.packbits(recover_bits).tobytes()

with open(OUT/"voice_recovered.spx","wb") as f:
    f.write(recover_payload)

subprocess.run([
    "ffmpeg","-y",
    "-i",OUT/"voice_recovered.spx",
    OUT/"voice_recovered.wav"
], check=True)

print("="*60)
print("SPEEX TEST FINISHED")
print("="*60)
print(stats)