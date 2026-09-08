"""
config.py
Cấu hình chung cho hệ thống Voice Secure.
"""

from pathlib import Path

# ==========================================================
# Project Path
# ==========================================================

ROOT_DIR = Path.home() / "voice_secure"

DATA_DIR = ROOT_DIR / "data"
OUTPUT_DIR = ROOT_DIR / "output"
LOG_DIR = ROOT_DIR / "logs"

INPUT_AUDIO = DATA_DIR / "voice.wav"
RECOVERED_AUDIO = OUTPUT_DIR / "recovered_voice.wav"

# ==========================================================
# Audio Parameters
# ==========================================================

SAMPLE_RATE = 16000
PCM_BITS = 16
CHANNELS = 1

FRAME_DURATION_MS = 20
FRAME_SIZE = SAMPLE_RATE * FRAME_DURATION_MS // 1000  # 320 samples
FRAME_BYTES = FRAME_SIZE * 2                          # 640 bytes
FRAME_BITS = FRAME_BYTES * 8                          # 5120 bits

# ==========================================================
# Polar Parameters
# ==========================================================

POLAR_N = 512
POLAR_K = 256
CODE_RATE = POLAR_K / POLAR_N

# ==========================================================
# AES Parameters
# ==========================================================

AES_KEY_SIZE = 16
NONCE_SIZE = 8

# ==========================================================
# UDP Parameters
# ==========================================================

TX_IP = "192.168.0.103"     # Pi4 hiện tại
RX_IP = "192.168.0.104"     # Pi3 sau này
UDP_PORT = 5005