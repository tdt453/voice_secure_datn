"""
Compression Experiment Configuration
Doc lap voi do an chinh.
"""

from pathlib import Path

# Audio
SAMPLE_RATE = 16000
CHANNELS = 1
FRAME_SAMPLES = 320          # 20 ms
FRAME_BYTES = FRAME_SAMPLES * 2

# Codec bitrate
OPUS_BITRATE = 24000
AAC_BITRATE = "32k"
SPEEX_BITRATE = 16000

ROOT = Path(".")
DATA_DIR = ROOT / "data"
RESULT_DIR = ROOT / "codec_results"

RESULT_DIR.mkdir(exist_ok=True)
(RESULT_DIR / "opus").mkdir(exist_ok=True)
(RESULT_DIR / "aac").mkdir(exist_ok=True)
(RESULT_DIR / "speex").mkdir(exist_ok=True)
(RESULT_DIR / "comparison").mkdir(exist_ok=True)