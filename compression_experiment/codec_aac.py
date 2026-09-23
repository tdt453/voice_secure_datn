"""
compression_experiment/codec_aac.py

AAC-LC Codec su dung FFmpeg.
Khong dung pydub (tuong thich Python 3.13).
"""

import subprocess
from pathlib import Path

from compression_experiment.config import AAC_BITRATE


def encode_file(input_wav, output_aac):
    """
    PCM WAV -> AAC-LC (.m4a)
    """
    input_wav = str(input_wav)
    output_aac = str(output_aac)

    Path(output_aac).parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", input_wav,
            "-c:a", "aac",
            "-profile:a", "aac_low",
            "-b:a", AAC_BITRATE,
            "-ar", "16000",
            "-ac", "1",
            output_aac,
        ],
        check=True,
    )


def decode_file(input_aac, output_wav):
    """
    AAC -> PCM WAV
    """
    input_aac = str(input_aac)
    output_wav = str(output_wav)

    Path(output_wav).parent.mkdir(parents=True, exist_ok=True)

    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-loglevel", "error",
            "-i", input_aac,
            "-ar", "16000",
            "-ac", "1",
            "-sample_fmt", "s16",
            output_wav,
        ],
        check=True,
    )