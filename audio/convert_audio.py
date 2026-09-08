"""
audio/convert_audio.py
Chuyển MP3 -> WAV (16 kHz, Mono, PCM16) bằng FFmpeg.
"""

from pathlib import Path
import subprocess

from config.config import SAMPLE_RATE


def mp3_to_wav(mp3_path, wav_path):
    mp3_path = Path(mp3_path)
    wav_path = Path(wav_path)

    if not mp3_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {mp3_path}")

    cmd = [
        "ffmpeg",
        "-y",
        "-i", str(mp3_path),
        "-ac", "1",                     # Mono
        "-ar", str(SAMPLE_RATE),        # 16 kHz
        "-sample_fmt", "s16",           # PCM16
        str(wav_path)
    ]

    subprocess.run(cmd, check=True)

    print(f"Đã tạo: {wav_path}")


if __name__ == "__main__":
    from config.config import DATA_DIR

    mp3_file = DATA_DIR / "voice.mp3"
    wav_file = DATA_DIR / "voice.wav"

    mp3_to_wav(mp3_file, wav_file)