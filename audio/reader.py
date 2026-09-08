"""
reader.py
Đọc file WAV và kiểm tra định dạng.
"""

from pathlib import Path
import soundfile as sf

from config.config import SAMPLE_RATE


def read_audio(file_path):
    """
    Đọc file WAV dạng PCM16 mono 16 kHz.
    """

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(f"Không tìm thấy {file_path}")

    audio, sr = sf.read(file_path, dtype="int16")

    if sr != SAMPLE_RATE:
        raise ValueError(
            f"Sample rate phải là {SAMPLE_RATE} Hz, nhận được {sr} Hz."
        )

    # Stereo -> Mono
    if audio.ndim == 2:
        audio = audio[:, 0]

    return audio, sr


def audio_info(audio, sr):

    duration = len(audio) / sr

    return {
        "Sample Rate": sr,
        "Samples": len(audio),
        "Duration (s)": round(duration, 3),
        "PCM": "16-bit",
        "Channels": 1
    }


if __name__ == "__main__":

    from config.config import INPUT_AUDIO
    import numpy as np

    audio, sr = read_audio(INPUT_AUDIO)

    print("========== AUDIO INFO ==========")
    print(f"Sample Rate : {sr}")
    print(f"Samples     : {len(audio)}")
    print(f"Duration    : {len(audio)/sr:.3f} s")

    print("\n===== SIGNAL STATISTICS =====")
    print("Min value :", np.min(audio))
    print("Max value :", np.max(audio))
    print("Mean      :", round(float(np.mean(audio)), 2))
    print("Std       :", round(float(np.std(audio)), 2))

    print("\nFirst 20 samples:")
    print(audio[:20])

    print("\nMiddle 20 samples:")
    mid = len(audio)//2
    print(audio[mid:mid+20])

    print("\nLast 20 samples:")
    print(audio[-20:])