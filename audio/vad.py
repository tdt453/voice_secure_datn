"""
audio/vad.py

Voice Activity Detection bằng năng lượng tín hiệu.
"""

import numpy as np

from audio.reader import read_audio
from audio.framing import split_frames
from config.config import INPUT_AUDIO


def frame_energy(frame):
    """Tính năng lượng RMS của một frame."""
    frame = frame.astype(np.float32)
    return np.mean(frame ** 2)


def detect_voice_frames(frames, threshold=5000):
    """
    Trả về:
        voiced_frames
        voiced_indices
        energies
    """
    energies = np.array([frame_energy(f) for f in frames])

    voiced_indices = np.where(energies > threshold)[0]

    voiced_frames = frames[voiced_indices]

    return voiced_frames, voiced_indices, energies


if __name__ == "__main__":

    audio, sr = read_audio(INPUT_AUDIO)

    frames = split_frames(audio)

    voiced_frames, voiced_indices, energies = detect_voice_frames(frames)

    print("========== VAD RESULT ==========")
    print("Total frames :", len(frames))
    print("Voice frames :", len(voiced_frames))
    print("Silence      :", len(frames) - len(voiced_frames))

    print("\nFirst 10 voiced frame indices:")
    print(voiced_indices[:10])

    print("\nFirst 10 frame energies:")
    print(np.round(energies[:10], 2))