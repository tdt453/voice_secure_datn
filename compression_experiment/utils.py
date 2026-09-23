"""
Utility cho cac codec.
"""

import soundfile as sf
import numpy as np

from compression_experiment.config import SAMPLE_RATE


def read_pcm(path):

    audio, sr = sf.read(path, dtype="int16")

    if sr != SAMPLE_RATE:
        raise ValueError("Sample rate phai la 16 kHz.")

    if audio.ndim > 1:
        audio = audio[:, 0]

    return audio.astype(np.int16)


def write_pcm(path, audio):

    sf.write(
        path,
        audio.astype(np.int16),
        SAMPLE_RATE,
        subtype="PCM_16"
    )


def split_frames(audio, frame_samples=320):

    frames = []

    for i in range(0, len(audio), frame_samples):

        frame = audio[i:i + frame_samples]

        if len(frame) < frame_samples:
            frame = np.pad(frame, (0, frame_samples - len(frame)))

        frames.append(frame.astype(np.int16))

    return frames


def merge_frames(frames, total_samples):

    audio = np.concatenate(frames)

    return audio[:total_samples]