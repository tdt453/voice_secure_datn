"""
audio/framing.py

Chia tín hiệu PCM thành các frame 20 ms.
"""

import numpy as np

from config.config import FRAME_SIZE
from audio.reader import read_audio


def split_frames(audio: np.ndarray):
    """
    Chia audio thành các frame cố định.
    Frame cuối được zero-padding nếu thiếu mẫu.
    """

    frames = []

    total_samples = len(audio)

    for start in range(0, total_samples, FRAME_SIZE):

        frame = audio[start:start + FRAME_SIZE]

        # Zero padding frame cuối
        if len(frame) < FRAME_SIZE:
            frame = np.pad(
                frame,
                (0, FRAME_SIZE - len(frame)),
                mode="constant"
            )

        frames.append(frame)

    return np.asarray(frames, dtype=np.int16)


def frame_info(frames):
    print("========== FRAME INFORMATION ==========")
    print(f"Number of frames : {len(frames)}")
    print(f"Samples/frame    : {frames.shape[1]}")
    print(f"Frame shape      : {frames.shape}")
    print("=======================================")


if __name__ == "__main__":

    from config.config import INPUT_AUDIO

    audio, sr = read_audio(INPUT_AUDIO)

    frames = split_frames(audio)

    frame_info(frames)

    # Chỉ chạy khi test trực tiếp module này
    print("\nFirst 10 samples of Frame 1:")
    print(frames[0][:10])

    print("\nFirst 10 samples of Last Frame:")
    print(frames[-1][:10])

    import numpy as np
    energy = np.mean(frames.astype(np.float32) ** 2, axis=1)

    print("\nFrame 1 energy     :", energy[0])
    print("Frame 2 energy     :", energy[1])
    print("Middle frame energy:", energy[len(energy)//2])
    print("Last frame energy  :", energy[-1])