"""
audio/bitstream.py

PCM16 ↔ Bitstream
Chia thành các block 256 bit cho Polar Code.
"""

import numpy as np
from config.config import POLAR_K


def pcm_to_bits(frame: np.ndarray) -> np.ndarray:
    frame_u16 = frame.astype(np.uint16)

    bits = ((frame_u16[:, None] >> np.arange(15, -1, -1)) & 1)

    return bits.astype(np.uint8).reshape(-1)


def bits_to_pcm(bits: np.ndarray) -> np.ndarray:

    bits = bits.reshape(-1, 16).astype(np.uint16)

    values = np.zeros(bits.shape[0], dtype=np.uint16)

    for i in range(16):
        values |= bits[:, i] << (15 - i)

    return values.view(np.int16)


def split_bit_blocks(bits: np.ndarray):

    return bits.reshape(-1, POLAR_K)


def merge_bit_blocks(blocks: np.ndarray):

    return blocks.reshape(-1)


if __name__ == "__main__":

    from audio.reader import read_audio
    from audio.framing import split_frames
    from config.config import INPUT_AUDIO

    audio, sr = read_audio(INPUT_AUDIO)

    frames = split_frames(audio)

    frame = frames[100]

    bits = pcm_to_bits(frame)

    blocks = split_bit_blocks(bits)

    recovered_bits = merge_bit_blocks(blocks)

    recovered_frame = bits_to_pcm(recovered_bits)

    print("========== BITSTREAM TEST ==========")
    print("Frame shape :", frame.shape)
    print("Bits shape  :", bits.shape)
    print("Blocks shape:", blocks.shape)

    print("\nBlock 1 first 32 bits:")
    print(blocks[0][:32])

    print("\nRecovered OK:", np.array_equal(frame, recovered_frame))