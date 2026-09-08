"""
audio/audio_blocks.py
============================================================
Audio Block Framing (DATN v1.0)

Target : Raspberry Pi 4 / Raspberry Pi 3

Block Size
----------
Polar Payload = 256 bits
Blocks / Frame = 20

Functions
---------
1. Split audio bitstream into 256-bit blocks.
2. Zero padding.
3. Merge blocks.
4. Remove padding.
5. Frame/Block indexing.
============================================================
"""

import numpy as np

from config.config import POLAR_K

BLOCK_SIZE = POLAR_K          # 256 bits
BLOCKS_PER_FRAME = 20


# ==========================================================
# Padding
# ==========================================================

def padding_length(bits):
    """
    Compute required zero padding.
    """

    remainder = len(bits) % BLOCK_SIZE

    if remainder == 0:
        return 0

    return BLOCK_SIZE - remainder


# ==========================================================
# Split Bitstream
# ==========================================================

def split_audio_blocks(bits):
    """
    Split bitstream into 256-bit blocks.

    Returns
    -------
    blocks : ndarray (num_blocks,256)
    pad_bits : int
    """

    bits = np.asarray(bits, dtype=np.uint8)

    pad_bits = padding_length(bits)

    if pad_bits > 0:
        bits = np.concatenate([
            bits,
            np.zeros(pad_bits, dtype=np.uint8)
        ])

    blocks = bits.reshape((-1, BLOCK_SIZE))

    return blocks.astype(np.uint8), pad_bits


# ==========================================================
# Merge Blocks
# ==========================================================

def merge_audio_blocks(blocks):
    """
    Merge blocks into one bitstream.
    """

    blocks = np.asarray(blocks, dtype=np.uint8)

    return blocks.reshape(-1)


# ==========================================================
# Remove Padding
# ==========================================================

def remove_padding(bits, pad_bits):
    """
    Remove zero padding from recovered bitstream.
    """

    bits = np.asarray(bits, dtype=np.uint8)

    if pad_bits == 0:
        return bits

    return bits[:-pad_bits]


# ==========================================================
# Frame Mapping
# ==========================================================

def frame_block_index(global_block):
    """
    Convert global block index to

    Frame ID
    Block ID
    """

    frame_id = global_block // BLOCKS_PER_FRAME
    block_id = global_block % BLOCKS_PER_FRAME

    return frame_id, block_id


# ==========================================================
# Statistics
# ==========================================================

def audio_block_statistics(blocks, pad_bits):

    blocks = np.asarray(blocks)

    stats = {
        "num_blocks": len(blocks),
        "block_size": BLOCK_SIZE,
        "padding_bits": pad_bits,
        "blocks_per_frame": BLOCKS_PER_FRAME,
        "num_frames": int(np.ceil(len(blocks)/BLOCKS_PER_FRAME))
    }

    return stats


# ==========================================================
# Self Test
# ==========================================================

if __name__ == "__main__":

    print("="*60)
    print("AUDIO BLOCK SELF TEST")
    print("="*60)

    bits = np.random.randint(
        0,
        2,
        10000,
        dtype=np.uint8
    )

    blocks, pad_bits = split_audio_blocks(bits)

    recover = merge_audio_blocks(blocks)
    recover = remove_padding(recover, pad_bits)

    ok = np.array_equal(bits, recover)

    stats = audio_block_statistics(blocks, pad_bits)

    print("Original Bits     :", len(bits))
    print("Padding Bits      :", pad_bits)
    print("Number of Blocks  :", stats["num_blocks"])
    print("Number of Frames  :", stats["num_frames"])

    print("\nRecovered Length  :", len(recover))
    print("Recovered Match   :", ok)

    assert ok

    print("\nFrame Mapping Example")

    for g in [0,1,19,20,21,39,40]:
        f,b = frame_block_index(g)
        print(f"Global {g:2d} -> Frame {f}, Block {b}")

    print("\nValidation: PASS")