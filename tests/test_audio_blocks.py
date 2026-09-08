"""
tests/test_audio_blocks.py
============================================================
Unit Test : Audio Block Framing
============================================================
"""

import numpy as np

from audio.audio_blocks import (
    split_audio_blocks,
    merge_audio_blocks,
    remove_padding,
    padding_length,
    frame_block_index,
)

print("="*60)
print("AUDIO BLOCK TEST")
print("="*60)

# ----------------------------------------------------------
# Random Bitstream
# ----------------------------------------------------------

bits = np.random.randint(
    0,
    2,
    12345,
    dtype=np.uint8
)

blocks, pad = split_audio_blocks(bits)

assert blocks.shape[1] == 256

print("[PASS] Split Blocks")

recover = merge_audio_blocks(blocks)
recover = remove_padding(recover, pad)

assert np.array_equal(bits, recover)

print("[PASS] Merge Blocks")

assert padding_length(bits) == pad

print("[PASS] Padding Removal")

# ----------------------------------------------------------
# Frame Mapping
# ----------------------------------------------------------

assert frame_block_index(0) == (0,0)
assert frame_block_index(19) == (0,19)
assert frame_block_index(20) == (1,0)
assert frame_block_index(39) == (1,19)
assert frame_block_index(40) == (2,0)

print("[PASS] Frame Index Mapping")

print("\nALL AUDIO BLOCK TESTS PASSED")