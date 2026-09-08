"""
audio/audio_crypto_pipeline.py
============================================================
Secure Audio Crypto Pipeline (DATN v1.0)

Target : Raspberry Pi 4 / Raspberry Pi 3

Pipeline
--------
Audio Bits
   ↓
Block Framing
   ↓
AES Keystream
   ↓
Crypto-Coding
   ↓
Polar Encode
   ↓
BPSK
   ↓
AWGN
   ↓
LLR
   ↓
SC Decoder
   ↓
Crypto Decoder
   ↓
Recovered Audio Bits
============================================================
"""

import numpy as np

from config.config import POLAR_K

from audio.audio_blocks import (
    split_audio_blocks,
    merge_audio_blocks,
    remove_padding,
    frame_block_index,
)

from crypto.key_manager import (
    get_master_key,
    get_iv,
    block_counter,
)

from crypto.keystream import generate_keystream

from polar.crypto_coding import (
    build_crypto_u_vector,
    extract_crypto_bits,
    decrypt_information_bits,
)

from polar.encoder import polar_encode
from polar.sc_decoder import polar_decode

from polar.utils import (
    bpsk_modulate,
    add_awgn,
    llr_awgn,
)

# ==========================================================
# Encrypt Audio Blocks
# ==========================================================

def encrypt_audio_blocks(
    audio_bits,
    ebno_db=20,
):
    """
    Encrypt every 256-bit audio block.

    Returns
    -------
    llr_blocks
    pad_bits
    statistics
    """

    blocks, pad_bits = split_audio_blocks(audio_bits)

    master_key = get_master_key()
    iv = get_iv()

    llr_blocks = []

    for global_block, info_bits in enumerate(blocks):

        frame_id, block_id = frame_block_index(global_block)

        counter = block_counter(frame_id, block_id)

        keystream = generate_keystream(
            master_key,
            iv,
            counter
        )

        cipher_u = build_crypto_u_vector(
            info_bits,
            keystream
        )

        cipher_bits, _ = extract_crypto_bits(cipher_u)

        codeword = polar_encode(
            cipher_bits,
            frozen_bits=keystream
        )

        tx = bpsk_modulate(codeword)

        rx, sigma2 = add_awgn(
            tx,
            ebno_db
        )

        llr = llr_awgn(
            rx,
            sigma2
        )

        llr_blocks.append(llr)

    stats = {
        "num_blocks": len(blocks),
        "num_frames": int(np.ceil(len(blocks)/20)),
        "padding_bits": pad_bits,
        "ebno_db": ebno_db,
    }

    return (
        np.asarray(llr_blocks, dtype=np.float32),
        pad_bits,
        stats,
    )


# ==========================================================
# Decrypt Audio Blocks
# ==========================================================

def decrypt_audio_blocks(
    llr_blocks,
    pad_bits,
):
    """
    Recover audio bitstream from LLR blocks.
    """

    master_key = get_master_key()
    iv = get_iv()

    recovered_blocks = []

    for global_block, llr in enumerate(llr_blocks):

        frame_id, block_id = frame_block_index(global_block)

        counter = block_counter(frame_id, block_id)

        keystream = generate_keystream(
            master_key,
            iv,
            counter
        )

        _, u_hat = polar_decode(
            llr,
            frozen_bits=keystream
        )

        cipher_hat, _ = extract_crypto_bits(u_hat)

        info_hat = decrypt_information_bits(
            cipher_hat,
            keystream
        )

        recovered_blocks.append(info_hat)

    bits = merge_audio_blocks(recovered_blocks)

    bits = remove_padding(bits, pad_bits)

    return bits.astype(np.uint8)


# ==========================================================
# End-to-End Secure Transmission
# ==========================================================

def secure_audio_transmission(
    audio_bits,
    ebno_db=20,
):
    """
    Complete secure audio pipeline.
    """

    llr_blocks, pad_bits, stats = encrypt_audio_blocks(
        audio_bits,
        ebno_db
    )

    recover_bits = decrypt_audio_blocks(
        llr_blocks,
        pad_bits
    )

    ber = np.mean(
        recover_bits != audio_bits
    )

    stats["ber"] = float(ber)

    return recover_bits, llr_blocks, stats


# ==========================================================
# Statistics
# ==========================================================

def audio_pipeline_statistics(stats):

    print("="*60)
    print("SECURE AUDIO PIPELINE")
    print("="*60)

    print("Audio Blocks     :", stats["num_blocks"])
    print("Audio Frames     :", stats["num_frames"])
    print("Padding Bits     :", stats["padding_bits"])
    print("Eb/N0            :", stats["ebno_db"], "dB")
    print("Recovered BER    :", stats["ber"])


# ==========================================================
# Self Test
# ==========================================================

if __name__ == "__main__":

    print("="*60)
    print("SECURE AUDIO PIPELINE SELF TEST")
    print("="*60)

    audio_bits = np.random.randint(
        0,
        2,
        256000,
        dtype=np.uint8
    )

    recover_bits, llr_blocks, stats = secure_audio_transmission(
        audio_bits,
        ebno_db=20
    )

    audio_pipeline_statistics(stats)

    print("\nRecovered Match :", np.array_equal(
        audio_bits,
        recover_bits
    ))

    assert np.array_equal(audio_bits, recover_bits)

    print("\nValidation: PASS")