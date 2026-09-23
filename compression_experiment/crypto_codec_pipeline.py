"""
Codec + Crypto Pipeline
"""

from pathlib import Path
import numpy as np

from compression_experiment.utils import (
    read_pcm,
    write_pcm,
    split_frames,
    merge_frames
)

from compression_experiment.codec_opus import (
    encode_audio,
    decode_audio
)

from audio.audio_io import audio_to_bits
from audio.audio_io import bits_to_audio

from audio.audio_crypto_pipeline import (
    secure_audio_transmission
)


def run_opus_crypto(input_wav, output_dir, ebno_db=20):

    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    pcm = read_pcm(input_wav)

    frames = split_frames(pcm)

    packets = encode_audio(frames)

    payload = b"".join(packets)

    np.frombuffer(payload, dtype=np.uint8).tofile(
        output_dir / "opus_payload.bin"
    )

    bits = np.unpackbits(
        np.frombuffer(payload, dtype=np.uint8)
    )

    recovered_bits, llr, stats = secure_audio_transmission(
        bits,
        ebno_db
    )

    recovered_payload = np.packbits(
        recovered_bits
    ).tobytes()

    idx = 0
    recovered_packets = []

    for packet in packets:

        L = len(packet)

        recovered_packets.append(
            recovered_payload[idx:idx + L]
        )

        idx += L

    recovered_frames = decode_audio(
        recovered_packets
    )

    recovered_audio = merge_frames(
        recovered_frames,
        len(pcm)
    )

    write_pcm(
        output_dir / "opus_recovered.wav",
        recovered_audio
    )

    return stats