"""
Opus Codec
"""

import opuslib
import numpy as np

from compression_experiment.config import (
    SAMPLE_RATE,
    CHANNELS,
    FRAME_SAMPLES,
    OPUS_BITRATE
)

encoder = opuslib.Encoder(
    SAMPLE_RATE,
    CHANNELS,
    opuslib.APPLICATION_AUDIO
)

decoder = opuslib.Decoder(
    SAMPLE_RATE,
    CHANNELS
)

encoder.bitrate = OPUS_BITRATE


def encode_frame(frame):

    return encoder.encode(
        frame.astype(np.int16).tobytes(),
        FRAME_SAMPLES
    )


def decode_frame(packet):

    pcm = decoder.decode(packet, FRAME_SAMPLES)

    return np.frombuffer(pcm, dtype=np.int16)


def encode_audio(frames):

    packets = []

    for frame in frames:
        packets.append(encode_frame(frame))

    return packets


def decode_audio(packets):

    frames = []

    for packet in packets:
        frames.append(decode_frame(packet))

    return frames