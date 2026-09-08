"""
audio/audio_io.py
============================================================
Audio I/O Utilities (DATN v1.0)

Target : Raspberry Pi 4 / Raspberry Pi 3

Functions
---------
1. Read WAV (16-bit PCM).
2. Write WAV.
3. Convert PCM samples -> bitstream.
4. Convert bitstream -> PCM samples.
5. Audio statistics.

Compatible with Crypto-Coding Polar Pipeline.
============================================================
"""

import numpy as np
import soundfile as sf

# ==========================================================
# Read WAV
# ==========================================================

def read_wav(filepath):
    """
    Read a WAV file.

    Returns
    -------
    samples : np.ndarray (int16)
    sample_rate : int
    """

    samples, sample_rate = sf.read(filepath, dtype="int16")

    samples = np.asarray(samples, dtype=np.int16)

    return samples, sample_rate


# ==========================================================
# Write WAV
# ==========================================================

def write_wav(filepath, samples, sample_rate):
    """
    Write int16 samples to WAV.
    """

    samples = np.asarray(samples, dtype=np.int16)

    sf.write(filepath, samples, sample_rate)


# ==========================================================
# PCM -> Bitstream
# ==========================================================

def audio_to_bits(samples):
    """
    Convert int16 PCM samples into bitstream.

    Returns
    -------
    bits : uint8 array
    """

    samples = np.asarray(samples, dtype=np.int16)

    # reinterpret memory as uint16
    uint_samples = samples.view(np.uint16)

    bytes_data = uint_samples.tobytes()

    bits = np.unpackbits(
        np.frombuffer(bytes_data, dtype=np.uint8)
    )

    return bits.astype(np.uint8)


# ==========================================================
# Bitstream -> PCM
# ==========================================================

def bits_to_audio(bits):
    """
    Recover int16 samples from bitstream.

    Parameters
    ----------
    bits : uint8 array

    Returns
    -------
    samples : int16 ndarray
    """

    bits = np.asarray(bits, dtype=np.uint8)

    if len(bits) % 16 != 0:
        raise ValueError(
            "Bitstream length must be multiple of 16."
        )

    bytes_data = np.packbits(bits)

    uint_samples = np.frombuffer(
        bytes_data.tobytes(),
        dtype=np.uint16
    )

    samples = uint_samples.view(np.int16)

    return samples.copy()


# ==========================================================
# Audio Statistics
# ==========================================================

def audio_statistics(samples, sample_rate):
    """
    Return audio information.
    """

    samples = np.asarray(samples)

    duration = len(samples) / sample_rate

    stats = {
        "sample_rate": sample_rate,
        "samples": len(samples),
        "duration_sec": duration,
        "bit_depth": 16,
        "channels": 1 if samples.ndim == 1 else samples.shape[1],
        "min": int(samples.min()),
        "max": int(samples.max()),
    }

    return stats


# ==========================================================
# Self Test
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AUDIO IO SELF TEST")
    print("=" * 60)

    SAMPLE_RATE = 16000

    duration = 1.0

    t = np.arange(
        int(duration * SAMPLE_RATE)
    ) / SAMPLE_RATE

    # 440 Hz sine wave
    samples = (
        12000 * np.sin(2 * np.pi * 440 * t)
    ).astype(np.int16)

    bits = audio_to_bits(samples)

    recover = bits_to_audio(bits)

    ok = np.array_equal(samples, recover)

    stats = audio_statistics(samples, SAMPLE_RATE)

    print("Sample Rate      :", stats["sample_rate"])
    print("Channels         :", stats["channels"])
    print("Samples          :", stats["samples"])
    print("Duration         :", stats["duration_sec"], "sec")
    print("Bitstream Length :", len(bits))

    print("\nRecovered Audio Match :", ok)

    assert ok

    print("\nValidation: PASS")