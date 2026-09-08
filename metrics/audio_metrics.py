"""
metrics/audio_metrics.py
============================================================
Audio Security Metrics (DATN v1.0)

Metrics for Secure Audio Crypto-Coding Evaluation

Target
------
Raspberry Pi 4 / Raspberry Pi 3

Functions
---------
1. Shannon Entropy
2. Adjacent Sample Correlation
3. Mean Squared Error (MSE)
4. Peak Signal-to-Noise Ratio (PSNR)
5. Signal-to-Noise Ratio (SNR)
6. Bit Error Rate (BER)

Used in
--------
LAB 11E
LAB 11F
Chapter 4 Experiments
============================================================
"""

import numpy as np


# ==========================================================
# Shannon Entropy
# ==========================================================

def shannon_entropy(signal: np.ndarray) -> float:
    """
    Compute Shannon entropy of an audio signal.

    Parameters
    ----------
    signal : ndarray
        int16 or float waveform.

    Returns
    -------
    float
        Shannon entropy (bits/sample).
    """

    signal = np.asarray(signal)

    if signal.dtype != np.uint8:
        signal = signal.astype(np.int16)

        signal = signal.astype(np.int32) + 32768
        signal = signal // 256
        signal = signal.astype(np.uint8)

    histogram = np.bincount(signal, minlength=256).astype(np.float64)

    probability = histogram / histogram.sum()

    probability = probability[probability > 0]

    entropy = -np.sum(probability * np.log2(probability))

    return float(entropy)


# ==========================================================
# Adjacent Correlation
# ==========================================================

def adjacent_correlation(signal: np.ndarray) -> float:
    """
    Correlation between adjacent samples.

    Returns
    -------
    float
    """

    signal = np.asarray(signal, dtype=np.float64)

    x = signal[:-1]
    y = signal[1:]

    if np.std(x) == 0 or np.std(y) == 0:
        return 0.0

    corr = np.corrcoef(x, y)[0, 1]

    return float(corr)


# ==========================================================
# Mean Squared Error
# ==========================================================

def mse(reference: np.ndarray,
        recovered: np.ndarray) -> float:
    """
    Mean Squared Error.
    """

    reference = np.asarray(reference, dtype=np.float64)
    recovered = np.asarray(recovered, dtype=np.float64)

    return float(np.mean((reference - recovered) ** 2))


# ==========================================================
# PSNR
# ==========================================================

def psnr(reference: np.ndarray,
         recovered: np.ndarray,
         peak=32767.0) -> float:
    """
    Peak Signal-to-Noise Ratio.
    """

    error = mse(reference, recovered)

    if error == 0:
        return float("inf")

    value = 10 * np.log10((peak ** 2) / error)

    return float(value)


# ==========================================================
# SNR
# ==========================================================

def snr(reference: np.ndarray,
        recovered: np.ndarray) -> float:
    """
    Signal-to-Noise Ratio.
    """

    reference = np.asarray(reference, dtype=np.float64)
    recovered = np.asarray(recovered, dtype=np.float64)

    noise = reference - recovered

    signal_power = np.mean(reference ** 2)
    noise_power = np.mean(noise ** 2)

    if noise_power == 0:
        return float("inf")

    value = 10 * np.log10(signal_power / noise_power)

    return float(value)


# ==========================================================
# Bit Error Rate
# ==========================================================

def bit_error_rate(bits_ref: np.ndarray,
                   bits_rec: np.ndarray) -> float:
    """
    BER between two bitstreams.
    """

    bits_ref = np.asarray(bits_ref, dtype=np.uint8)
    bits_rec = np.asarray(bits_rec, dtype=np.uint8)

    if len(bits_ref) != len(bits_rec):
        raise ValueError("Bitstream lengths do not match.")

    return float(np.mean(bits_ref != bits_rec))


# ==========================================================
# Metrics Summary
# ==========================================================

def evaluate_audio(reference_audio,
                   encrypted_audio,
                   recovered_audio,
                   reference_bits=None,
                   recovered_bits=None):
    """
    Compute all metrics used in Chapter 4.

    Returns
    -------
    dict
    """

    metrics = {

        "OriginalEntropy":
            shannon_entropy(reference_audio),

        "EncryptedEntropy":
            shannon_entropy(encrypted_audio),

        "RecoveredEntropy":
            shannon_entropy(recovered_audio),

        "OriginalCorrelation":
            adjacent_correlation(reference_audio),

        "EncryptedCorrelation":
            adjacent_correlation(encrypted_audio),

        "RecoveredCorrelation":
            adjacent_correlation(recovered_audio),

        "MSE":
            mse(reference_audio, recovered_audio),

        "PSNR":
            psnr(reference_audio, recovered_audio),

        "SNR":
            snr(reference_audio, recovered_audio),

    }

    if reference_bits is not None and recovered_bits is not None:

        metrics["BER"] = bit_error_rate(
            reference_bits,
            recovered_bits
        )

    return metrics


# ==========================================================
# Standalone Validation
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("AUDIO METRICS SELF TEST")
    print("=" * 60)

    np.random.seed(42)

    original = np.random.randint(
        -32768,
        32767,
        16000,
        dtype=np.int16
    )

    recovered = original.copy()

    encrypted = np.random.randint(
        -32768,
        32767,
        16000,
        dtype=np.int16
    )

    metrics = evaluate_audio(
        original,
        encrypted,
        recovered
    )

    print("\nEntropy")
    print("Original :", round(metrics["OriginalEntropy"], 4))
    print("Encrypted:", round(metrics["EncryptedEntropy"], 4))
    print("Recovered:", round(metrics["RecoveredEntropy"], 4))

    print("\nCorrelation")
    print("Original :", round(metrics["OriginalCorrelation"], 4))
    print("Encrypted:", round(metrics["EncryptedCorrelation"], 4))
    print("Recovered:", round(metrics["RecoveredCorrelation"], 4))

    print("\nQuality")
    print("MSE  :", metrics["MSE"])
    print("PSNR :", metrics["PSNR"])
    print("SNR  :", metrics["SNR"])

    assert metrics["MSE"] == 0
    assert np.isinf(metrics["PSNR"])
    assert np.isinf(metrics["SNR"])

    print("\nValidation: PASS")