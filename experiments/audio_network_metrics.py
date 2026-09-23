"""
Danh gia file voice.wav sau khi truyen TCP.
"""

import numpy as np
import soundfile as sf

from metrics.audio_metrics import (
    bit_error_rate,
    mse,
    psnr,
    snr,
)

TX_FILE = "data/voice.wav"
RX_FILE = "output/network_received.wav"


def main():

    tx, sr1 = sf.read(TX_FILE, dtype="int16")
    rx, sr2 = sf.read(RX_FILE, dtype="int16")

    assert sr1 == sr2 == 16000

    L = min(len(tx), len(rx))
    tx = tx[:L]
    rx = rx[:L]

    ber = bit_error_rate(tx.view(np.uint16), rx.view(np.uint16))
    mse_value = mse(tx, rx)
    psnr_value = psnr(tx, rx)
    snr_value = snr(tx, rx)

    print("=" * 60)
    print("NETWORK AUDIO QUALITY")
    print("=" * 60)
    print(f"Samples      : {L}")
    print(f"BER          : {ber:.8f}")
    print(f"MSE          : {mse_value:.6f}")
    print(f"PSNR         : {psnr_value:.2f} dB")
    print(f"SNR          : {snr_value:.2f} dB")


if __name__ == "__main__":
    main()