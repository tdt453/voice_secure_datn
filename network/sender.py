"""
network/sender.py

TCP Sender + Wireless Channel Emulator
DATN - Raspberry Pi TX

Chuc nang
---------
1. Doc voice.wav (PCM16, 16 kHz).
2. Chia thanh frame 20 ms (320 samples = 640 bytes).
3. Dong goi TCP packet.
4. Gia lap packet loss / delay / jitter / bit error.
5. Gui packet va nhan ACK.
"""

import csv
import socket
import time
from pathlib import Path

import numpy as np
import soundfile as sf

from network.config import NETWORK, AUDIO
from network.packet import build_packet
from network.channel_emulator import (
    apply_delay,
    packet_dropped,
    inject_bit_errors,
)

Path("logs").mkdir(exist_ok=True)

FRAME_SAMPLES = AUDIO["FRAME_SAMPLES"]      # 320
FRAME_BYTES = AUDIO["FRAME_BYTES"]          # 640


# ==========================================================
# Doc audio WAV va chia thanh frame
# ==========================================================
def load_audio():

    samples, sr = sf.read(AUDIO["TX_FILE"], dtype="int16")

    if samples.ndim == 2:
        samples = samples[:, 0]

    frames = []

    for start in range(0, len(samples), FRAME_SAMPLES):

        frame = samples[start:start + FRAME_SAMPLES]

        if len(frame) < FRAME_SAMPLES:
            frame = np.pad(
                frame,
                (0, FRAME_SAMPLES - len(frame)),
                mode="constant"
            )

        frames.append(frame.tobytes())

    return frames


# ==========================================================
# Main Sender
# ==========================================================
def main():

    frames = load_audio()

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.settimeout(NETWORK["TIMEOUT"])

    print("=" * 60)
    print("VOICE SECURE TRANSMITTER")
    print("=" * 60)
    print(f"Connecting to {NETWORK['RX_IP']}:{NETWORK['PORT']} ...")

    client.connect((NETWORK["RX_IP"], NETWORK["PORT"]))

    print("Connected.\n")

    rtt_csv = open("logs/rtt_log.csv", "w", newline="")
    rtt_writer = csv.writer(rtt_csv)

    rtt_writer.writerow([
        "FrameID",
        "RTT_ms",
        "Dropped",
        "BitFlips"
    ])

    sent_frames = 0
    dropped_frames = 0
    total_bytes = 0

    t_begin = time.time()

    try:

        for frame_id, payload in enumerate(frames):

            # ---------------- Packet Loss Emulator ----------------
            if packet_dropped():

                dropped_frames += 1

                print(f"Frame {frame_id:03d} | DROPPED")

                rtt_writer.writerow([
                    frame_id,
                    "",
                    1,
                    0
                ])

                continue

            # ---------------- Delay / Jitter ----------------
            apply_delay()

            # ---------------- Bit Error Emulator ----------------
            payload, flips = inject_bit_errors(payload)

            packet = build_packet(frame_id, payload)

            t0 = time.perf_counter_ns()

            client.sendall(packet)

            # ==================================================
            # ACK luon doc DUNG 3 BYTE
            # ==================================================
            ack = b""

            while len(ack) < 3:
                ack += client.recv(3 - len(ack))

            ack = ack.decode()

            rtt_ms = (time.perf_counter_ns() - t0) / 1e6

            sent_frames += 1
            total_bytes += len(payload)

            print(
                f"Frame {frame_id:03d} | "
                f"{len(payload):3d} bytes | "
                f"ACK={ack:3s} | "
                f"RTT={rtt_ms:6.2f} ms | "
                f"Flips={flips}"
            )

            rtt_writer.writerow([
                frame_id,
                round(rtt_ms, 3),
                0,
                flips
            ])

    finally:

        elapsed = max(time.time() - t_begin, 1e-9)

        throughput = total_bytes / elapsed / 1024

        print("\n" + "=" * 60)
        print("TRANSMISSION SUMMARY")
        print("=" * 60)
        print(f"Frames Total       : {len(frames)}")
        print(f"Frames Sent        : {sent_frames}")
        print(f"Frames Dropped     : {dropped_frames}")
        print(
            f"Packet Loss        : "
            f"{100*dropped_frames/len(frames):.2f}%"
        )
        print(f"Bytes Sent         : {total_bytes}")
        print(f"Average Throughput : {throughput:.2f} KB/s")

        rtt_csv.close()
        client.close()


if __name__ == "__main__":
    main()