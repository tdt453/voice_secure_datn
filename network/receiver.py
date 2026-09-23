"""
network/receiver.py

TCP Receiver + Real-time Audio Playback
DATN - Raspberry Pi RX

Chuc nang
---------
1. Nhan packet TCP tu Raspberry Pi TX.
2. Kiem tra CRC va gui ACK / NAK.
3. Phat audio realtime qua jack 3.5 mm.
4. Chi luu moi FrameID mot lan.
5. Chen silence neu packet bi mat.
6. Tao network_received.bin va network_received.wav.
"""

import csv
import math
import socket
import struct
import time
from pathlib import Path

import numpy as np
import pyaudio
import soundfile as sf

from network.config import NETWORK, FRAME, LOG, AUDIO
from network.packet import parse_packet

HEADER_SIZE = struct.calcsize(FRAME["HEADER_FORMAT"])
TRAILER_SIZE = struct.calcsize(FRAME["TRAILER_FORMAT"])

Path("logs").mkdir(exist_ok=True)
Path("results").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)


# ==========================================================
# Receive exact bytes from TCP
# ==========================================================
def recv_exact(sock, size):
    data = b""
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise ConnectionError("TCP connection closed.")
        data += chunk
    return data


# ==========================================================
# Main Receiver
# ==========================================================
def main():

    # ------------------------------------------------------
    # Audio information
    # ------------------------------------------------------
    info = sf.info(AUDIO["TX_FILE"])

    TOTAL_SAMPLES = info.frames
    TOTAL_FRAMES = math.ceil(
        TOTAL_SAMPLES / AUDIO["FRAME_SAMPLES"]
    )

    print("=" * 60)
    print("VOICE SECURE RECEIVER (Realtime)")
    print("=" * 60)
    print(f"Original Samples : {TOTAL_SAMPLES}")
    print(f"Expected Frames  : {TOTAL_FRAMES}")
    print("=" * 60)

    # ------------------------------------------------------
    # CSV Logs
    # ------------------------------------------------------
    latency_csv = open(LOG["LATENCY_CSV"], "w", newline="")
    packet_csv = open(LOG["PACKET_CSV"], "w", newline="")
    rx_log = open(LOG["RX_LOG"], "w")

    latency_writer = csv.writer(latency_csv)
    packet_writer = csv.writer(packet_csv)

    latency_writer.writerow(["FrameID", "Processing_ms"])

    packet_writer.writerow([
        "FrameID",
        "PayloadBytes",
        "CRC_OK",
        "Timestamp_TX",
        "Timestamp_RX"
    ])

    # ------------------------------------------------------
    # Store unique frames
    # ------------------------------------------------------
    received_frames = {}

    # ------------------------------------------------------
    # PyAudio
    # ------------------------------------------------------
    pa = pyaudio.PyAudio()

    stream = pa.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=AUDIO["SAMPLE_RATE"],
        output=True,
        output_device_index=0,
        frames_per_buffer=AUDIO["FRAME_SAMPLES"],
    )

    # ------------------------------------------------------
    # TCP Server
    # ------------------------------------------------------
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    server.bind((NETWORK["RX_IP"], NETWORK["PORT"]))
    server.listen(1)

    print(f"Listening on {NETWORK['RX_IP']}:{NETWORK['PORT']}")
    print("Waiting for transmitter...\n")

    conn, addr = server.accept()

    print(f"Connected from {addr}\n")
    rx_log.write(f"Connected: {addr}\n")

    # ------------------------------------------------------
    # Statistics
    # ------------------------------------------------------
    frame_counter = 0
    crc_ok_counter = 0
    crc_fail_counter = 0

    # ------------------------------------------------------
    # Receive loop
    # ------------------------------------------------------
    try:

        while True:

            # HEADER
            header = recv_exact(conn, HEADER_SIZE)

            (
                _,
                _,
                frame_id,
                timestamp_tx,
                payload_len,
            ) = struct.unpack(
                FRAME["HEADER_FORMAT"],
                header
            )

            # PAYLOAD
            payload = recv_exact(conn, payload_len)

            # TRAILER
            trailer = recv_exact(conn, TRAILER_SIZE)

            packet = header + payload + trailer

            t0 = time.perf_counter_ns()

            info_packet = parse_packet(packet)

            t1 = time.perf_counter_ns()

            processing_ms = (t1 - t0) / 1e6
            timestamp_rx = time.time_ns()

            # CRC check
            if info_packet["valid"] and info_packet["crc_ok"]:

                conn.sendall(b"ACK")      # 3 bytes

                crc_ok_counter += 1

                audio_frame = bytes(info_packet["payload"])

                status = "CRC OK"

            else:

                conn.sendall(b"NAK")      # 3 bytes

                crc_fail_counter += 1

                audio_frame = bytes(AUDIO["FRAME_BYTES"])

                status = "CRC FAIL"

            # Ensure exactly 640 bytes
            if len(audio_frame) > AUDIO["FRAME_BYTES"]:
                audio_frame = audio_frame[:AUDIO["FRAME_BYTES"]]

            elif len(audio_frame) < AUDIO["FRAME_BYTES"]:
                audio_frame += bytes(
                    AUDIO["FRAME_BYTES"] - len(audio_frame)
                )

            # Playback realtime
            stream.write(audio_frame)

            # Store ONLY FIRST occurrence
            if frame_id not in received_frames:
                received_frames[frame_id] = audio_frame

            # Logs
            latency_writer.writerow([
                frame_id,
                round(processing_ms, 4)
            ])

            packet_writer.writerow([
                frame_id,
                AUDIO["FRAME_BYTES"],
                info_packet["crc_ok"],
                timestamp_tx,
                timestamp_rx
            ])

            rx_log.write(
                f"{frame_id},"
                f"{AUDIO['FRAME_BYTES']},"
                f"{status},"
                f"{processing_ms:.4f}\n"
            )

            print(
                f"Frame {frame_id:03d} | "
                f"{AUDIO['FRAME_BYTES']} bytes | "
                f"{status:7s} | "
                f"{processing_ms:7.3f} ms"
            )

            frame_counter += 1

    except ConnectionError:

        print("\nTransmission completed.")

    finally:

        # --------------------------------------------------
        # Stop Audio
        # --------------------------------------------------
        stream.stop_stream()
        stream.close()
        pa.terminate()

        # --------------------------------------------------
        # Rebuild PCM in correct frame order
        # --------------------------------------------------
        pcm_bytes = bytearray()
        lost_frame_counter = 0

        silence_frame = bytes(AUDIO["FRAME_BYTES"])

        for frame_id in range(TOTAL_FRAMES):

            if frame_id in received_frames:
                pcm_bytes.extend(received_frames[frame_id])
            else:
                pcm_bytes.extend(silence_frame)
                lost_frame_counter += 1

        # Cut to original audio size
        pcm_bytes = pcm_bytes[:TOTAL_SAMPLES * 2]

        # Save BIN
        with open("output/network_received.bin", "wb") as f:
            f.write(pcm_bytes)

        # BIN -> PCM16
        pcm = np.frombuffer(
            pcm_bytes,
            dtype=np.int16
        ).copy()

        # Save WAV
        sf.write(
            "output/network_received.wav",
            pcm,
            AUDIO["SAMPLE_RATE"],
            subtype="PCM_16"
        )

        # --------------------------------------------------
        # Summary
        # --------------------------------------------------
        print("\n" + "=" * 60)
        print("RECEIVER SUMMARY")
        print("=" * 60)
        print(f"Expected Frames      : {TOTAL_FRAMES}")
        print(f"Frames Received      : {frame_counter}")
        print(f"Unique Frames Stored : {len(received_frames)}")
        print(f"CRC OK Frames        : {crc_ok_counter}")
        print(f"CRC FAIL Frames      : {crc_fail_counter}")
        print(f"Lost Frames Inserted : {lost_frame_counter}")
        print(f"Recovered Samples    : {len(pcm)}")
        print(f"Recovered Bytes      : {len(pcm_bytes)}")
        print(
            f"Recovered Duration   : "
            f"{len(pcm)/AUDIO['SAMPLE_RATE']:.3f} s"
        )
        print("Saved WAV            : output/network_received.wav")
        print("=" * 60)

        latency_csv.close()
        packet_csv.close()
        rx_log.close()

        conn.close()
        server.close()


if __name__ == "__main__":
    main()