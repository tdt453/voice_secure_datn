import socket
import time
import numpy as np

from audio.reader import read_audio
from audio.framing import split_frames
from audio.bitstream import pcm_to_bits
from audio.audio_blocks import split_audio_blocks
from audio.audio_crypto_pipeline import encrypt_audio_blocks
from network.packet import build_packet
from network.config import NETWORK, AUDIO


def main():

    audio, sr = read_audio(AUDIO["TX_FILE"])
    frames = split_frames(audio)

    client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client.connect((NETWORK["RX_IP"], NETWORK["PORT"]))

    print("=" * 60)
    print("SECURE AUDIO TRANSMITTER")
    print("=" * 60)
    print(f"Frames: {len(frames)}")

    frame_id = 0
    total_bytes = 0

    start = time.time()

    for frame in frames:

        bits = pcm_to_bits(frame)

        llr_blocks, _, _ = encrypt_audio_blocks(bits, ebno_db=20)

        payload = llr_blocks.astype(np.float32).tobytes()

        packet = build_packet(frame_id, payload)

        t0 = time.perf_counter()

        client.sendall(packet)

        ack = client.recv(4).decode()

        t1 = time.perf_counter()

        rtt = (t1 - t0) * 1000

        print(
            f"Frame {frame_id:04d} | "
            f"{len(payload):4d} bytes | "
            f"{ack:4s} | RTT={rtt:.2f} ms"
        )

        total_bytes += len(packet)
        frame_id += 1

    elapsed = time.time() - start

    print("\nTransmission Completed")
    print("Frames Sent :", frame_id)
    print("Bytes Sent  :", total_bytes)
    print(f"Time        : {elapsed:.2f} s")
    print(f"Throughput  : {total_bytes/elapsed/1024:.2f} KB/s")

    client.close()


if __name__ == "__main__":
    main()