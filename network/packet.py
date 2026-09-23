"""
network/packet.py

Dong goi va giai ma packet TCP cho Voice Secure.
DATN - Raspberry Pi
"""

import struct
import time
import zlib

from network.config import FRAME

# ==========================================================
# Packet Format
# ==========================================================

HEADER_FORMAT = FRAME["HEADER_FORMAT"]
TRAILER_FORMAT = FRAME["TRAILER_FORMAT"]

HEADER_SIZE = struct.calcsize(HEADER_FORMAT)
TRAILER_SIZE = struct.calcsize(TRAILER_FORMAT)


# ==========================================================
# Build Packet
# ==========================================================

def build_packet(frame_id, payload):
    """
    Tao packet TCP:
    HEADER | PAYLOAD | TRAILER
    """

    payload = bytes(payload)

    timestamp = time.time_ns()

    header = struct.pack(
        HEADER_FORMAT,
        FRAME["START_MARK"],
        FRAME["VERSION"],
        frame_id,
        timestamp,
        len(payload),
    )

    crc = zlib.crc32(payload) & 0xFFFFFFFF

    trailer = struct.pack(
        TRAILER_FORMAT,
        FRAME["END_MARK"],
        crc,
    )

    return header + payload + trailer


# ==========================================================
# Parse Packet
# ==========================================================

def parse_packet(packet):
    """
    Giai ma packet TCP va kiem tra CRC.
    """

    # Packet toi thieu
    if len(packet) < HEADER_SIZE + TRAILER_SIZE:
        return {
            "valid": False,
            "crc_ok": False,
        }

    try:

        header = packet[:HEADER_SIZE]

        (
            start_mark,
            version,
            frame_id,
            timestamp,
            payload_len,
        ) = struct.unpack(
            HEADER_FORMAT,
            header,
        )

        expected_size = HEADER_SIZE + payload_len + TRAILER_SIZE

        # Packet khong du kich thuoc
        if len(packet) != expected_size:
            return {
                "valid": False,
                "crc_ok": False,
                "frame_id": frame_id,
                "payload_len": payload_len,
            }

        payload_start = HEADER_SIZE
        payload_end = payload_start + payload_len

        payload = packet[payload_start:payload_end]

        trailer = packet[payload_end:payload_end + TRAILER_SIZE]

        end_mark, recv_crc = struct.unpack(
            TRAILER_FORMAT,
            trailer,
        )

        calc_crc = zlib.crc32(payload) & 0xFFFFFFFF

        valid = (
            start_mark == FRAME["START_MARK"]
            and end_mark == FRAME["END_MARK"]
            and version == FRAME["VERSION"]
        )

        crc_ok = recv_crc == calc_crc

        return {
            "valid": valid,
            "crc_ok": crc_ok,
            "frame_id": frame_id,
            "timestamp": timestamp,
            "payload_len": payload_len,
            "payload": payload,
            "recv_crc": recv_crc,
            "calc_crc": calc_crc,
        }

    except Exception:
        return {
            "valid": False,
            "crc_ok": False,
        }


# ==========================================================
# Self Test
# ==========================================================

if __name__ == "__main__":

    payload = bytes(range(64))

    packet = build_packet(5, payload)

    info = parse_packet(packet)

    print("=" * 60)
    print("PACKET SELF TEST")
    print("=" * 60)
    print("Valid      :", info["valid"])
    print("CRC OK     :", info["crc_ok"])
    print("Frame ID   :", info["frame_id"])
    print("Payload Len:", info["payload_len"])
    print("Payload OK :", info["payload"] == payload)