"""
network/config.py

Cau hinh mang TCP cho Raspberry Pi TX/RX.
"""

FRAME = {

    # ===== Packet Format =====
    "HEADER_FORMAT": "!4sBHQH",
    "TRAILER_FORMAT": "!4sI",

    # ===== Packet Markers =====
    "START_MARK": b"VSEC",
    "END_MARK": b"END!",

    # ===== Version =====
    "VERSION": 1,

    # ===== Payload =====
    "PAYLOAD_SIZE": 900,

    # Header = 17 bytes
    "HEADER_SIZE": 17,

    # Trailer = 8 bytes
    "TRAILER_SIZE": 8,
}

NETWORK = {
    "TX_IP": "192.168.0.104",
    "RX_IP": "192.168.0.102",
    "PORT": 5000,
    "BUFFER_SIZE": 4096,
    "TIMEOUT": 5,
}

AUDIO = {
    "TX_FILE": "data/voice.wav",
    "RX_FILE": "output/network_received.wav",
    "FRAME_SAMPLES": 320,
    "FRAME_BYTES": 640,
    "FRAME_DURATION_MS": 20,
    "SAMPLE_RATE": 16000,
}

LOG = {
    "LATENCY_CSV": "logs/latency.csv",
    "PACKET_CSV": "logs/packet_log.csv",
    "TX_LOG": "logs/tx.log",
    "RX_LOG": "logs/rx.log",
}