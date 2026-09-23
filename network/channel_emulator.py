"""
network/channel_emulator.py

Wireless Channel Emulator
DATN - Raspberry Pi
"""

import random
import time

# ==========================================================
# Cau hinh mac dinh
# ==========================================================
PACKET_LOSS_RATE = 0.0
BIT_ERROR_RATE = 0.0
DELAY_MS = 0.0
JITTER_MS = 0.0


def configure(loss=0.0, ber=0.0, delay=0.0, jitter=0.0):
    global PACKET_LOSS_RATE, BIT_ERROR_RATE, DELAY_MS, JITTER_MS

    PACKET_LOSS_RATE = loss
    BIT_ERROR_RATE = ber
    DELAY_MS = delay
    JITTER_MS = jitter


# ==========================================================
# Packet Loss
# ==========================================================
def packet_dropped():
    return random.random() < PACKET_LOSS_RATE


# ==========================================================
# Delay + Jitter
# ==========================================================
def apply_delay():

    delay = DELAY_MS

    if JITTER_MS > 0:
        delay += random.gauss(0.0, JITTER_MS)

    if delay > 0:
        time.sleep(delay / 1000.0)


# ==========================================================
# Bit Error
# ==========================================================
def inject_bit_errors(payload: bytes):

    if BIT_ERROR_RATE <= 0:
        return payload, 0

    data = bytearray(payload)
    flips = 0

    for i in range(len(data)):
        value = data[i]

        for bit in range(8):
            if random.random() < BIT_ERROR_RATE:
                value ^= (1 << bit)
                flips += 1

        data[i] = value

    return bytes(data), flips