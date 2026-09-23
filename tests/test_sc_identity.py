"""
Identity Test
Encoder -> Decoder (No Noise)
"""

import numpy as np

from config.config import POLAR_K
from polar.encoder import polar_encode
from polar.sc_decoder import polar_decode
from polar.utils import bpsk_modulate

print("="*60)
print("POLAR IDENTITY TEST (NO NOISE)")
print("="*60)

# Random information bits
info = np.random.randint(
    0,2,POLAR_K,dtype=np.uint8
)

# Encode
codeword = polar_encode(info)

# Perfect channel (BPSK without noise)
symbols = bpsk_modulate(codeword)

# Infinite LLR
llr = symbols * 1000.0

decoded, _ = polar_decode(llr)

ber = np.mean(decoded != info)

print("BER =", ber)

assert ber == 0

print("[PASS] Identity Test")