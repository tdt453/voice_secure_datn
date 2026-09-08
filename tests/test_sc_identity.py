"""
tests/test_sc_identity.py
Kiểm tra Encoder -> Decoder KHÔNG có nhiễu.
"""

import numpy as np

from config.config import POLAR_K
from polar.encoder import polar_encode
from polar.sc_decoder import polar_decode
from polar.utils import bpsk_modulate

print("="*60)
print("POLAR IDENTITY TEST (NO NOISE)")
print("="*60)

info = np.random.randint(0,2,POLAR_K,dtype=np.uint8)

codeword = polar_encode(info)

# Không thêm nhiễu
symbols = bpsk_modulate(codeword)

# LLR rất lớn
llr = symbols * 1000.0

info_hat, _ = polar_decode(llr)

ber = np.mean(info != info_hat)

print("BER =", ber)

assert ber == 0

print("PASS")