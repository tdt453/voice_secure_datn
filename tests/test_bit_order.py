"""
Kiểm tra thứ tự bit giữa Encoder và SC Decoder.
"""

import numpy as np

from polar.encoder import polar_encode
from polar.reliability import (
    information_positions,
    frozen_positions,
    build_u_vector,
)
from polar.utils import (
    bpsk_modulate,
    llr_awgn,
)

from polar.sc_decoder import polar_decode

print("="*60)
print("BIT ORDER TEST")
print("="*60)

# Sinh information bits
info = np.random.randint(0,2,256,dtype=np.uint8)

# Encode
x = polar_encode(info)

# Không thêm nhiễu
symbols = bpsk_modulate(x)
llr = llr_awgn(symbols, sigma2=1e-9)

# Decode
info_hat, u_hat = polar_decode(llr)

print("Recovered BER :", np.mean(info != info_hat))

# Kiểm tra frozen bits
print("Frozen bits decoded:",
      np.sum(u_hat[frozen_positions()]))

print("Info bits first 16")
print(info[:16])

print("Recovered first 16")
print(info_hat[:16])