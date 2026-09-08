"""
tests/test_sc_decoder.py
SC Decoder Unit Test
"""

import numpy as np

from config.config import POLAR_N, POLAR_K
from polar.encoder import polar_encode
from polar.sc_decoder import polar_decode
from polar.utils import (
    bpsk_modulate,
    add_awgn,
    llr_awgn
)

print("=" * 60)
print("RUNNING SC DECODER TEST")
print("=" * 60)

# -------------------------------------------------------
# Test 1 : Decoder output size
# -------------------------------------------------------

info = np.random.randint(
    0,2,POLAR_K,dtype=np.uint8
)

codeword = polar_encode(info)

symbols = bpsk_modulate(codeword)

noisy, sigma2 = add_awgn(symbols, ebno_db=12)

llr = llr_awgn(noisy,sigma2)

info_hat, u_hat = polar_decode(llr)

assert len(info_hat)==POLAR_K
assert len(u_hat)==POLAR_N

print("[PASS] Decoder Output Length")

# -------------------------------------------------------
# Test 2 : BER at High Eb/N0
# -------------------------------------------------------

ber = np.mean(info != info_hat)

print(f"[INFO] BER = {ber:.6f}")

assert ber < 0.02

print("[PASS] High-SNR Decode")

# -------------------------------------------------------
# Test 3 : Multiple Random Frames
# -------------------------------------------------------

frames = 20
total_error = 0

for _ in range(frames):

    info = np.random.randint(
        0,2,POLAR_K,dtype=np.uint8
    )

    codeword = polar_encode(info)

    symbols = bpsk_modulate(codeword)

    noisy, sigma2 = add_awgn(symbols, ebno_db=10)

    llr = llr_awgn(noisy,sigma2)

    info_hat,_ = polar_decode(llr)

    total_error += np.sum(info != info_hat)

ber = total_error/(frames*POLAR_K)

print(f"[INFO] Average BER = {ber:.6f}")

assert ber < 0.03

print("[PASS] 20 Random Frames")

print("\nALL SC DECODER TESTS PASSED")