"""
tests/test_crypto_end_to_end.py
============================================================
End-to-End Crypto-Coding Test (DATN v2.0)

Pipeline

Information Bits
      ↓
Crypto Coding (XOR + Frozen Replacement)
      ↓
Polar Encoder
      ↓
BPSK
      ↓
AWGN
      ↓
LLR
      ↓
SC Decoder
      ↓
Crypto Decoder (XOR)
============================================================
"""

import numpy as np

from config.config import POLAR_K

from crypto.key_manager import (
    get_master_key,
    get_iv,
    block_counter,
)

from crypto.keystream import generate_keystream

from polar.crypto_coding import (
    build_crypto_u_vector,
    extract_crypto_bits,
    decrypt_information_bits,
    verify_frozen_bits,
)

from polar.encoder import polar_encode
from polar.sc_decoder import polar_decode

from polar.utils import (
    bpsk_modulate,
    add_awgn,
    llr_awgn,
)

# ==========================================================
# Parameters
# ==========================================================

FRAME_ID = 0
BLOCK_ID = 0
EBNO_DB = 20

print("=" * 60)
print("END-TO-END CRYPTO CODING TEST")
print("=" * 60)

# ==========================================================
# Random Information Bits
# ==========================================================

info_bits = np.random.randint(
    0, 2,
    POLAR_K,
    dtype=np.uint8,
)

print("Information Bits :", len(info_bits))

# ==========================================================
# AES Keystream
# ==========================================================

key = get_master_key()
iv = get_iv()

counter = block_counter(FRAME_ID, BLOCK_ID)

keystream = generate_keystream(
    key,
    iv,
    counter=counter,
)

print("Keystream Bits   :", len(keystream))

# ==========================================================
# Crypto Coding
# ==========================================================

u = build_crypto_u_vector(
    info_bits,
    keystream,
)

# ==========================================================
# Polar Encode
# ==========================================================

cipher_bits, _ = extract_crypto_bits(u)

codeword = polar_encode(
    cipher_bits,
    frozen_bits=keystream,
)

print("Codeword Length  :", len(codeword))

# ==========================================================
# BPSK + AWGN
# ==========================================================

tx = bpsk_modulate(codeword)

rx, sigma2 = add_awgn(
    tx,
    ebno_db=EBNO_DB,
)

llr = llr_awgn(
    rx,
    sigma2,
)

# ==========================================================
# Polar Decode
# ==========================================================

_, u_hat = polar_decode(
    llr,
    frozen_bits=keystream,
)

cipher_hat, frozen_hat = extract_crypto_bits(u_hat)

# ==========================================================
# Frozen Verification
# ==========================================================

ok, frozen_ber = verify_frozen_bits(
    frozen_hat,
    keystream,
)

# ==========================================================
# Crypto Decryption
# ==========================================================

recover_bits = decrypt_information_bits(
    cipher_hat,
    keystream,
)

ber = np.mean(recover_bits != info_bits)

# ==========================================================
# Results
# ==========================================================

print("\nRecovered BER :", ber)
print("Frozen BER    :", frozen_ber)

assert ok
assert frozen_ber == 0
assert ber == 0

print("\n[PASS] Frozen Verification")
print("[PASS] Crypto Decryption")
print("[PASS] End-to-End BER = 0")

print("\nALL END-TO-END TESTS PASSED")