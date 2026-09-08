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
)

from polar.encoder import polar_encode
from polar.sc_decoder import polar_decode

from polar.utils import (
    bpsk_modulate,
    add_awgn,
    llr_awgn,
)

# ======================================================
# Parameters
# ======================================================

FRAME_ID = 0
BLOCK_ID = 0
EBNO_DB = 20

print("="*60)
print("WRONG MASTER KEY TEST")
print("="*60)

# ======================================================
# Original Information
# ======================================================

info_bits = np.random.randint(
    0,2,POLAR_K,dtype=np.uint8
)

# ======================================================
# Correct Key
# ======================================================

master_key = get_master_key()
iv = get_iv()

counter = block_counter(
    FRAME_ID,
    BLOCK_ID
)

keystream = generate_keystream(
    master_key,
    iv,
    counter=counter
)

# ======================================================
# Wrong Key (Flip exactly ONE BIT)
# ======================================================

wrong_key = bytearray(master_key)

wrong_key[0] ^= 0x01      # Flip LSB of first byte.

wrong_keystream = generate_keystream(
    bytes(wrong_key),
    iv,
    counter=counter
)

# ======================================================
# Crypto Coding + Polar
# ======================================================

u = build_crypto_u_vector(
    info_bits,
    keystream
)

cipher_bits, _ = extract_crypto_bits(u)

codeword = polar_encode(
    cipher_bits,
    frozen_bits=keystream
)

# ======================================================
# Channel
# ======================================================

tx = bpsk_modulate(codeword)

rx, sigma2 = add_awgn(
    tx,
    ebno_db=EBNO_DB
)

llr = llr_awgn(rx, sigma2)

# ======================================================
# Receiver (Bob)
# ======================================================

_, u_hat = polar_decode(
    llr,
    frozen_bits=keystream
)

cipher_hat, _ = extract_crypto_bits(u_hat)

recover_correct = decrypt_information_bits(
    cipher_hat,
    keystream
)

ber_correct = np.mean(
    recover_correct != info_bits
)

# ======================================================
# Receiver (Eve)
# ======================================================

recover_wrong = decrypt_information_bits(
    cipher_hat,
    wrong_keystream
)

ber_wrong = np.mean(
    recover_wrong != info_bits
)

# ======================================================
# Statistics
# ======================================================

bit_diff = np.mean(
    keystream != wrong_keystream
)

print("Correct BER      :", ber_correct)
print("Wrong-Key BER    :", ber_wrong)
print("Keystream Diff   :", bit_diff)

assert ber_correct == 0

print("\nPASS : Correct key recovers perfectly.")

if ber_wrong > 0.45:
    print("PASS : Wrong key destroys plaintext.")
else:
    print("WARNING : Avalanche effect too weak.")