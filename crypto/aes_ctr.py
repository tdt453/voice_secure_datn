"""
crypto/aes_ctr.py

AES-CTR cho Polar information blocks.
"""

from Crypto.Cipher import AES
from Crypto.Util import Counter
import numpy as np

from config.config import AES_KEY_SIZE, NONCE_SIZE


# ==========================================================
# Key
# ==========================================================

MASTER_KEY = bytes.fromhex(
    "00112233445566778899AABBCCDDEEFF"
)

NONCE = bytes.fromhex(
    "1234567890ABCDEF"
)


# ==========================================================
# CTR Cipher
# ==========================================================

def create_cipher():

    ctr = Counter.new(
        64,
        prefix=NONCE,
        initial_value=0,
        little_endian=False
    )

    return AES.new(
        MASTER_KEY,
        AES.MODE_CTR,
        counter=ctr
    )


# ==========================================================
# Block Encryption
# ==========================================================

def encrypt_block(bits):

    cipher = create_cipher()

    packed = np.packbits(bits)

    encrypted = cipher.encrypt(packed.tobytes())

    enc_bits = np.unpackbits(
        np.frombuffer(encrypted, dtype=np.uint8)
    )

    return enc_bits[:len(bits)]


def decrypt_block(bits):
    return encrypt_block(bits)


if __name__ == "__main__":

    test_bits = np.random.randint(
        0,
        2,
        256,
        dtype=np.uint8
    )

    enc = encrypt_block(test_bits)

    dec = decrypt_block(enc)

    print("========== AES TEST ==========")
    print("Input bits :", len(test_bits))
    print("Encrypted  :", len(enc))
    print("Recovered  :", np.array_equal(test_bits, dec))