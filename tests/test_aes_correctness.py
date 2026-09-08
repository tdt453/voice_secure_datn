import numpy as np
from crypto.aes_ctr import encrypt_block, decrypt_block

bits = np.random.randint(0, 2, 256, dtype=np.uint8)

enc = encrypt_block(bits)
dec = decrypt_block(enc)

print("Input bits :", len(bits))
print("Encrypted  :", len(enc))
print("Recovered  :", np.array_equal(bits, dec))