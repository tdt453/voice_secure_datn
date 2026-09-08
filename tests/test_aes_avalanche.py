import numpy as np

from crypto.aes_ctr import encrypt_block


bits = np.random.randint(0,2,256,dtype=np.uint8)

bits_flip = bits.copy()

bits_flip[0] ^= 1

enc1 = encrypt_block(bits)
enc2 = encrypt_block(bits_flip)

changed = np.sum(enc1 != enc2)

print("Changed bits :", changed)
print("Avalanche (%) :", changed/256*100)