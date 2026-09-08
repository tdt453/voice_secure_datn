import numpy as np

from polar.reliability import (
    validate_sequence,
    reliability_sequence,
    information_positions,
    frozen_positions,
    frozen_mask,
)

validate_sequence()

seq = reliability_sequence()
info = information_positions()
frozen = frozen_positions()
mask = frozen_mask()

print("========== TEST RELIABILITY ==========")
print("Sequence Length :", len(seq))
print("Information Bits:", len(info))
print("Frozen Bits     :", len(frozen))

print("Mask(0) =", np.sum(mask == 0))
print("Mask(1) =", np.sum(mask == 1))

print("Overlap =", np.intersect1d(info, frozen).size)

assert np.sum(mask == 0) == 256
assert np.sum(mask == 1) == 256

print("\nPASS")