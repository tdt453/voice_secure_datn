"""
tests/test_utils.py
Unit tests for polar/utils.py
"""

import numpy as np

from polar.utils import (
    bit_reverse_indices,
    bit_reverse,
    bpsk_modulate,
    bpsk_demodulate,
    add_awgn,
    llr_awgn,
    hard_decision,
    f_function,
    g_function
)

print("="*55)
print("RUNNING POLAR UTILS TEST")
print("="*55)

# -------------------------------------------------------
# Test 1 : Bit-Reversal
# -------------------------------------------------------

idx = bit_reverse_indices(8)

expected = np.array([0,4,2,6,1,5,3,7])

assert np.array_equal(idx, expected)

print("[PASS] Bit-Reversal")

# -------------------------------------------------------
# Test 2 : Permutation
# -------------------------------------------------------

x = np.arange(8)

xr = bit_reverse(x)

assert np.array_equal(
    xr,
    np.array([0,4,2,6,1,5,3,7])
)

print("[PASS] Bit-Reversal Permutation")

# -------------------------------------------------------
# Test 3 : BPSK
# -------------------------------------------------------

bits = np.random.randint(0,2,32)

symbols = bpsk_modulate(bits)

bits_hat = bpsk_demodulate(symbols)

assert np.array_equal(bits,bits_hat)

print("[PASS] BPSK Mod/Demod")

# -------------------------------------------------------
# Test 4 : AWGN + LLR
# -------------------------------------------------------

noisy, sigma2 = add_awgn(symbols, ebno_db=8)

llr = llr_awgn(noisy,sigma2)

bits_hat = hard_decision(llr)

ber = np.mean(bits_hat != bits)

print(f"[PASS] AWGN LLR BER = {ber:.4f}")

# -------------------------------------------------------
# Test 5 : f Function
# -------------------------------------------------------

a = np.array([2,-3,1,-4],dtype=np.float32)
b = np.array([1,5,-2,-1],dtype=np.float32)

expected = np.array([1,-3,-1,1],dtype=np.float32)

assert np.array_equal(
    f_function(a,b),
    expected
)

print("[PASS] f() Function")

# -------------------------------------------------------
# Test 6 : g Function
# -------------------------------------------------------

u = np.array([0,1,0,1],dtype=np.uint8)

expected = np.array([3,8,-1,3],dtype=np.float32)

assert np.array_equal(
    g_function(a,b,u),
    expected
)

print("[PASS] g() Function")

print("\nALL POLAR UTILS TEST PASSED")