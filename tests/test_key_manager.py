"""
tests/test_key_manager.py
==========================================================
Key Manager Unit Test
==========================================================
"""

from crypto.key_manager import *

print("=" * 60)
print("KEY MANAGER TEST")
print("=" * 60)

key = get_master_key()
iv = get_iv()

assert len(key) == 32
assert len(iv) == 16

assert hex_to_key(key_to_hex(key)) == key
assert hex_to_iv(iv_to_hex(iv)) == iv

print("[PASS] Key Length")
print("[PASS] IV Length")
print("[PASS] HEX Conversion")

assert block_counter(0,0) == 0
assert block_counter(0,19) == 19
assert block_counter(1,0) == 20
assert block_counter(1,19) == 39
assert block_counter(2,5) == 45

print("[PASS] Counter Generation")

counters = frame_counters(5)

assert len(counters) == 20
assert counters[0] == 100
assert counters[-1] == 119

print("[PASS] Frame Counter List")

print("\nALL KEY MANAGER TESTS PASSED")