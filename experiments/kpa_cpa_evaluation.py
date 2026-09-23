import hashlib
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from crypto.key_manager import get_master_key, get_iv, block_counter
from crypto.keystream import generate_keystream

# ============================================================
# CONFIGURATION
# ============================================================

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

KEY_BITS = 256
PLOT_BITS = 100          # So bit hien thi trong hinh
RANDOM_SEED = 2026

np.random.seed(RANDOM_SEED)

# ============================================================
# KEYSTREAM GENERATION
# ============================================================

MASTER_KEY = get_master_key()
IV = get_iv()


def generate_audio_keystream(audio_name, frame_id=0, block_id=0):
    """
    Sinh keystream duy nhat cho moi audio file.
    Audio name duoc hash vao counter -> khong tai su dung keystream.
    """

    audio_hash = hashlib.sha256(audio_name.encode()).digest()

    offset = int.from_bytes(audio_hash[:4], "big")

    counter = block_counter(frame_id, block_id) ^ offset

    ks = generate_keystream(
        MASTER_KEY,
        IV,
        counter
    )

    return ks.astype(np.uint8)


# ============================================================
# BER / NPCR / UACI
# ============================================================

def bit_error_rate(a, b):
    return np.mean(a != b)


def npcr(bits1, bits2):
    return np.mean(bits1 != bits2) * 100


def uaci(bits1, bits2):
    return np.mean(np.abs(bits1.astype(np.int16) - bits2.astype(np.int16))) * 100


# ============================================================
# KPA
# ============================================================

print("=" * 60)
print("KNOWN PLAINTEXT ATTACK (KPA)")
print("=" * 60)

key_A = generate_audio_keystream("Animal")
key_B = generate_audio_keystream("Drum")

error_kpa = key_A ^ key_B

ber_kpa = bit_error_rate(key_A, key_B)
npcr_kpa = npcr(key_A, key_B)
uaci_kpa = uaci(key_A, key_B)
corr_kpa = np.corrcoef(key_A, key_B)[0, 1]

# ============================================================
# CPA
# ============================================================

print("=" * 60)
print("CHOSEN PLAINTEXT ATTACK (CPA)")
print("=" * 60)

key_C = generate_audio_keystream("Silence")

error_cpa = key_C ^ key_B

ber_cpa = bit_error_rate(key_C, key_B)
npcr_cpa = npcr(key_C, key_B)
uaci_cpa = uaci(key_C, key_B)
corr_cpa = np.corrcoef(key_C, key_B)[0, 1]

# ============================================================
# SAVE CSV
# ============================================================

metrics = pd.DataFrame([
    {
        "Attack": "KPA",
        "BER_percent": ber_kpa * 100,
        "NPCR_percent": npcr_kpa,
        "UACI_percent": uaci_kpa,
        "Correlation": corr_kpa
    },
    {
        "Attack": "CPA",
        "BER_percent": ber_cpa * 100,
        "NPCR_percent": npcr_cpa,
        "UACI_percent": uaci_cpa,
        "Correlation": corr_cpa
    }
])

metrics.to_csv(
    RESULT_DIR / "kpa_cpa_metrics.csv",
    index=False
)

# ============================================================
# FIGURE STYLE
# ============================================================

plt.rcParams.update({
    "font.family": "Times New Roman",
    "font.size": 10
})


def draw_attack_figure(key_ref,
                       key_target,
                       error_map,
                       title1,
                       title2,
                       title3,
                       color1,
                       color2,
                       filename,
                       ber):

    fig, ax = plt.subplots(3, 1, figsize=(7.2, 5.5), sharex=True)

    x = np.arange(PLOT_BITS)

    ax[0].step(
        x,
        key_ref[:PLOT_BITS],
        where="post",
        color=color1,
        linewidth=1.2
    )

    ax[0].set_ylim(-0.1, 1.1)
    ax[0].set_ylabel("Bit")
    ax[0].set_title(title1, fontsize=10)

    ax[1].step(
        x,
        key_target[:PLOT_BITS],
        where="post",
        color=color2,
        linewidth=1.2
    )

    ax[1].set_ylim(-0.1, 1.1)
    ax[1].set_ylabel("Bit")
    ax[1].set_title(title2, fontsize=10)

    ax[2].fill_between(
        x,
        error_map[:PLOT_BITS],
        step="post",
        color="gray",
        alpha=0.8
    )

    ax[2].set_ylim(-0.1, 1.1)
    ax[2].set_ylabel("Error State")
    ax[2].set_xlabel("Keystream Bit Index")
    ax[2].set_title(f"{title3} (BER: {ber*100:.2f}%)", fontsize=10)

    for a in ax:
        a.grid(True, alpha=0.3)
        a.set_xlim(0, PLOT_BITS)

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / filename,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# FIGURE 6 - KPA
# ============================================================

draw_attack_figure(
    key_A,
    key_B,
    error_kpa,
    'Key A: Extracted from Known File "Animal"',
    'Key B: Actual Key for Target File "Drum"',
    "KPA Decryption Error Map",
    "blue",
    "red",
    "figure6_kpa_attack.png",
    ber_kpa
)

# ============================================================
# FIGURE 7 - CPA
# ============================================================

draw_attack_figure(
    key_C,
    key_B,
    error_cpa,
    'Key C: Derived from Chosen Input "All Zeros"',
    'Key B: Actual Key for Target File "Drum"',
    "CPA Decryption Error Map",
    "magenta",
    "red",
    "figure7_cpa_attack.png",
    ber_cpa
)

# ============================================================
# TERMINAL OUTPUT
# ============================================================

print("=" * 60)
print("KPA / CPA EVALUATION RESULTS")
print("=" * 60)

print(metrics.round(6).to_string(index=False))

print("=" * 60)
print("Saved CSV : results/kpa_cpa_metrics.csv")
print("Saved Fig : results/figure6_kpa_attack.png")
print("Saved Fig : results/figure7_cpa_attack.png")
print("=" * 60)