"""
experiments/plot_avalanche.py
"""

import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv("results/avalanche_results.csv")

plt.figure(figsize=(8,4))

plt.plot(
    df["KeyBit"],
    df["AvalanchePercent"],
    linewidth=1
)

plt.xlabel("Flipped Master-Key Bit")
plt.ylabel("Avalanche (%)")
plt.grid(True)

plt.tight_layout()

plt.savefig(
    "results/avalanche_curve.png",
    dpi=600
)

plt.close()

print("Saved -> results/avalanche_curve.png")