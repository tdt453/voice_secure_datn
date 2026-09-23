"""
experiments/latency_breakdown.py

Danh gia do tre xu ly va RTT.
Chay duoc tren RX (khong co RTT) hoac RX sau khi copy RTT tu TX.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

LATENCY_CSV = Path("logs/latency.csv")
RTT_CSV = Path("logs/rtt_log.csv")

# ======================================================
# Processing latency (RX)
# ======================================================
latency = pd.read_csv(LATENCY_CSV)

summary = [
    ["Processing Mean (ms)", latency["Processing_ms"].mean()],
    ["Processing Max (ms)", latency["Processing_ms"].max()],
    ["Processing Min (ms)", latency["Processing_ms"].min()],
    ["Processing Std (ms)", latency["Processing_ms"].std()],
]

# ======================================================
# Plot Processing Latency
# ======================================================
plt.figure(figsize=(8,3.2))
plt.plot(latency["FrameID"], latency["Processing_ms"], linewidth=1)
plt.xlabel("Frame ID")
plt.ylabel("Processing Latency (ms)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULT_DIR/"processing_latency.png", dpi=600, bbox_inches="tight")
plt.close()

# Histogram
plt.figure(figsize=(6,3.5))
plt.hist(latency["Processing_ms"], bins=30)
plt.xlabel("Processing Latency (ms)")
plt.ylabel("Frames")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULT_DIR/"processing_latency_hist.png", dpi=600, bbox_inches="tight")
plt.close()

# ======================================================
# RTT (neu co file tu TX)
# ======================================================
if RTT_CSV.exists():

    rtt = pd.read_csv(RTT_CSV)
    rtt_valid = rtt["RTT_ms"].dropna()

    summary.extend([
        ["RTT Mean (ms)", rtt_valid.mean()],
        ["RTT Max (ms)", rtt_valid.max()],
        ["RTT Min (ms)", rtt_valid.min()],
        ["RTT Std (ms)", rtt_valid.std()],
    ])

    plt.figure(figsize=(8,3.2))
    plt.plot(rtt["FrameID"], rtt["RTT_ms"], linewidth=1)
    plt.xlabel("Frame ID")
    plt.ylabel("Round Trip Time (ms)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR/"network_rtt.png", dpi=600, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6,3.5))
    plt.hist(rtt_valid, bins=30)
    plt.xlabel("RTT (ms)")
    plt.ylabel("Frames")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR/"network_rtt_hist.png", dpi=600, bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(5,4))
    plt.boxplot(
        [latency["Processing_ms"], rtt_valid],
        tick_labels=["Processing", "RTT"]
    )
    plt.ylabel("Latency (ms)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR/"latency_boxplot.png", dpi=600, bbox_inches="tight")
    plt.close()

    print("RTT log found -> RTT figures generated.")

else:
    print("RTT log not found. Only processing latency is analyzed.")

# ======================================================
# Save CSV
# ======================================================
summary_df = pd.DataFrame(summary, columns=["Metric","Value"])
summary_df.to_csv(
    RESULT_DIR/"latency_breakdown.csv",
    index=False
)

print("="*60)
print(summary_df.to_string(index=False))
print("="*60)
print("Saved -> results/latency_breakdown.csv")