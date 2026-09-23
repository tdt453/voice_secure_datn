"""
experiments/network_qos_analysis.py

Danh gia QoS mang cho he thong Voice Secure.
Sinh bang CSV va bieu do 600 DPI.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

LATENCY_CSV = Path("logs/latency.csv")
PACKET_CSV = Path("logs/packet_log.csv")
RTT_CSV = Path("logs/rtt_log.csv")

latency = pd.read_csv(LATENCY_CSV)
packet = pd.read_csv(PACKET_CSV)

rtt = pd.read_csv(RTT_CSV) if RTT_CSV.exists() else None

# ==========================================================
# QoS Metrics
# ==========================================================
total_packets = len(packet)

crc_success = packet["CRC_OK"].sum()
crc_success_rate = crc_success / total_packets * 100

packet_loss_rate = 100 - crc_success_rate

payload_bytes = packet["PayloadBytes"].sum()

avg_payload = packet["PayloadBytes"].mean()

processing = latency["Processing_ms"]

processing_mean = processing.mean()
processing_max = processing.max()
processing_std = processing.std()

summary = {
    "Packets": total_packets,
    "CRC Success (%)": crc_success_rate,
    "Packet Loss (%)": packet_loss_rate,
    "Processing Mean (ms)": processing_mean,
    "Processing Std (ms)": processing_std,
    "Processing Max (ms)": processing_max,
    "Payload Bytes": payload_bytes,
    "Average Payload / Frame (Bytes)": avg_payload,
}

# ==========================================================
# RTT / Throughput / Jitter
# ==========================================================
if rtt is not None:

    rtt_valid = rtt["RTT_ms"].dropna()

    summary["RTT Mean (ms)"] = rtt_valid.mean()
    summary["RTT Max (ms)"] = rtt_valid.max()
    summary["RTT Std (ms)"] = rtt_valid.std()

    summary["Jitter Mean (ms)"] = rtt_valid.diff().abs().dropna().mean()

    throughput = payload_bytes / (rtt_valid.sum() / 1000) / 1024

    summary["Throughput (KB/s)"] = throughput

# ==========================================================
# Save CSV
# ==========================================================
pd.DataFrame(summary.items(), columns=["Metric", "Value"]).to_csv(
    RESULT_DIR / "network_qos_summary.csv",
    index=False
)

# ==========================================================
# Plot 1 : Processing Latency
# ==========================================================
plt.figure(figsize=(8, 3.2))
plt.plot(latency["FrameID"], latency["Processing_ms"], linewidth=1)
plt.xlabel("Frame ID")
plt.ylabel("Processing Latency (ms)")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULT_DIR / "processing_latency_qos.png",
            dpi=600,
            bbox_inches="tight")
plt.close()

# ==========================================================
# Plot 2 : RTT
# ==========================================================
if rtt is not None:

    plt.figure(figsize=(8, 3.2))
    plt.plot(rtt["FrameID"], rtt["RTT_ms"], linewidth=1)
    plt.xlabel("Frame ID")
    plt.ylabel("RTT (ms)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "network_rtt_qos.png",
                dpi=600,
                bbox_inches="tight")
    plt.close()

# ==========================================================
# Plot 3 : Throughput
# ==========================================================
if rtt is not None:

    cumulative_time = rtt["RTT_ms"].fillna(0).cumsum() / 1000

    cumulative_bytes = packet["PayloadBytes"].cumsum()

    throughput_series = cumulative_bytes / cumulative_time.replace(0, 1e-9) / 1024

    plt.figure(figsize=(8, 3.2))
    plt.plot(packet["FrameID"], throughput_series, linewidth=1.5)
    plt.xlabel("Frame ID")
    plt.ylabel("Throughput (KB/s)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "throughput_qos.png",
                dpi=600,
                bbox_inches="tight")
    plt.close()

# ==========================================================
# Plot 4 : Packet Delivery
# ==========================================================
plt.figure(figsize=(8, 3.2))
plt.scatter(packet["FrameID"], packet["CRC_OK"], s=6)
plt.yticks([0, 1], ["Lost", "Success"])
plt.xlabel("Frame ID")
plt.ylabel("Packet Status")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(RESULT_DIR / "packet_delivery_qos.png",
            dpi=600,
            bbox_inches="tight")
plt.close()

# ==========================================================
# Plot 5 : Jitter Histogram
# ==========================================================
if rtt is not None:

    jitter = rtt_valid.diff().abs().dropna()

    plt.figure(figsize=(6, 3.5))
    plt.hist(jitter, bins=25)
    plt.xlabel("Jitter (ms)")
    plt.ylabel("Frames")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(RESULT_DIR / "jitter_histogram.png",
                dpi=600,
                bbox_inches="tight")
    plt.close()

print("=" * 60)
print("NETWORK QoS SUMMARY")
print("=" * 60)

for k, v in summary.items():
    if isinstance(v, float):
        print(f"{k:32s}: {v:.4f}")
    else:
        print(f"{k:32s}: {v}")

print("=" * 60)
print("Saved figures -> results/")