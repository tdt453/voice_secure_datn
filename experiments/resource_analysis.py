"""
experiments/resource_analysis.py

Phan tich CPU, RAM, Temperature, Voltage cua Raspberry Pi.
Sinh bang tong hop va cac hinh 600 DPI.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULT_DIR = Path("results")

CSV_FILE = RESULT_DIR / "resource_latency.csv"

df = pd.read_csv(CSV_FILE)

# ======================================================
# Summary Table
# ======================================================
summary = pd.DataFrame({
    "Metric": [
        "CPU Mean (%)",
        "CPU Max (%)",
        "CPU Std (%)",
        "RAM Mean (%)",
        "RAM Max (%)",
        "Temperature Mean (C)",
        "Temperature Max (C)",
        "Temperature Std (C)",
        "CPU Frequency Mean (MHz)",
        "CPU Frequency Max (MHz)",
        "Voltage Mean (V)",
        "Voltage Min (V)"
    ],
    "Value": [
        df["CPU_percent"].mean(),
        df["CPU_percent"].max(),
        df["CPU_percent"].std(),
        df["RAM_percent"].mean(),
        df["RAM_percent"].max(),
        df["Temperature_C"].mean(),
        df["Temperature_C"].max(),
        df["Temperature_C"].std(),
        df["CPU_Freq_MHz"].mean(),
        df["CPU_Freq_MHz"].max(),
        df["Voltage_V"].mean(),
        df["Voltage_V"].min()
    ]
})

summary.to_csv(
    RESULT_DIR / "resource_summary.csv",
    index=False
)


# ======================================================
# Plot Helper
# ======================================================
def save_plot(y, ylabel, filename):

    plt.figure(figsize=(8, 3.2))

    plt.plot(
        df["Time_s"],
        df[y],
        linewidth=1.8
    )

    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)

    plt.grid(True, alpha=0.3)

    plt.tight_layout()

    plt.savefig(
        RESULT_DIR / filename,
        dpi=600,
        bbox_inches="tight"
    )

    plt.close()


save_plot("CPU_percent", "CPU Usage (%)", "cpu_usage.png")
save_plot("RAM_percent", "RAM Usage (%)", "ram_usage.png")
save_plot("Temperature_C", "Temperature (°C)", "temperature.png")
save_plot("CPU_Freq_MHz", "CPU Frequency (MHz)", "cpu_frequency.png")
save_plot("Voltage_V", "Voltage (V)", "voltage.png")

# ======================================================
# Combined Figure
# ======================================================
fig, axs = plt.subplots(4, 1, figsize=(8, 8), sharex=True)

axs[0].plot(df["Time_s"], df["CPU_percent"])
axs[0].set_ylabel("CPU (%)")
axs[0].grid(True, alpha=0.3)

axs[1].plot(df["Time_s"], df["RAM_percent"])
axs[1].set_ylabel("RAM (%)")
axs[1].grid(True, alpha=0.3)

axs[2].plot(df["Time_s"], df["Temperature_C"])
axs[2].set_ylabel("Temp (°C)")
axs[2].grid(True, alpha=0.3)

axs[3].plot(df["Time_s"], df["CPU_Freq_MHz"])
axs[3].set_ylabel("Freq (MHz)")
axs[3].set_xlabel("Time (s)")
axs[3].grid(True, alpha=0.3)

fig.tight_layout()

fig.savefig(
    RESULT_DIR / "resource_overview.png",
    dpi=600,
    bbox_inches="tight"
)

plt.close(fig)

print("=" * 60)
print("RESOURCE ANALYSIS SUMMARY")
print("=" * 60)
print(summary.to_string(index=False))
print("=" * 60)
print("Saved figures to results/")