"""
experiments/resource_latency_monitor.py

Theo doi tai nguyen Raspberry Pi khi truyen audio.
Sinh CSV de phan tich offline.
"""

import csv
import subprocess
import time
from pathlib import Path

import psutil

RESULT_DIR = Path("results")
RESULT_DIR.mkdir(exist_ok=True)

CSV_FILE = RESULT_DIR / "resource_latency.csv"

DURATION = 30        # seconds
INTERVAL = 0.2       # 200 ms


def get_temperature():
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_temp"],
            text=True
        )
        return float(out.split("=")[1].replace("'C\n", ""))
    except:
        return 0.0


def get_clock():
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_clock", "arm"],
            text=True
        )
        return int(out.split("=")[1]) / 1e6
    except:
        return 0.0


def get_voltage():
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_volts"],
            text=True
        )
        return float(out.split("=")[1].replace("V\n", ""))
    except:
        return 0.0


with open(CSV_FILE, "w", newline="") as f:

    writer = csv.writer(f)

    writer.writerow([
        "Time_s",
        "CPU_percent",
        "RAM_percent",
        "Temperature_C",
        "CPU_Freq_MHz",
        "Voltage_V"
    ])

    t0 = time.time()

    while time.time() - t0 < DURATION:

        writer.writerow([
            round(time.time() - t0, 3),
            psutil.cpu_percent(interval=None),
            psutil.virtual_memory().percent,
            get_temperature(),
            get_clock(),
            get_voltage()
        ])

        time.sleep(INTERVAL)

print("=" * 60)
print("RESOURCE MONITOR FINISHED")
print("=" * 60)
print(f"Saved -> {CSV_FILE}")