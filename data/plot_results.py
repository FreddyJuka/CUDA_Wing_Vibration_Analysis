#!/usr/bin/env python3
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os

# create folder for plots
plot_dir = "output/plots"
os.makedirs(plot_dir, exist_ok=True)

# Individual sensor spectrum
sensor_file = "output/spectra/sensor_001.csv_spectrum.csv"
if os.path.exists(sensor_file):
    df = pd.read_csv(sensor_file)
    plt.figure(figsize=(8,5))
    plt.plot(df["freq_hz"], df["magnitude"], color='blue')
    plt.title("Vibration Spectrum - sensor_001.csv")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/sensor_001_spectrum.png")
    plt.close()

# Dominant frequency per sensor
summary_file = "output/results_summary.txt"
if os.path.exists(summary_file):
    results = pd.read_csv(summary_file)
    plt.figure(figsize=(10,5))
    plt.bar(range(len(results)), results["dominant_freq_hz"], color='orange')
    plt.title("Dominant Frequency Detected per Sensor")
    plt.xlabel("Sensor index")
    plt.ylabel("Dominant Frequency [Hz]")
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/dominant_frequency_per_sensor.png")
    plt.close()

# Comparison of spectra (first 5 sensors)
files = sorted(glob.glob("output/spectra/*.csv"))[:5]
if len(files) > 0:
    plt.figure(figsize=(8,5))
    for f in files:
        df = pd.read_csv(f)
        label = os.path.basename(f).replace("_spectrum.csv","")
        plt.plot(df["freq_hz"], df["magnitude"], label=label)
    plt.title("Comparison of Vibration Spectra (first 5 sensors)")
    plt.xlabel("Frequency [Hz]")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{plot_dir}/comparison_first5_sensors.png")
    plt.close()

print(f"Plots saved in {plot_dir}/ folder.")
