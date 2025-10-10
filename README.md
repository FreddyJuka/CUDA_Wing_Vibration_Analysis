# CUDA Wing Vibration Analysis – Aerospace Application

## Overview
**CUDA_Wing_Vibration_Analysis** implements a **GPU-accelerated vibration signal analysis pipeline** using NVIDIA CUDA and cuFFT.  
The project performs **Fast Fourier Transform (FFT)** computations on *synthetic* vibration data from multiple virtual sensors along an aircraft wing.  

> **Note:** All signals were **synthetically generated** for this project to emulate wing responses under operational conditions; no real sensors were used.  

By leveraging GPU computing, the program can process hundreds of simulated sensor files in parallel, extract dominant frequencies, and generate detailed frequency spectra much faster than CPU-based methods.

This project is part of the **GPU Programming Specialization** from **Johns Hopkins University** and demonstrates high-performance signal analysis techniques relevant to **aerospace engineering**, such as modal analysis, flutter detection, and structural health monitoring.

---

## Key Features
- **GPU-accelerated FFT computation** using the NVIDIA cuFFT library.  
- **Automated batch processing** of multiple CSV sensor files.  
- **Dominant frequency detection** for each sensor in the 0–200 Hz range.  
- **Automated visualization** with Python (spectra, distributions, summaries).  
- **Cross-platform structure**, compatible with Linux, Windows (via WSL2), and NVIDIA GPUs with compute capability ≥ 5.0.

---

## Aerospace Engineering Context
In real-world aerospace applications, accelerometers on wings and fuselage components monitor **vibrational modes** and **structural resonances**.  
Dominant frequencies can indicate:
- **Aeroelastic phenomena**  
- **Structural fatigue**  
- **Early-stage damage or anomalies**  

A GPU-based FFT pipeline enables:
- Rapid processing of dozens or hundreds of sensor signals.  
- Near real-time frequency analysis for onboard or ground monitoring systems.  
- Foundations for high-frequency **modal analysis** and **flutter prediction**.

---

## Key Components

| File | Description |
|------|-------------|
| `main.cu` | Main C++/CUDA source handling data reading, GPU FFT, and dominant frequency detection. |
| `kernels.cu` | CUDA kernels for magnitude computation (`mag_r2c_kernel`) and other GPU operations. |
| `kernels.h` | Declares CUDA kernel prototypes and GPU-related constants. |
| `vibration.h` | Defines data structures and function prototypes for vibration signal management (sampling rate, signal length, frequency resolution). |
| `plot_results.py` | Generates frequency-domain plots, distributions, and summary visualizations. |
| `generate_data.py` | Creates synthetic 3-axis accelerometer data with added noise. |
| `run.sh` | Automation script: cleans, compiles, executes CUDA program, and runs Python plotting scripts. |

---

## Dependencies

### CUDA
- CUDA Toolkit ≥ 11.4  
- cuFFT library (included in CUDA Toolkit)  
- nvcc compiler  

### Python
- `numpy`  
- `pandas`  
- `matplotlib`  

---

## System Requirements

| Component | Requirement |
|-----------|------------|
| GPU       | NVIDIA GPU (Compute Capability ≥ 5.0) |
| OS        | Linux / Windows Subsystem for Linux (WSL2) |
| CPU       | x86_64 |
| RAM       | ≥ 8 GB recommended |
| Disk      | ~500 MB for generated data and spectra |

---

## How It Works

1. **Signal Generation (Simulated Sensors)**  
   Synthetic signals emulate wing accelerometer data at a defined sampling rate.

2. **Data Loading**  
   CSV files are read from `data/` and transferred to GPU memory.

3. **FFT Computation**  
   cuFFT transforms each signal into the frequency domain on the GPU.

4. **Frequency Analysis**  
   Dominant frequencies (typically 0–200 Hz) are extracted to identify vibration modes.

5. **Results Output**  
   Outputs include a summary table, per-sensor frequency spectra, and automated plots.

---

## Installation & Execution

### 1. Clone the Repository
bash
git clone https://github.com/<FreddyJuka>/CUDA_Wing_Vibration_Analysis.git
cd CUDA_Wing_Vibration_Analysis

## Windows Users: Setup WSL and Python Virtual Environment

sudo apt update
sudo apt install python3-venv python3-pip -y
python3 -m venv venv
source venv/bin/activate

    Python scripts (generate_data.py, plot_results.py) must be run within WSL for correct GPU and file access.

## Build and Run

./run.sh

This script performs:

- CUDA code compilation (make clean build).
- Synthetic data generation.
- GPU FFT analysis and frequency extraction.
- Automatic plotting of results.

Outputs:

- output/results_summary.txt
- output/spectra/
- output/plots/

## Example Output
            
Checking Python dependencies...
Generating synthetic vibration data...
100 files generated in data/sensors folder.
Compiling CUDA code...
rm -rf bin output
mkdir -p bin
nvcc -O3 -std=c++14 -Iinclude -o bin/vib_analysis src/main.cu src/kernels.cu -lcufft
Running GPU analysis...
100 files processed.
Total execution time (overall CPU + GPU): 1.39168 seconds
Total execution time (GPU only): 0.0311257 seconds
Estimated CPU-only time: 1.36056 seconds
Generating plots...
Plots saved in output/plots/ folder.

Execution completed successfully.
Results and plots are in the 'output' folder.

# Key Concepts Demonstrated

- GPU-accelerated FFT using cuFFT
- CUDA memory management and kernel launches
- Parallel data processing at scale
- Scientific signal analysis workflows
- Integration of GPU computation with Python visualization
- Cross-platform execution via WSL2

## Notes

All signals are synthetic, purely for demonstration and benchmarking.
The GPU timing metrics show performance improvements over CPU-only processing.
This repository serves as a template for high-performance vibration analysis in aerospace or mechanical systems.
