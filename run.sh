#!/bin/bash

# Default parameters
NFILES=100
NSAMPLES=10000
FS=1000
DATA_DIR="data/sensors"
OUT_DIR="output"

mkdir -p ${DATA_DIR}
mkdir -p ${OUT_DIR}/spectra

echo "=========================================================="
echo "              CUDA Wing Vibration Analysis                "
echo "=========================================================="

# detect venv environment 
if [ -d "venv" ]; then
    PYTHON="./venv/bin/python3"
else
    PYTHON="python3"
fi

# check and install python dependencies
echo "Checking Python dependencies..."
$PYTHON -m pip install --quiet --upgrade pip
$PYTHON -m pip install --quiet -r requirements.txt

# generate synthetic data
echo "Generating synthetic vibration data..."
$PYTHON data/generate_data.py --nfiles ${NFILES} --nsamples ${NSAMPLES} --fs ${FS} --out ${DATA_DIR}

# compile
echo "Compiling CUDA code..."
make clean build || make

# run
echo "Running GPU analysis..."
bin/vib_analysis --data ${DATA_DIR} --out ${OUT_DIR} --fs ${FS} --nfiles ${NFILES}

# post-process
echo "Generating plots..."
$PYTHON data/plot_results.py

echo "=========================================================="
echo "Execution completed successfully."
echo "Results and plots are in the 'output' folder."
echo "=========================================================="
