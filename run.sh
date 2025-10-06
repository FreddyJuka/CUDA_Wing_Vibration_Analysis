#!/bin/bash

# Default parameters
NFILES=100
NSAMPLES=10000
FS=1000
DATA_DIR="data/sensors"
OUT_DIR="output"

mkdir -p ${DATA_DIR}
mkdir -p ${OUT_DIR}/spectra

# Generate data
./venv/bin/python3 data/generate_data.py --nfiles ${NFILES} --nsamples ${NSAMPLES} --fs ${FS} --out ${DATA_DIR}

# Compile
make

# Run the analysis
bin/vib_analysis --data ${DATA_DIR} --out ${OUT_DIR} --fs ${FS} --nfiles ${NFILES}

echo "Results in output folder."
echo "Generating plots..."
./venv/bin/python3 data/plot_results.py
echo "Done."
