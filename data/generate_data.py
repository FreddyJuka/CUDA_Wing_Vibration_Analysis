#!/usr/bin/env python3

"""
Synthetic accelerometer-like data generator.
Each CSV file contains: time, acc_x, acc_y, acc_z
Simulates modes with multiple dominant frequencies and added noise.
"""
import argparse
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--nfiles', type=int, default=100)
parser.add_argument('--nsamples', type=int, default=10000)
parser.add_argument('--fs', type=float, default=1000.0)
parser.add_argument('--out', type=str, default='data/sensors')
args = parser.parse_args()

os.makedirs(args.out, exist_ok=True)

t = np.arange(args.nsamples) / args.fs

# generate nfiles with different dominant frequencies
for i in range(1, args.nfiles+1):
    # select 1-3 modes (frequencies in 5–200 Hz, reasonable for vibrations)
    nmodes = np.random.choice([1,2,3], p=[0.6,0.3,0.1])
    freqs = np.random.uniform(5, 200, size=nmodes)
    amps = np.random.uniform(0.05, 0.5, size=nmodes)
    phase = np.random.uniform(0, 2*np.pi, size=nmodes)

    # generate base signal for each axial component with slight variations
    sig_x = np.zeros_like(t)
    sig_y = np.zeros_like(t)
    sig_z = np.zeros_like(t)

    for f,a,ph in zip(freqs, amps, phase):
        sig_x += a * np.sin(2*np.pi*f*t + ph)
        sig_y += (a*0.8) * np.sin(2*np.pi*(f*1.01)*t + ph*1.1)
        sig_z += (a*0.6) * np.sin(2*np.pi*(f*0.99)*t + ph*0.9)

    # add white noise and slow drift
    noise_level = np.random.uniform(0.005, 0.02)
    sig_x += noise_level * np.random.randn(len(t))
    sig_y += noise_level * np.random.randn(len(t))
    sig_z += noise_level * np.random.randn(len(t))

    drift = 0.0001 * np.sin(2*np.pi*0.1*t)
    sig_x += drift
    sig_y += drift*0.5
    sig_z += drift*0.2

    data = np.vstack([t, sig_x, sig_y, sig_z]).T
    fname = os.path.join(args.out, f'sensor_{i:03d}.csv')
    header = 'time,acc_x,acc_y,acc_z'
    np.savetxt(fname, data, delimiter=',', header=header, comments='')

print(f'{args.nfiles} files generated in {args.out} folder.')
