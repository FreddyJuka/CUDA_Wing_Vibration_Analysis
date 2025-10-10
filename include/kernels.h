#ifndef KERNELS_H
#define KERNELS_H

#include <cuda_runtime.h>
#include <cufft.h>

/**
 * @brief Kernel to compute magnitude of a complex spectrum
 *        (FFT), where each element has real (x) and imaginary (y) components.
 *
 * @param freq Input complex vector (length n)
 * @param mag Output vector containing magnitudes
 * @param nfreq Number of frequencies to process
 */
__global__ void magnitude_kernel(const cufftComplex* __restrict__ freq, float* __restrict__ mag, int nfreq);

/**
 * @brief Specialized kernel for real-input FFTs (CUFFT_R2C),
 *        computes magnitude of N/2+1 frequencies.
 */
__global__ void mag_r2c_kernel(const cufftComplex* __restrict__ freq, float* __restrict__ mag, int nfreq);

#endif // KERNELS_H
