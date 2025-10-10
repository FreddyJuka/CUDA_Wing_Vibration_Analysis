#include "../include/kernels.h"
#include <math.h>

/**
 * @brief Generic kernel to compute magnitude of a complex vector.
 */
__global__ void magnitude_kernel(const cufftComplex* __restrict__ freq,
                                 float* __restrict__ mag, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        float re = freq[idx].x;
        float im = freq[idx].y;
        mag[idx] = sqrtf(re * re + im * im);
    }
}

/**
 * @brief Kernel for CUFFT_R2C output (only N/2+1 frequencies).
 */
__global__ void mag_r2c_kernel(const cufftComplex* __restrict__ freq,
                               float* __restrict__ mag, int nfreq) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < nfreq) {
        float re = freq[idx].x;
        float im = freq[idx].y;
        mag[idx] = sqrtf(re * re + im * im);
    }
}