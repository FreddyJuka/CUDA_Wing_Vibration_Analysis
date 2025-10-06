#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <cmath>
#include <dirent.h>
#include <sys/stat.h>
#include <algorithm>
#include <chrono>      

#include <cuda_runtime.h>
#include <cufft.h>

#include "../include/vibration.h"
#include "../include/kernels.h"

// -----------------------------------------------------------------------------
// Error checking macros
// -----------------------------------------------------------------------------
#define CUDA_CHECK(call) \
    do { cudaError_t e = (call); if (e != cudaSuccess) { \
        std::cerr << "CUDA error: " << cudaGetErrorString(e) \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        return 1; } } while(0)

#define CUFFT_CHECK(call) \
    do { cufftResult r = (call); if (r != CUFFT_SUCCESS) { \
        std::cerr << "CUFFT error: " << r \
                  << " at " << __FILE__ << ":" << __LINE__ << "\n"; \
        return 1; } } while(0)

// -----------------------------------------------------------------------------
// CSV reading: computes acceleration magnitude (sqrt(ax² + ay² + az²))
// -----------------------------------------------------------------------------
std::vector<float> read_csv_magnitude(const std::string &path, int &nsamples) {
    std::ifstream in(path);
    std::string line;
    std::vector<float> mag;

    if (!std::getline(in, line)) return mag; // skip header
    while (std::getline(in, line)) {
        std::stringstream ss(line);
        std::string item;
        std::vector<float> row;
        while (std::getline(ss, item, ',')) row.push_back(std::stof(item));
        if (row.size() >= 4) {
            float ax = row[1];
            float ay = row[2];
            float az = row[3];
            mag.push_back(std::sqrt(ax * ax + ay * ay + az * az));
        }
    }
    nsamples = static_cast<int>(mag.size());
    return mag;
}

// -----------------------------------------------------------------------------
// List CSV files in a directory
// -----------------------------------------------------------------------------
std::vector<std::string> list_csv(const std::string &dir) {
    std::vector<std::string> files;
    DIR *dp;
    struct dirent *entry;
    dp = opendir(dir.c_str());
    if (dp == NULL) return files;
    while ((entry = readdir(dp))) {
        std::string name = entry->d_name;
        if (name.size() > 4 && name.substr(name.size() - 4) == ".csv") {
            files.push_back(dir + "/" + name);
        }
    }
    closedir(dp);
    std::sort(files.begin(), files.end());
    return files;
}

// -----------------------------------------------------------------------------
// Next power of 2 (for FFT)
// -----------------------------------------------------------------------------
int next_pow2(int v) {
    int p = 1;
    while (p < v) p <<= 1;
    return p;
}

// -----------------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------------
int main(int argc, char** argv) {
    // Total time (CPU + GPU)
    auto t_start = std::chrono::high_resolution_clock::now();

    // GPU acc time
    float gpu_total_milliseconds = 0.0f;

    // Default options
    ProgramOptions opt;
    opt.data_dir = "data/sensors";
    opt.out_dir = "output";
    opt.nfiles = 100;
    opt.fs = 1000;

    // Argument parsing
    for (int i = 1; i < argc; i++) {
        std::string s = argv[i];
        if (s == "--data") opt.data_dir = argv[++i];
        else if (s == "--out") opt.out_dir = argv[++i];
        else if (s == "--fs") opt.fs = atoi(argv[++i]);
        else if (s == "--nfiles") opt.nfiles = atoi(argv[++i]);
    }

    auto files = list_csv(opt.data_dir);
    if (files.empty()) {
        std::cerr << "No CSV files found in " << opt.data_dir << "\n";
        return 1;
    }

    mkdir(opt.out_dir.c_str(), 0755);
    mkdir((opt.out_dir + "/spectra").c_str(), 0755);

    std::ofstream summary((opt.out_dir + "/results_summary.txt").c_str());
    summary << "sensor,nsamples,dominant_freq_hz\n";

    int processed = 0;

    // CUDA events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    for (size_t fi = 0; fi < files.size() && processed < opt.nfiles; ++fi) {
        int nsamples = 0;
        auto mag = read_csv_magnitude(files[fi], nsamples);
        if (nsamples == 0) continue;

        // Pad to next power of 2
        int N = next_pow2(nsamples);
        if (N > 16384) N = 16384;
        std::vector<float> signal(N, 0.0f);
        for (int i = 0; i < nsamples; i++) signal[i] = mag[i];

        // Allocate memory on GPU
        float *d_signal = nullptr;
        cufftComplex *d_freq = nullptr;
        float *d_mag = nullptr;

        CUDA_CHECK(cudaMalloc((void**)&d_signal, sizeof(float) * N));
        int nfreq = N / 2 + 1;
        CUDA_CHECK(cudaMalloc((void**)&d_freq, sizeof(cufftComplex) * nfreq));
        CUDA_CHECK(cudaMalloc((void**)&d_mag, sizeof(float) * nfreq));

        CUDA_CHECK(cudaMemcpy(d_signal, signal.data(),
                              sizeof(float) * N, cudaMemcpyHostToDevice));

        // FFT plan
        cufftHandle plan;
        CUFFT_CHECK(cufftPlan1d(&plan, N, CUFFT_R2C, 1));

        // ------------------------------------------------------------
        // GPU timing for this file
        // ------------------------------------------------------------
        CUDA_CHECK(cudaEventRecord(start));
        CUFFT_CHECK(cufftExecR2C(plan, d_signal, d_freq));
        int threads = 256;
        int blocks = (nfreq + threads - 1) / threads;
        mag_r2c_kernel<<<blocks, threads>>>(d_freq, d_mag, nfreq);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float gpu_milliseconds_file = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&gpu_milliseconds_file, start, stop));
        gpu_total_milliseconds += gpu_milliseconds_file;
        // ------------------------------------------------------------

        // Copy result to host
        std::vector<float> host_mag(nfreq);
        CUDA_CHECK(cudaMemcpy(host_mag.data(), d_mag,
                              sizeof(float) * nfreq, cudaMemcpyDeviceToHost));

        // Normalization
        for (int k = 0; k < nfreq; ++k)
            host_mag[k] /= static_cast<float>(N);

        // Search dominant frequency in 5–200 Hz
        float freq_res = static_cast<float>(opt.fs) / static_cast<float>(N);
        int min_bin = static_cast<int>(5.0f / freq_res);
        int max_bin = static_cast<int>(200.0f / freq_res);
        if (max_bin >= nfreq) max_bin = nfreq - 1;

        int peak_idx = min_bin;
        float peak_val = host_mag[min_bin];
        for (int k = min_bin + 1; k <= max_bin; ++k) {
            if (host_mag[k] > peak_val) {
                peak_val = host_mag[k];
                peak_idx = k;
            }
        }
        float dominant_hz = peak_idx * freq_res;

        // Save full spectrum
        std::string base = files[fi].substr(files[fi].find_last_of("/\\") + 1);
        std::ofstream sp((opt.out_dir + "/spectra/" + base + "_spectrum.csv").c_str());
        sp << "freq_hz,magnitude\n";
        for (int k = 0; k < nfreq; ++k)
            sp << (k * freq_res) << "," << host_mag[k] << "\n";
        sp.close();

        // Save summary
        summary << base << "," << nsamples << "," << dominant_hz << "\n";

        // Free memory
        CUDA_CHECK(cudaFree(d_signal));
        CUDA_CHECK(cudaFree(d_freq));
        CUDA_CHECK(cudaFree(d_mag));
        CUFFT_CHECK(cufftDestroy(plan));

        processed++;
    }

    // Destroy CUDA events
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    summary.close();
    std::cout << processed << " files processed.\n";

    // Total CPU + GPU time
    auto t_end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_total = t_end - t_start;

    // GPU time
    float gpu_time_sec = gpu_total_milliseconds / 1000.0f;

    std::cout << "Total execution time (overall CPU + GPU): " << elapsed_total.count() << " seconds\n";
    std::cout << "Total execution time (GPU only): " << gpu_time_sec << " seconds\n";
    std::cout << "Estimated CPU-only time: " << (elapsed_total.count() - gpu_time_sec) << " seconds\n";

    return 0;
}