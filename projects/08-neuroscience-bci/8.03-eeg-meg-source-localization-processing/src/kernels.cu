// ===========================================================================
// src/kernels.cu  --  cuFFT batched FFT + power kernel
// ---------------------------------------------------------------------------
// Project 8.03 : EEG/MEG Spectral Processing (cuFFT)
//
// GPU twin of dft_power_cpu(): cuFFT computes the same X[k] as the naive DFT,
// then a tiny kernel takes |X|^2. main.cu compares the resulting band powers.
// See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <cufft.h>
#include <cstdio>
#include <cstdlib>

// cuFFT has its own status type, so it needs its own check macro (mirrors
// CUDA_CHECK but for cufftResult). Every cuFFT call is guarded and explained.
#define CUFFT_CHECK(call)                                                       \
    do {                                                                        \
        cufftResult st__ = (call);                                             \
        if (st__ != CUFFT_SUCCESS) {                                           \
            std::fprintf(stderr, "[CUFFT_CHECK] %s:%d -> error %d\n",          \
                         __FILE__, __LINE__, static_cast<int>(st__));          \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                       \
    } while (0)

// power_kernel: |X|^2 / N^2 for each complex output bin. cufftComplex IS float2
// (.x = real, .y = imaginary), so we read it as float2. One thread per bin.
__global__ void power_kernel(const float2* __restrict__ X, int total, float invN2,
                             float* __restrict__ power) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < total) {
        const float2 v = X[i];
        power[i] = (v.x * v.x + v.y * v.y) * invN2;
    }
}

void spectrum_gpu(const EegData& d, std::vector<float>& power, float* kernel_ms) {
    const int n = d.n, nch = d.n_ch;
    const int nf = n / 2 + 1;             // R2C output length per channel
    const int total = nch * nf;
    power.assign(total, 0.0f);

    // Device buffers. cufftReal == float, cufftComplex == float2.
    cufftReal*    d_in    = nullptr;      // [nch*n]  real input
    cufftComplex* d_out   = nullptr;      // [nch*nf] complex spectra
    float*        d_power = nullptr;      // [total]  power spectrum
    CUDA_CHECK(cudaMalloc(&d_in, static_cast<std::size_t>(nch) * n * sizeof(cufftReal)));
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(nch) * nf * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_power, static_cast<std::size_t>(total) * sizeof(float)));
    // d.x is float, which is exactly cufftReal -> a straight copy.
    CUDA_CHECK(cudaMemcpy(d_in, d.x.data(),
                          static_cast<std::size_t>(nch) * n * sizeof(cufftReal),
                          cudaMemcpyHostToDevice));

    // ---- The library call, NOT a black box -------------------------------
    // cufftPlan1d(plan, n, CUFFT_R2C, batch) builds a plan for `batch` independent
    // length-n real-to-complex FFTs laid out contiguously: input stride n, output
    // stride n/2+1. cufftExecR2C then computes, for each channel c and bin k:
    //     X_c[k] = sum_{t=0}^{n-1} x_c[t] * exp(-2*pi*i*k*t/n),  k = 0..n/2
    // (the same sum dft_power_cpu does by hand, via the O(n log n) FFT). R2C only
    // stores the non-redundant half (Hermitian symmetry of a real signal's FFT).
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan1d(&plan, n, CUFFT_R2C, nch));

    const int block = 256;
    const int grid = (total + block - 1) / block;
    const float invN2 = 1.0f / (static_cast<float>(n) * static_cast<float>(n));

    GpuTimer timer;
    timer.start();
    CUFFT_CHECK(cufftExecR2C(plan, d_in, d_out));           // the batched FFT
    power_kernel<<<grid, block>>>(reinterpret_cast<const float2*>(d_out),
                                  total, invN2, d_power);   // |X|^2 / N^2
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("power_kernel");

    CUDA_CHECK(cudaMemcpy(power.data(), d_power,
                          static_cast<std::size_t>(total) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    cufftDestroy(plan);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_power));
}
