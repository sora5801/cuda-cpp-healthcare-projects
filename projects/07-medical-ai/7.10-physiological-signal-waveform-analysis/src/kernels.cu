// ===========================================================================
// src/kernels.cu  --  Shared-memory tiled 1-D convolution kernel + wrapper
// ---------------------------------------------------------------------------
// Project 7.10 : Physiological Signal & Waveform Analysis
//
// GPU twin of conv1d_cpu(): same math, but each block stages its input window
// in shared memory once. main.cu runs both and checks they agree.
// See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <cstdio>
#include <cstdlib>

// Filter taps in CONSTANT memory: tiny, read by every thread, never change
// during the launch -> the constant cache broadcasts a tap to a whole warp.
__constant__ float c_h[CONV_K_MAX];

// ---------------------------------------------------------------------------
// conv1d_kernel: one thread per output sample, reading from a shared tile.
//   Tile layout: tile[j] holds input x[blockStart - halo + j]. The block loads
//   blockDim.x "main" samples plus `halo` extra on each side (the HALO), with
//   zeros past the signal ends. After __syncthreads, thread t computes
//   y = sum_k h[k] * tile[t + k]  (which equals sum_k h[k] * x[n - halo + k]).
// ---------------------------------------------------------------------------
__global__ void conv1d_kernel(const float* __restrict__ x, int n, int K, int halo,
                              float* __restrict__ y) {
    extern __shared__ float tile[];          // size = blockDim.x + 2*halo
    const int t = threadIdx.x;
    const int start = blockIdx.x * blockDim.x;
    const int gi = start + t;                 // this thread's output index

    // 1) Load the "main" sample for this thread into the tile interior.
    tile[t + halo] = (gi < n) ? x[gi] : 0.0f;

    // 2) The first `halo` threads also load the left and right halo samples.
    //    (Requires halo <= blockDim.x, which the host wrapper guarantees.)
    if (t < halo) {
        const int li = start - halo + t;                 // left halo source
        tile[t] = (li >= 0 && li < n) ? x[li] : 0.0f;
        const int ri = start + blockDim.x + t;           // right halo source
        tile[blockDim.x + halo + t] = (ri < n) ? x[ri] : 0.0f;
    }
    __syncthreads();                          // tile fully loaded before reads

    // 3) Convolve from shared memory (filter from constant memory).
    if (gi < n) {
        float acc = 0.0f;
        for (int k = 0; k < K; ++k) acc += c_h[k] * tile[t + k];
        y[gi] = acc;
    }
}

void conv1d_gpu(const Signal& s, const std::vector<float>& h,
                std::vector<float>& y, float* kernel_ms) {
    const int n = s.n;
    const int K = static_cast<int>(h.size());
    const int halo = (K - 1) / 2;
    y.assign(n, 0.0f);

    // Sanity: the constant buffer and the halo-loading scheme have fixed limits.
    if (K > CONV_K_MAX || halo > CONV_BLOCK) {
        std::fprintf(stderr, "[conv1d_gpu] filter too long (K=%d): max %d, halo<=%d\n",
                     K, CONV_K_MAX, CONV_BLOCK);
        std::exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaMemcpyToSymbol(c_h, h.data(), K * sizeof(float)));

    float *d_x = nullptr, *d_y = nullptr;
    CUDA_CHECK(cudaMalloc(&d_x, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_y, n * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_x, s.x.data(), n * sizeof(float), cudaMemcpyHostToDevice));

    const int block = CONV_BLOCK;
    const int grid = (n + block - 1) / block;
    const std::size_t shmem = static_cast<std::size_t>(block + 2 * halo) * sizeof(float);

    GpuTimer timer;
    timer.start();
    conv1d_kernel<<<grid, block, shmem>>>(d_x, n, K, halo, d_y);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("conv1d_kernel");

    CUDA_CHECK(cudaMemcpy(y.data(), d_y, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_x));
    CUDA_CHECK(cudaFree(d_y));
}
