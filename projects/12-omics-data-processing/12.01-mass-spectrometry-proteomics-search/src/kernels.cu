// ===========================================================================
// src/kernels.cu  --  Cosine spectral-search kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 12.01 : Mass-Spectrometry Proteomics Search
//
// GPU twin of cosine_cpu(): one thread per library spectrum, the query in
// constant memory. main.cu runs both and compares scores. See ../THEORY.md.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <cstdio>
#include <cstdlib>

// The query spectrum in CONSTANT memory: read by every thread, never written
// during the launch -> the constant cache broadcasts each bin warp-wide. Sized
// at MAX_BINS (1024 floats = 4 KB, well within the 64 KB constant bank).
__constant__ float c_query[MAX_BINS];

static constexpr int THREADS_PER_BLOCK = 256;

// One thread computes the cosine score for one library spectrum.
//   cosine = dot(query, lib_i) / (||query|| * ||lib_i||).
// The dot product accumulates in double (matching the CPU reference); the norms
// are precomputed once on the host.
__global__ void cosine_kernel(const float* __restrict__ lib, const double* __restrict__ libnorm,
                              int N, int bins, double qnorm, float* __restrict__ scores) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const float* row = lib + static_cast<std::size_t>(i) * bins;
    double dot = 0.0;
    for (int b = 0; b < bins; ++b)
        dot += static_cast<double>(c_query[b]) * static_cast<double>(row[b]);
    const double denom = qnorm * libnorm[i];
    scores[i] = (denom > 0.0) ? static_cast<float>(dot / denom) : 0.0f;
}

void cosine_gpu(const SpectralData& s, double qnorm, const std::vector<double>& libnorm,
                std::vector<float>& scores, float* kernel_ms) {
    const int N = s.N, bins = s.bins;
    scores.assign(N, 0.0f);
    if (bins > MAX_BINS) {
        std::fprintf(stderr, "[cosine_gpu] bins=%d exceeds MAX_BINS=%d\n", bins, MAX_BINS);
        std::exit(EXIT_FAILURE);
    }

    // Upload the query to constant memory.
    CUDA_CHECK(cudaMemcpyToSymbol(c_query, s.query.data(), bins * sizeof(float)));

    float* d_lib = nullptr; double* d_libnorm = nullptr; float* d_scores = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lib, s.lib.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_libnorm, static_cast<std::size_t>(N) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_scores, static_cast<std::size_t>(N) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_lib, s.lib.data(), s.lib.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_libnorm, libnorm.data(), static_cast<std::size_t>(N) * sizeof(double),
                          cudaMemcpyHostToDevice));

    const int grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    cosine_kernel<<<grid, THREADS_PER_BLOCK>>>(d_lib, d_libnorm, N, bins, qnorm, d_scores);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("cosine_kernel");

    CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, static_cast<std::size_t>(N) * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_lib));
    CUDA_CHECK(cudaFree(d_libnorm));
    CUDA_CHECK(cudaFree(d_scores));
}
