// ===========================================================================
// src/kernels.cu  --  DECT decomposition kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 4.20 : Dual-Energy / Spectral CT Reconstruction
//
// This is the GPU twin of decompose_cpu() in reference_cpu.cpp. main.cu runs
// both and asserts they agree. Because both call the SAME __host__ __device__
// core decompose_bin() (dect.h), the agreement is EXACT (bit-for-bit), not
// approximate. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "dect.h"                // SpectralModel, decompose_bin, DecompResult
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

#include <cstddef>

// ---------------------------------------------------------------------------
// The scanner physics in CONSTANT memory.
//   * The SpectralModel (two spectra + two attenuation curves, a few hundred
//     bytes) is identical for every bin and never written during the launch ->
//     constant memory is the textbook fit: its broadcast cache serves the whole
//     warp from one load, so the Newton inner loop's repeated reads of mu1/mu2/
//     w_lo/w_hi are nearly free.
//   * Filled once per launch by cudaMemcpyToSymbol() in decompose_gpu().
//   * sizeof(SpectralModel) = 5 * NUM_ENERGIES * 8 bytes ~= 960 B, far under the
//     64 KB constant bank.
// ---------------------------------------------------------------------------
__constant__ SpectralModel c_model;

// 128 threads/block: a multiple of the 32-lane warp. This kernel is register-
// and math-heavy (each thread runs several Newton iterations, each looping over
// NUM_ENERGIES with transcendental exp/log), so a moderate block size keeps
// occupancy healthy without over-subscribing registers on sm_75..sm_89. See
// THEORY "GPU mapping" for the occupancy reasoning.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// decompose_kernel: one logical thread per sinogram bin, via a grid-stride loop
// so a fixed-size grid covers an arbitrarily large sinogram.
//   Thread (blockIdx.x, threadIdx.x) starts at i = block*blockDim + thread and
//   strides by the total thread count until i >= n.
//   Per bin: read (m_lo,m_hi) from global memory, build the linear seed inline
//   (the same closed-form as reference_cpu.cpp::linear_init, replicated here so
//   the device needs no host call), then run the shared Newton core. Outputs are
//   fully independent -> no shared memory, no atomics, no races.
//   Memory: c_model from constant cache; d_m_lo/d_m_hi rows from global memory
//   (coalesced: consecutive threads read consecutive bins).
// ---------------------------------------------------------------------------
__global__ void decompose_kernel(const double* __restrict__ d_m_lo,
                                 const double* __restrict__ d_m_hi,
                                 int n,
                                 double* __restrict__ d_t1,
                                 double* __restrict__ d_t2,
                                 int* __restrict__ d_iters) {
    const int stride = blockDim.x * gridDim.x;             // total threads in grid
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        const double m_lo = d_m_lo[i];
        const double m_hi = d_m_hi[i];

        // --- Linear seed (identical closed form to host linear_init) ----------
        // Spectrum-averaged ("effective") attenuation coefficients of the two
        // materials for each spectrum. These read c_model straight from the
        // constant cache. Kept in a small local loop so the device does not
        // depend on any host-only function.
        double a_lo_1 = 0.0, a_lo_2 = 0.0, a_hi_1 = 0.0, a_hi_2 = 0.0;
        #pragma unroll
        for (int k = 0; k < NUM_ENERGIES; ++k) {
            a_lo_1 += c_model.w_lo[k] * c_model.mu1[k];
            a_lo_2 += c_model.w_lo[k] * c_model.mu2[k];
            a_hi_1 += c_model.w_hi[k] * c_model.mu1[k];
            a_hi_2 += c_model.w_hi[k] * c_model.mu2[k];
        }
        double det = a_lo_1 * a_hi_2 - a_lo_2 * a_hi_1;
        if (fabs(det) < 1e-12) det = (det < 0.0 ? -1e-12 : 1e-12);
        double s1 = ( a_hi_2 * m_lo - a_lo_2 * m_hi) / det;
        double s2 = (-a_hi_1 * m_lo + a_lo_1 * m_hi) / det;
        if (s1 < 0.0) s1 = 0.0;
        if (s2 < 0.0) s2 = 0.0;

        // --- Newton solve via the SHARED core -> bit-identical to the CPU -----
        const DecompResult r = decompose_bin(m_lo, m_hi, c_model, s1, s2,
                                             MAX_NEWTON_ITER, NEWTON_TOL);
        d_t1[i]    = r.t1;
        d_t2[i]    = r.t2;
        d_iters[i] = r.iters;
    }
}

// ---------------------------------------------------------------------------
// decompose_gpu: the five canonical CUDA steps, with the spectral model going to
// constant memory instead of a global buffer. We time ONLY the kernel (CUDA
// events), not the H2D/D2H copies (discussed separately in THEORY).
// ---------------------------------------------------------------------------
void decompose_gpu(const DectSinogram& sino, const SpectralModel& sm,
                   std::vector<double>& t1, std::vector<double>& t2,
                   std::vector<int>& iters, float* kernel_ms) {
    const int n = sino.n;
    t1.assign(static_cast<std::size_t>(n), 0.0);
    t2.assign(static_cast<std::size_t>(n), 0.0);
    iters.assign(static_cast<std::size_t>(n), 0);

    const std::size_t d_bytes = static_cast<std::size_t>(n) * sizeof(double);
    const std::size_t i_bytes = static_cast<std::size_t>(n) * sizeof(int);

    // (a) Upload the (small, read-only) spectral model to the __constant__ symbol.
    CUDA_CHECK(cudaMemcpyToSymbol(c_model, &sm, sizeof(SpectralModel)));

    // (b) Allocate + upload the measurements, and allocate the outputs.
    double *d_m_lo = nullptr, *d_m_hi = nullptr, *d_t1 = nullptr, *d_t2 = nullptr;
    int    *d_iters = nullptr;
    CUDA_CHECK(cudaMalloc(&d_m_lo, d_bytes));
    CUDA_CHECK(cudaMalloc(&d_m_hi, d_bytes));
    CUDA_CHECK(cudaMalloc(&d_t1,   d_bytes));
    CUDA_CHECK(cudaMalloc(&d_t2,   d_bytes));
    CUDA_CHECK(cudaMalloc(&d_iters, i_bytes));
    CUDA_CHECK(cudaMemcpy(d_m_lo, sino.m_lo.data(), d_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_m_hi, sino.m_hi.data(), d_bytes, cudaMemcpyHostToDevice));

    // (c) Launch. Enough blocks to cover n one-thread-per-bin, capped so the grid
    //     stays modest; the grid-stride loop handles any larger n.
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks < 1)    blocks = 1;
    if (blocks > 1024) blocks = 1024;
    GpuTimer timer;
    timer.start();
    decompose_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_m_lo, d_m_hi, n, d_t1, d_t2, d_iters);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("decompose_kernel");

    // (d) Copy results back, then (e) free device memory.
    CUDA_CHECK(cudaMemcpy(t1.data(),    d_t1,    d_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(t2.data(),    d_t2,    d_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(iters.data(), d_iters, i_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_m_lo));
    CUDA_CHECK(cudaFree(d_m_hi));
    CUDA_CHECK(cudaFree(d_t1));
    CUDA_CHECK(cudaFree(d_t2));
    CUDA_CHECK(cudaFree(d_iters));
}
