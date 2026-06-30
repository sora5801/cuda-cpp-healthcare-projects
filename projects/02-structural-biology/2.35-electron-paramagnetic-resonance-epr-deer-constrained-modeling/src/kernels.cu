// ===========================================================================
// src/kernels.cu  --  GPU DEER back-calculation kernel + its host wrapper.
// ---------------------------------------------------------------------------
// Project 2.35 : Electron Paramagnetic Resonance (EPR/DEER) Constrained Modeling
//
// GPU twin of deer_backcalc_cpu(): one thread per MD frame runs the SHARED
// deer_member_histogram() (deer.h), so the device histograms are bit-for-bit
// identical to the CPU ones. The downstream reweighting is shared host code
// (reference_cpu.cpp), so both pipelines converge to the same weights. main.cu
// compares the histograms and the recovered weights.  See ../THEORY.md.
//
// READ THIS AFTER: kernels.cuh, deer.h.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

#include <cstddef>               // std::size_t

// 256 threads/block is a solid occupancy default on sm_75..sm_89. Each thread
// does a ROTAMERS^2 convolution, so the block is compute-bound, not memory-bound;
// the exact block size barely matters here, and 256 keeps register pressure low.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// deer_backcalc_kernel  --  one thread, one frame's P_m(r).
//   Thread m = blockIdx.x*blockDim.x + threadIdx.x owns frame m. It points at
//   the m-th slice of the rotamer arrays and the m-th output row, then calls the
//   SHARED histogram routine. Because each thread writes a disjoint output row,
//   there are no atomics and no inter-thread races -- pure data parallelism.
//   The ragged-last-block guard (m >= M -> return) drops the surplus threads.
// ---------------------------------------------------------------------------
__global__ void deer_backcalc_kernel(int M,
                                     const Spin3* __restrict__ siteA,
                                     const Spin3* __restrict__ siteB,
                                     double* __restrict__ hist) {
    const int m = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's frame
    if (m >= M) return;                                    // guard the ragged tail

    // Pointers into this frame's own data (no overlap with other threads).
    const Spin3* A = siteA + static_cast<std::size_t>(m) * ROTAMERS_PER_SITE;
    const Spin3* B = siteB + static_cast<std::size_t>(m) * ROTAMERS_PER_SITE;
    double* h      = hist  + static_cast<std::size_t>(m) * NBINS;

    // Exactly the same call the CPU reference makes -> identical numbers.
    deer_member_histogram(A, B, h);
}

// ---------------------------------------------------------------------------
// deer_backcalc_gpu  --  host wrapper: H2D copy, launch, D2H copy, timing.
// ---------------------------------------------------------------------------
void deer_backcalc_gpu(int M,
                       const std::vector<Spin3>& siteA,
                       const std::vector<Spin3>& siteB,
                       std::vector<double>& hist,
                       float* kernel_ms) {
    const std::size_t n_rot  = static_cast<std::size_t>(M) * ROTAMERS_PER_SITE;
    const std::size_t n_hist = static_cast<std::size_t>(M) * NBINS;
    hist.assign(n_hist, 0.0);

    // --- Device buffers ----------------------------------------------------
    // Two input rotamer clouds + one output histogram matrix. Sizes are exact;
    // CUDA_CHECK aborts with a clear message on an out-of-memory failure.
    Spin3*  d_A = nullptr;
    Spin3*  d_B = nullptr;
    double* d_hist = nullptr;
    CUDA_CHECK(cudaMalloc(&d_A,    n_rot  * sizeof(Spin3)));
    CUDA_CHECK(cudaMalloc(&d_B,    n_rot  * sizeof(Spin3)));
    CUDA_CHECK(cudaMalloc(&d_hist, n_hist * sizeof(double)));

    // --- H2D: upload the rotamer clouds ------------------------------------
    CUDA_CHECK(cudaMemcpy(d_A, siteA.data(), n_rot * sizeof(Spin3), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, siteB.data(), n_rot * sizeof(Spin3), cudaMemcpyHostToDevice));

    // --- Launch: one thread per frame --------------------------------------
    const int grid = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;  // ceil(M / B)
    GpuTimer timer;
    timer.start();
    deer_backcalc_kernel<<<grid, THREADS_PER_BLOCK>>>(M, d_A, d_B, d_hist);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time (events)
    CUDA_CHECK_LAST("deer_backcalc_kernel");

    // --- D2H: bring the histograms back ------------------------------------
    CUDA_CHECK(cudaMemcpy(hist.data(), d_hist, n_hist * sizeof(double), cudaMemcpyDeviceToHost));

    // --- Free --------------------------------------------------------------
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_hist));
}
