// ===========================================================================
// src/kernels.cu  --  Ensemble cardiac-electromechanics kernel (thread per heart)
// ---------------------------------------------------------------------------
// Project 6.16 : Cardiac Mechanics & Electromechanical Coupling
//
// GPU twin of integrate_cpu(): each thread runs the same multi-beat RK4 loop
// (cardiac.h) for one virtual heart (one point on the contractility x afterload
// sweep) and writes its PV-loop summary. main.cu compares the per-heart results
// against the CPU reference. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

// Block size. 128 threads/block is a solid occupancy default on sm_75..sm_89.
// Each thread is register/local-memory heavy (it holds the whole ODE state and
// runs a long time loop), so a modest block keeps register pressure reasonable.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// ensemble_kernel
//   thread idx  ->  virtual-heart idx  (0 .. nT*nR-1)
//   Each thread:
//     1. guards the ragged last block,
//     2. runs integrate_member() -- the FULL n_beats * steps_per_beat RK4 loop
//        in registers/local memory (no shared memory, no atomics: every heart
//        is independent), and
//     3. writes exactly one CycleResult.
//   Divergence is mild: all hearts run the same number of steps; only the
//   ejection/filling branches inside deriv() differ, and those reconverge each
//   step. This is the "batch ODE, one integration point per thread" pattern.
// ---------------------------------------------------------------------------
__global__ void ensemble_kernel(EnsembleConfig c, CycleResult* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's heart
    if (idx >= ensemble_size(c)) return;                     // guard last block
    out[idx] = integrate_member(c, idx);                     // shared cardiac.h math
}

// ---------------------------------------------------------------------------
// integrate_gpu -- host wrapper: launch, copy back, time the kernel.
//   All CUDA bookkeeping (allocate, launch, copy, free) is hidden here so
//   main.cu reads cleanly. The kernel time is measured with CUDA events
//   (util/timer.cuh) -- the only fair way to time an asynchronous launch.
// ---------------------------------------------------------------------------
void integrate_gpu(const EnsembleConfig& c, std::vector<CycleResult>& results,
                   float* kernel_ms) {
    const int M = ensemble_size(c);
    results.assign(static_cast<std::size_t>(M), CycleResult{});

    // Device output buffer: one CycleResult per heart. There are NO device
    // inputs to copy -- the whole EnsembleConfig is small and passed BY VALUE as
    // a kernel argument (it lives in constant/param memory), and every heart's
    // parameters are derived on the fly by member_params().
    CycleResult* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(M) * sizeof(CycleResult)));

    const int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    GpuTimer timer;
    timer.start();
    ensemble_kernel<<<blocks, THREADS_PER_BLOCK>>>(c, d_out);
    *kernel_ms = timer.stop_ms();          // blocks until the kernel finishes
    CUDA_CHECK_LAST("ensemble_kernel");    // catch launch + execution errors

    // Copy the M summaries back to the host for verification + reporting.
    CUDA_CHECK(cudaMemcpy(results.data(), d_out,
                          static_cast<std::size_t>(M) * sizeof(CycleResult),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
}
