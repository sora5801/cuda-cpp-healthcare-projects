// ===========================================================================
// src/kernels.cu  --  GPU Monte Carlo stray-dose kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 5.10 : Secondary Cancer Risk & Stray-Dose Monte Carlo
//
// WHAT THIS FILE DOES
//   GPU twin of stray_cpu(): it runs the IDENTICAL histories (shared physics in
//   stray_physics.h), just in parallel and scored with atomicAdd instead of '+='.
//   main.cu runs both CPU and GPU and asserts the fixed-point dose tallies are
//   bit-identical. See ../THEORY.md "GPU mapping" and "Numerical considerations".
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea),
//   stray_physics.h (the shared transport this kernel calls).
// ===========================================================================
#include "kernels.cuh"
#include "stray_physics.h"       // simulate_history, Rng, DepositList (HD-shared)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, enough warps to hide latency, and it leaves many blocks resident.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// stray_kernel: grid-stride over primary histories. Each iteration simulates one
// photon with its own reproducible RNG stream and atomically adds its deposits.
//
//   Launch config (set in dose_gpu):
//     grid  = 1024 blocks (a fixed grid; the grid-stride loop covers any history
//             count with these threads, so occupancy stays high regardless of N)
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: thread (blockIdx.x, threadIdx.x) starts at history
//     `start = blockIdx.x*blockDim.x + threadIdx.x`, then strides by the total
//     thread count until all n_histories are consumed.
//
//   Memory: no shared memory. The organ tally is tiny but written by many threads,
//   so it lives in global memory and is updated with atomicAdd on 64-bit integers.
//   Because deposits are FIXED-POINT INTEGERS, the atomics COMMUTE -> the result is
//   deterministic and matches the CPU tally exactly (a float atomic would not).
//
//   Divergence: different photons interact/roulette differently, so lanes in a
//   warp take different paths and finish at different times -- the classic MC
//   challenge. Production codes sort/regenerate particles by state to reduce it;
//   here we keep it simple and explain it (THEORY.md "GPU mapping").
// ---------------------------------------------------------------------------
__global__ void stray_kernel(SimParams sp,
                             unsigned long long* __restrict__ dose) {
    // Total number of threads in the grid = stride of the grid-stride loop.
    const unsigned long long stride =
        static_cast<unsigned long long>(blockDim.x) * gridDim.x;
    // This thread's first history index.
    const unsigned long long start =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    // Per-thread scratch deposit list (lives in registers/local memory). Reused
    // across the thread's histories so there is no per-history allocation.
    DepositList dl;

    for (unsigned long long i = start; i < sp.n_histories; i += stride) {
        // Seed THIS history's stream from (seed, i) -- identical to the CPU, so
        // history i produces the identical deposits on both sides.
        Rng rng = rng_seed(sp.seed, i);
        simulate_history(sp, rng, dl);            // shared transport (stray_physics.h)
        for (int d = 0; d < dl.count; ++d)
            atomicAdd(&dose[dl.organ[d]], dl.dose[d]);   // many threads -> same organs
    }
}

// ---------------------------------------------------------------------------
// dose_gpu: host wrapper. Allocate + zero the device tally, launch the histories,
// copy the per-organ dose back. We time ONLY the kernel with CUDA events, so the
// reported figure is compute cost, not the (tiny) copy cost.
// ---------------------------------------------------------------------------
void dose_gpu(const StrayProblem& prob, std::vector<unsigned long long>& dose,
              float* kernel_ms) {
    const int n_organs = prob.sp.n_organs;
    dose.assign(static_cast<std::size_t>(n_organs), 0ULL);
    const std::size_t bytes = static_cast<std::size_t>(n_organs) * sizeof(unsigned long long);

    // (1) Device tally. d_ marks a DEVICE pointer (CLAUDE.md section 12).
    unsigned long long* d_dose = nullptr;
    CUDA_CHECK(cudaMalloc(&d_dose, bytes));        // can fail: out of device memory
    CUDA_CHECK(cudaMemset(d_dose, 0, bytes));      // start every organ at zero dose

    // (2) Launch. A fixed 1024-block grid gives the GPU plenty of resident warps;
    //     the grid-stride loop inside the kernel covers any n_histories.
    const int blocks = 1024;
    GpuTimer timer;
    timer.start();
    stray_kernel<<<blocks, THREADS_PER_BLOCK>>>(prob.sp, d_dose);
    *kernel_ms = timer.stop_ms();                  // GPU-measured kernel time
    CUDA_CHECK_LAST("stray_kernel");               // catch launch + execution errors

    // (3) Copy the tally back, then free the device buffer (no GC on the GPU).
    CUDA_CHECK(cudaMemcpy(dose.data(), d_dose, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_dose));
}
