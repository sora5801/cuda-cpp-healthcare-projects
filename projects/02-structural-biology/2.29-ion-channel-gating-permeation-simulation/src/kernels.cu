// ===========================================================================
// src/kernels.cu  --  GPU Brownian-dynamics permeation kernel (RNG + atomics)
// ---------------------------------------------------------------------------
// Project 2.29 : Ion Channel Gating & Permeation Simulation
//
// WHAT THIS FILE DOES
//   Implements the device kernel (permeation_kernel) and the host-side glue
//   (permeation_gpu) that allocates GPU memory, launches the kernel, times it,
//   and brings the integer tallies back. This is the GPU twin of the serial CPU
//   reference in reference_cpu.cpp: it runs the IDENTICAL ion trajectories
//   (shared RNG + bd_step from channel_physics.h) in parallel, scored with
//   atomicAdd. main.cu runs both and asserts the tallies are bit-identical.
//
//   Comment density here targets >= 1:1 (CLAUDE.md §6.2).
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea),
//                  channel_physics.h (the shared per-step physics).
// ===========================================================================
#include "kernels.cuh"
#include "channel_physics.h"     // ChannelParams, Rng, bd_step, bin_of (host+device)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, enough warps (8) for the scheduler to hide the RNG/transcendental
// latency, and leaves many blocks resident for occupancy.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// permeation_kernel: one thread integrates one or more independent ions.
//   Launch config (set in permeation_gpu):
//     grid  = a fixed number of blocks (1024); a GRID-STRIDE loop lets this cover
//             any n_ions, so we never need to resize the grid for big inputs.
//     block = THREADS_PER_BLOCK threads.
//   Thread-to-data map: thread t = blockIdx.x*blockDim.x + threadIdx.x owns ions
//   t, t+stride, t+2*stride, ... where stride = blockDim.x * gridDim.x.
//
//   Memory spaces touched:
//     * registers : z, the RNG state, this ion's crossing counters (hot, private)
//     * global    : occupancy[n_bins] and crossings[2], updated with atomicAdd
//       because MANY threads land in the SAME bins/counters. Integer atomics
//       commute, so the result is deterministic and equals the CPU's.
//   No shared memory: the histogram can be larger than a block would want to
//   stage, and global atomics on integers are already exact; THEORY.md discusses
//   the shared-memory privatization optimization as an exercise.
// ---------------------------------------------------------------------------
__global__ void permeation_kernel(ChannelParams cp,
                                  unsigned long long n_ions,
                                  unsigned long long seed,
                                  unsigned long long* __restrict__ occupancy,
                                  unsigned long long* __restrict__ crossings) {
    // Grid-stride bounds. `start` is this thread's first ion; `stride` is how far
    // to jump to the next ion this thread owns.
    const unsigned long long stride =
        static_cast<unsigned long long>(blockDim.x) * gridDim.x;
    const unsigned long long start =
        static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;

    for (unsigned long long i = start; i < n_ions; i += stride) {
        // Per-ion private state -- lives in registers, no contention.
        Rng rng = rng_seed(seed, i);   // SAME seeding as the CPU -> same stream
        double z = 0.0;                // start at the intracellular mouth (z=0)
        unsigned long long fwd = 0, rev = 0;   // this ion's crossing counters

        for (int s = 0; s < cp.n_steps; ++s) {
            // One shared Brownian step; updates fwd/rev on permeation + re-injects.
            z = bd_step(cp, rng, z, &fwd, &rev);
            // Occupancy histogram: integer atomicAdd into the bin holding z now.
            // Many threads (different ions) hit the same bin at the same step, so
            // the add MUST be atomic; being an integer, it stays deterministic.
            atomicAdd(&occupancy[bin_of(cp, z)], 1ULL);
        }

        // Fold this ion's crossing counts into the two global counters. We do
        // this ONCE per ion (not per step) to minimize atomic traffic on the two
        // hottest addresses -- a tiny, exact optimization (still integer adds).
        if (fwd) atomicAdd(&crossings[0], fwd);
        if (rev) atomicAdd(&crossings[1], rev);
    }
}

// ---------------------------------------------------------------------------
// permeation_gpu: host wrapper. The canonical CUDA dance, but the only thing we
// copy IN is the small ChannelParams (passed by value as a kernel argument); the
// only things we copy OUT are the integer tallies. There are no big input arrays
// because the ions are GENERATED on the device from their indices.
//   (1) allocate + zero device tallies
//   (2) launch the kernel (timed with CUDA events)
//   (3) copy the integer results device->host
//   (4) free device memory
// We time ONLY the kernel so the figure is the compute cost, not allocation.
// ---------------------------------------------------------------------------
void permeation_gpu(const PermeationProblem& prob, PermeationResult& out,
                    float* kernel_ms) {
    const int n_bins = prob.cp.n_bins;
    out.occupancy.assign(static_cast<std::size_t>(n_bins), 0ULL);
    out.fwd = 0;
    out.rev = 0;

    // (1) Device tallies. d_ prefix marks DEVICE pointers (CLAUDE.md §12).
    unsigned long long* d_occ = nullptr;   // [n_bins] occupancy histogram
    unsigned long long* d_cross = nullptr; // [2] = {forward, reverse} counts
    const std::size_t occ_bytes =
        static_cast<std::size_t>(n_bins) * sizeof(unsigned long long);
    CUDA_CHECK(cudaMalloc(&d_occ, occ_bytes));       // can fail: out of memory
    CUDA_CHECK(cudaMalloc(&d_cross, 2 * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_occ, 0, occ_bytes));     // zero before accumulating
    CUDA_CHECK(cudaMemset(d_cross, 0, 2 * sizeof(unsigned long long)));

    // (2) Launch. A fixed grid + grid-stride loop covers any number of ions and
    //     keeps the GPU saturated with resident warps.
    const int blocks = 1024;
    GpuTimer timer;
    timer.start();
    permeation_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        prob.cp, prob.n_ions, prob.seed, d_occ, d_cross);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time
    CUDA_CHECK_LAST("permeation_kernel");  // catch launch + execution errors

    // (3) Copy the integer results back to the host.
    CUDA_CHECK(cudaMemcpy(out.occupancy.data(), d_occ, occ_bytes,
                          cudaMemcpyDeviceToHost));
    unsigned long long h_cross[2] = {0ULL, 0ULL};
    CUDA_CHECK(cudaMemcpy(h_cross, d_cross, 2 * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    out.fwd = h_cross[0];
    out.rev = h_cross[1];

    // (4) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_occ));
    CUDA_CHECK(cudaFree(d_cross));
}
