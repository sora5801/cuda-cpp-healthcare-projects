// ===========================================================================
// src/kernels.cu  --  GPU TPS kernel (per-thread shooters + integer atomics)
// ---------------------------------------------------------------------------
// Project 2.32 : Protein Folding Pathway Extraction (Transition Path Sampling)
//
// GPU twin of tps_cpu(): runs the IDENTICAL shooting moves (shared run_shot in
// tps_physics.h), but in parallel and scored with atomicAdd. main.cu runs both
// CPU and GPU and asserts the integer tallies are identical. See ../THEORY.md
// "GPU mapping" for the thread/block/grid reasoning.
// ===========================================================================
#include "kernels.cuh"
#include "tps_physics.h"          // SimParams, run_shot, ShotResult (host+device)
#include "util/cuda_check.cuh"    // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"         // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, eight warps to hide the long-latency math (sqrt/log/cos in the
// Gaussian RNG and the BD loop), and plenty of resident blocks for occupancy.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// tps_kernel: grid-stride over shooters. Each iteration runs ONE complete
// shooting move and atomically adds its integer results to the shared tallies.
//
//   Launch config (set in tps_gpu):
//     grid  = a fixed number of blocks (1024); the grid-stride loop covers any
//             n_shooters, so we never need to size the grid to the input.
//     block = THREADS_PER_BLOCK threads.
//   Thread-to-data map: thread (blockIdx.x, threadIdx.x) starts at shooter
//     `start = blockIdx.x*blockDim.x + threadIdx.x` and strides by `blockDim.x *
//     gridDim.x`, so shooter indices are partitioned across all threads exactly
//     once -- the same index set the CPU loops over, in a different order.
//
//   Memory: no shared memory. The four tally targets live in GLOBAL memory and
//   are updated with atomicAdd because MANY threads hit the SAME scalar counters
//   and the SAME histogram bins. Integer increments => the atomics commute =>
//   deterministic, CPU-matching result (PATTERNS.md §3). The "divergence" cost
//   of Monte Carlo shows up here: different shooters take different numbers of
//   BD steps before committing, so warp lanes finish at different times -- the
//   classic stochastic-simulation challenge (THEORY.md §numerics).
//
//   atomicAdd on unsigned long long is supported on all our target archs
//   (sm_75+). We cast each 0/1 indicator to unsigned long long before adding.
// ---------------------------------------------------------------------------
__global__ void tps_kernel(SimParams sp,
                           unsigned long long* __restrict__ d_n_transitions,
                           unsigned long long* __restrict__ d_n_fwd_to_B,
                           unsigned long long* __restrict__ d_shots_per_bin,
                           unsigned long long* __restrict__ d_committed_per_bin) {
    const int stride = blockDim.x * gridDim.x;                  // threads in grid
    const int start  = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's first shooter

    for (int i = start; i < sp.n_shooters; i += stride) {
        // Run the ONE TRUE shooting move for shooter i (shared with the CPU).
        // Identical RNG stream + identical dynamics => identical ShotResult.
        ShotResult r = run_shot(sp, i);

        // Scalar tallies. Each is a 0/1 indicator; atomicAdd serialises the
        // increments but, being integer, the final sum is order-independent.
        if (r.is_transition)
            atomicAdd(d_n_transitions, 1ULL);
        if (r.committed_B)
            atomicAdd(d_n_fwd_to_B, 1ULL);

        // Per-bin committor histogram. sp_bin is already clamped in-range by
        // committor_bin(), so this global index is always valid.
        atomicAdd(&d_shots_per_bin[r.sp_bin], 1ULL);
        if (r.committed_B)
            atomicAdd(&d_committed_per_bin[r.sp_bin], 1ULL);
    }
}

// ---------------------------------------------------------------------------
// tps_gpu: host wrapper. The canonical CUDA steps, specialised for a tally:
//   (1) allocate device counters (scalars + two [n_bins] histograms),
//   (2) zero them (the kernel only ADDS),
//   (3) launch the kernel over all shooters (timed with CUDA events),
//   (4) copy the integer tallies back and convert to the host's signed type,
//   (5) free device memory.
// We time ONLY the kernel (step 3); the tiny copies are negligible and would
// only muddy the teaching figure (THEORY.md §verification).
// ---------------------------------------------------------------------------
void tps_gpu(const TpsProblem& prob, TpsTally& tally, float* kernel_ms) {
    const SimParams& P = prob.sp;
    const int nb = P.n_bins;
    tally.n_transitions = 0;
    tally.n_fwd_to_B    = 0;
    tally.resize(nb);

    // (1) Device counters. Two scalars and two per-bin histograms, all u64.
    unsigned long long *d_ntr = nullptr, *d_nB = nullptr;
    unsigned long long *d_shots = nullptr, *d_comm = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ntr, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_nB,  sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_shots, static_cast<std::size_t>(nb) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMalloc(&d_comm,  static_cast<std::size_t>(nb) * sizeof(unsigned long long)));

    // (2) Zero everything: the kernel only does atomicAdd, so it relies on a
    //     clean slate. cudaMemset writes BYTES; 0 bytes => 0 in every u64.
    CUDA_CHECK(cudaMemset(d_ntr, 0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_nB,  0, sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_shots, 0, static_cast<std::size_t>(nb) * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemset(d_comm,  0, static_cast<std::size_t>(nb) * sizeof(unsigned long long)));

    // (3) Launch. A fixed grid of 1024 blocks gives the GPU plenty of resident
    //     warps to hide the BD loop's latency; the grid-stride loop covers any
    //     number of shooters with this fixed grid.
    const int blocks = 1024;
    GpuTimer timer;
    timer.start();
    tps_kernel<<<blocks, THREADS_PER_BLOCK>>>(P, d_ntr, d_nB, d_shots, d_comm);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("tps_kernel");   // catches launch + execution errors

    // (4) Copy the tallies back. Stage into u64 host buffers, then widen into
    //     the signed long long fields the CPU uses (values are tiny vs. 2^63).
    unsigned long long h_ntr = 0, h_nB = 0;
    std::vector<unsigned long long> h_shots(nb, 0ULL), h_comm(nb, 0ULL);
    CUDA_CHECK(cudaMemcpy(&h_ntr, d_ntr, sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(&h_nB,  d_nB,  sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_shots.data(), d_shots,
                          static_cast<std::size_t>(nb) * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_comm.data(), d_comm,
                          static_cast<std::size_t>(nb) * sizeof(unsigned long long),
                          cudaMemcpyDeviceToHost));

    tally.n_transitions = static_cast<long long>(h_ntr);
    tally.n_fwd_to_B    = static_cast<long long>(h_nB);
    for (int b = 0; b < nb; ++b) {
        tally.shots_per_bin[static_cast<std::size_t>(b)]     = static_cast<long long>(h_shots[b]);
        tally.committed_per_bin[static_cast<std::size_t>(b)] = static_cast<long long>(h_comm[b]);
    }

    // (5) Free device memory (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_ntr));
    CUDA_CHECK(cudaFree(d_nB));
    CUDA_CHECK(cudaFree(d_shots));
    CUDA_CHECK(cudaFree(d_comm));
}
