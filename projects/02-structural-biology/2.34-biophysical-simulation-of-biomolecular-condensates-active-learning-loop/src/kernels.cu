// ===========================================================================
// src/kernels.cu  --  Ensemble CG-MD kernel (one thread per trajectory)
// ---------------------------------------------------------------------------
// Project 2.34 : Biophysical Simulation of Biomolecular Condensates
//                (Active Learning Loop)  --  reduced-scope teaching version
//
// WHAT THIS FILE DOES
//   Implements the device kernel (ensemble_kernel) and the host-side glue
//   (integrate_gpu) that allocates GPU memory, launches the kernel, times it,
//   and brings the per-replica results back. Each thread runs the FULL Brownian-
//   dynamics trajectory for one candidate sequence by calling the shared
//   integrate_replica() in condensate.h -- the exact same code the CPU reference
//   runs -- so main.cu can compare them and trust the GPU when they agree.
//
//   There is deliberately NO device RNG state and NO atomics: the thermal noise
//   is a counter-based hash of (replica, step, bead, axis) (condensate.h), so
//   the result is bit-reproducible AND identical to the CPU's per-member draw.
//
// READ THIS AFTER: kernels.cuh (the thread-mapping idea), condensate.h (physics).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a solid default here: each thread does a LOT of
// sequential work (a whole trajectory) and uses a handful of fixed-size local
// arrays (3 * CND_MAX_BEADS doubles), so we keep the block modest to leave
// registers/local memory headroom and still give the scheduler several warps
// per block to hide latency. (Tune per GPU; see THEORY "GPU mapping".)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// ensemble_kernel: thread idx owns ensemble member idx.
//   Launch config (set in integrate_gpu):
//     grid  = ceil(n_members / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: idx = blockIdx.x * blockDim.x + threadIdx.x  -> member idx.
//   Memory: the thread reads its (lambda, seed) from the by-value config, runs
//   the trajectory entirely in registers/local memory, and writes exactly one
//   ReplicaResult to global memory. No shared memory, no atomics, no cross-thread
//   communication -- pure embarrassing parallelism over the ensemble.
//   Divergence is mild: all members run the same step count; only data-dependent
//   branches (the eq_steps latch, production measurement) differ trivially.
// ---------------------------------------------------------------------------
__global__ void ensemble_kernel(EnsembleConfig c, ReplicaResult* __restrict__ out) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's member
    // GUARD THE RAGGED LAST BLOCK: n_members is rarely a multiple of the block
    // size, so the final block has threads with idx >= n_members; they must do
    // nothing or they would write out of bounds (an illegal-address crash).
    if (idx >= ensemble_size(c)) return;

    // Select this candidate's stickiness and integrate its whole trajectory.
    // member_lambda() and integrate_replica() are the SAME functions the CPU
    // reference calls (condensate.h / reference_cpu.h) -> matching numbers.
    const double lam = member_lambda(c, idx);
    out[idx] = integrate_replica(c.model, idx, lam, c.k_cohese);
}

// ---------------------------------------------------------------------------
// integrate_gpu: host wrapper. The canonical CUDA steps for an ensemble:
//   (1) allocate the device result buffer
//   (2) launch one thread per member (no inputs to copy -- every thread derives
//       its own parameters from the small by-value config)
//   (3) copy the results device->host
//   (4) free device memory
// We time ONLY the kernel (step 2) with CUDA events so the reported figure is
// the compute cost, not the tiny D2H copy (discussed in THEORY).
// ---------------------------------------------------------------------------
void integrate_gpu(const EnsembleConfig& c, std::vector<ReplicaResult>& results,
                   float* kernel_ms) {
    const int M = ensemble_size(c);
    results.assign(static_cast<std::size_t>(M), ReplicaResult{});
    const std::size_t bytes = static_cast<std::size_t>(M) * sizeof(ReplicaResult);

    // (1) One ReplicaResult slot per member. d_ marks a DEVICE pointer (CLAUDE
    //     §12): dereferencing it on the host would crash, so the name matters.
    ReplicaResult* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, bytes));   // can fail: out of device memory

    // (2) Launch. Blocks must cover all M members, hence the ceiling division
    //     (M + B - 1) / B -- integer "round up". The whole config travels by
    //     value as a kernel argument (it is small and read-only).
    const int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    ensemble_kernel<<<blocks, THREADS_PER_BLOCK>>>(c, d_out);
    *kernel_ms = timer.stop_ms();            // GPU-measured kernel time
    CUDA_CHECK_LAST("ensemble_kernel");      // catch launch + execution errors

    // (3) Bring the per-replica results back to the host vector.
    CUDA_CHECK(cudaMemcpy(results.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // (4) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_out));
}
