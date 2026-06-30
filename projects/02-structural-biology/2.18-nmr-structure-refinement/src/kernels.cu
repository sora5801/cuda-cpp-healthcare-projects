// ===========================================================================
// src/kernels.cu  --  Ensemble simulated-annealing kernel (one thread/replica)
// ---------------------------------------------------------------------------
// Project 2.18 : NMR Structure Refinement
//
// WHAT THIS FILE DOES
//   Implements the device kernel (anneal_kernel) and the host glue
//   (anneal_ensemble_gpu) that allocates the result buffer, launches the kernel,
//   times it, and brings the per-replica results back. The per-replica annealer
//   itself (RNG, energy, the Metropolis loop) is the SHARED anneal_one() from
//   nmr_refine.h -- the same code reference_cpu.cpp runs -- so the GPU and CPU
//   ensembles match to round-off. main.cu runs both and compares them.
//
// READ THIS AFTER: kernels.cuh (the thread-mapping idea), nmr_refine.h (anneal_one).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a solid default here: each thread does a LONG,
// register/local-memory-heavy annealing loop (not a memory-bound one-liner), so a
// smaller block keeps per-SM register/local pressure reasonable while still giving
// the scheduler several warps to hide latency. (Tune per GPU; THEORY.md section 3.)
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// anneal_kernel: thread r owns replica r.
//   Launch config (set in anneal_ensemble_gpu):
//     grid  = ceil(n_replicas / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: r = blockIdx.x * blockDim.x + threadIdx.x  ->  replica r.
//
//   Memory: each thread holds its own current/best coordinate scratch in
//   PER-THREAD LOCAL MEMORY (the two fixed-size arrays below). No shared memory
//   and NO atomics are needed -- replicas never touch each other's data, so there
//   is nothing to synchronise. That independence is exactly why the ensemble maps
//   so cleanly onto the GPU. (Divergence is mild: every replica runs the same
//   n_steps; only the accept/reject branch differs, which is inherent to MC.)
//
//   The scratch arrays are sized to the compile-time cap NMR_MAX_BEADS so the
//   kernel needs no dynamic allocation; a replica uses only the first n_beads.
// ---------------------------------------------------------------------------
__global__ void anneal_kernel(RefineConfig c, ReplicaResult* __restrict__ out) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's replica
    if (r >= c.n_replicas) return;                         // guard the ragged block

    // Per-thread working coordinates (x = current trial, xbest = best so far),
    // 3 doubles per bead. These live in local memory; the annealer reads/writes
    // them thousands of times, but they fit comfortably for chains <= NMR_MAX_BEADS.
    double x[3 * NMR_MAX_BEADS];
    double xbest[3 * NMR_MAX_BEADS];

    // Run the SHARED annealer -- byte-for-byte the same code path the CPU uses.
    out[r] = anneal_one(c, static_cast<uint64_t>(r), x, xbest);
}

// ---------------------------------------------------------------------------
// anneal_ensemble_gpu: host wrapper. Three of the canonical CUDA steps (there is
//   no input array to copy up -- the whole job rides in the by-value RefineConfig):
//     (1) allocate the device result buffer
//     (2) launch one thread per replica (timed with CUDA events)
//     (3) copy results device->host, then free
//   We time ONLY the kernel so the reported figure is compute, not the (tiny) D2H
//   copy. This is a teaching artifact, never a benchmark claim (CLAUDE.md s.12).
// ---------------------------------------------------------------------------
void anneal_ensemble_gpu(const RefineConfig& c,
                         std::vector<ReplicaResult>& results, float* kernel_ms) {
    const int M = c.n_replicas;
    results.assign(static_cast<std::size_t>(M), ReplicaResult{});

    // (1) Device buffer for the M per-replica results.
    ReplicaResult* d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_out, static_cast<std::size_t>(M) * sizeof(ReplicaResult)));

    // (2) Launch: enough blocks to cover all replicas (ceiling division).
    const int blocks = (M + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    anneal_kernel<<<blocks, THREADS_PER_BLOCK>>>(c, d_out);
    *kernel_ms = timer.stop_ms();            // GPU-measured kernel time
    CUDA_CHECK_LAST("anneal_kernel");        // catch launch + execution errors

    // (3) Bring results back, then release the device buffer (no GC on the GPU).
    CUDA_CHECK(cudaMemcpy(results.data(), d_out,
                          static_cast<std::size_t>(M) * sizeof(ReplicaResult),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_out));
}
