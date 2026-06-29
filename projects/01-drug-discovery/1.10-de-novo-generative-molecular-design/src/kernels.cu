// ===========================================================================
// src/kernels.cu  --  GPU generation kernel + host wrapper (constant-mem model)
// ---------------------------------------------------------------------------
// Project 1.10 : De Novo Generative Molecular Design (reduced-scope teaching).
//
// GPU twin of generate_and_score_cpu(): each thread generates ONE molecule from
// its own RNG stream (rng_seed(seed, i)) using the SHARED generator.h functions,
// scores it, and writes (score, length). Because the device kernel and the CPU
// reference call the identical inline functions with the identical per-index
// seed, molecule i is bit-identical on both -- so main.cu can verify with an
// EXACT integer comparison. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "generator.h"           // MarkovModel, generate_molecule, score_molecule
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

// A good occupancy default on sm_75..sm_89; the work per thread is a short
// integer loop, so 256 threads/block keeps plenty of warps resident to hide the
// latency of the constant-memory reads.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// THE MODEL IN CONSTANT MEMORY
//   Every thread reads the SAME transition table and never writes it, so it is a
//   textbook fit for __constant__ memory: a small (NSYM*NSYM*4 bytes ~ 784 B)
//   read-only region whose dedicated cache BROADCASTS one fetched value to every
//   thread in a warp. Passing the model by value as a kernel argument would copy
//   it into every thread's stack frame instead -- wasteful. (Same idea as the
//   constant-memory query in flagship 1.12.)
//
//   c_model is a device-side symbol; the host fills it with cudaMemcpyToSymbol.
// ---------------------------------------------------------------------------
__constant__ MarkovModel c_model;

// ---------------------------------------------------------------------------
// generate_kernel: one thread generates and scores one molecule.
//   Thread-to-data mapping: i = blockIdx.x * blockDim.x + threadIdx.x owns
//   molecule i. The ragged last block is guarded by `if (i >= n_gen) return;`.
//
//   Each thread:
//     1. seeds its private RNG from (seed, i)  -> independent, reproducible;
//     2. generates a string into a PER-THREAD local buffer (lives in registers/
//        local memory, never shared -- molecules do not interact);
//     3. scores it and writes the two small outputs.
//   No atomics, no shared memory, no __syncthreads -- embarrassingly parallel.
//   The classic caveat is DIVERGENCE: molecules have different lengths, so
//   threads in a warp run their generation loops for different counts of steps
//   and the warp waits for its slowest thread (discussed in THEORY §numerics).
// ---------------------------------------------------------------------------
__global__ void generate_kernel(int n_gen, unsigned long long seed,
                                int* __restrict__ scores,
                                int* __restrict__ lengths) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's molecule
    if (i >= n_gen) return;                                // guard ragged last block

    char buf[MAX_LEN + 1];                                 // per-thread scratch string
    Rng rng = rng_seed(seed, static_cast<unsigned long long>(i));  // private stream
    int len = generate_molecule(c_model, rng, buf);        // sample (shared loop)
    int sc  = score_molecule(buf, len);                    // reward (shared scorer)

    scores[i]  = sc;     // coalesced write: adjacent threads -> adjacent slots
    lengths[i] = len;
}

// ---------------------------------------------------------------------------
// generate_and_score_gpu: host wrapper that hides all CUDA bookkeeping.
//   Steps: upload model -> constant memory; allocate the two device outputs;
//   launch the kernel with ceil(n_gen / B) blocks; time just the kernel with
//   CUDA events; copy results D2H; free. main.cu calls exactly this.
// ---------------------------------------------------------------------------
void generate_and_score_gpu(const MarkovModel& model, int n_gen, uint64_t seed,
                            std::vector<int>& scores, std::vector<int>& lengths,
                            float* kernel_ms) {
    scores.assign(static_cast<size_t>(n_gen), 0);
    lengths.assign(static_cast<size_t>(n_gen), 0);

    // Upload the read-only transition model to constant memory (one small copy).
    CUDA_CHECK(cudaMemcpyToSymbol(c_model, &model, sizeof(MarkovModel)));

    // Device output arrays: one int score + one int length per molecule.
    int* d_scores  = nullptr;
    int* d_lengths = nullptr;
    const size_t bytes = static_cast<size_t>(n_gen) * sizeof(int);
    CUDA_CHECK(cudaMalloc(&d_scores,  bytes));
    CUDA_CHECK(cudaMalloc(&d_lengths, bytes));

    // 1-D grid: exactly enough blocks of THREADS_PER_BLOCK to cover n_gen
    // molecules (the ceil-division idiom; the last block may be partly idle and
    // is handled by the in-kernel guard).
    const int blocks = (n_gen + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    GpuTimer timer;
    timer.start();
    generate_kernel<<<blocks, THREADS_PER_BLOCK>>>(
        n_gen, static_cast<unsigned long long>(seed), d_scores, d_lengths);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("generate_kernel");   // catch launch + execution errors

    // Copy results back to the host vectors.
    CUDA_CHECK(cudaMemcpy(scores.data(),  d_scores,  bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(lengths.data(), d_lengths, bytes, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_scores));
    CUDA_CHECK(cudaFree(d_lengths));
}
