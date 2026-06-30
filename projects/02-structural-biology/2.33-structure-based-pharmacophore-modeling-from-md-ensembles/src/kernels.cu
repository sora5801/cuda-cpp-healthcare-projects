// ===========================================================================
// src/kernels.cu  --  Pharmacophore-screen kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 2.33 : Structure-Based Pharmacophore Modeling from MD Ensembles
//
// WHAT THIS FILE DOES
//   GPU twin of screen_cpu(): one thread per library molecule, the query
//   pharmacophore in CONSTANT memory, scoring with the SAME score_molecule()
//   the CPU reference uses (pharmacophore.h). main.cu runs both and compares the
//   per-molecule scores. See ../THEORY.md for the science -> math -> GPU mapping.
//
// READ THIS AFTER: kernels.cuh (the idea), pharmacophore.h (the formula).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// The query pharmacophore in CONSTANT memory.
//   Every thread reads the same small set of query features and NEVER writes
//   them during the launch, which is the textbook case for __constant__ memory:
//   its dedicated cache broadcasts one value to a whole warp in a single cycle,
//   so there is no per-thread global-load traffic for the query. Sized at the
//   compile-time MAX_QUERY_FEATS (64 features * 24 B = 1.5 KB, well inside the
//   64 KB constant bank). We also stash the query length so the kernel knows how
//   many of the 64 slots are live.
// ---------------------------------------------------------------------------
__constant__ Feature c_query[MAX_QUERY_FEATS];
__constant__ int     c_n_query;

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, enough warps to hide memory latency, and many blocks resident
// for occupancy. The per-thread work here is a short double loop of exp() calls,
// so the kernel is compute-light and launch/occupancy dominated on tiny inputs
// (PATTERNS.md §7: timing is a teaching artifact, not a benchmark).
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// screen_kernel: one thread scores one library molecule.
//   Thread-to-data map: k = blockIdx.x * blockDim.x + threadIdx.x  (molecule id)
//   Memory: reads the query from __constant__ c_query; reads this molecule's
//   features from global `lib_feats` via the CSR `offset` array; writes one
//   float to global `scores`. No shared memory or atomics -- the molecules are
//   completely independent, the purest form of the independent-jobs pattern.
// ---------------------------------------------------------------------------
__global__ void screen_kernel(const Feature* __restrict__ lib_feats,
                              const int* __restrict__ offset,
                              int N, int n_query, double self_qq,
                              float* __restrict__ scores) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's molecule
    if (k >= N) return;                                    // guard the ragged last block

    // Where molecule k's features live in the flat CSR buffer.
    const int beg = offset[k];
    const int n_k = offset[k + 1] - beg;          // feature count of molecule k
    const Feature* lib_k = lib_feats + beg;

    // The ONE TRUE formula, identical to the CPU path (pharmacophore.h). The
    // query comes from constant memory; self_qq was precomputed once on the host.
    scores[k] = score_molecule(c_query, n_query, self_qq, lib_k, n_k);
}

// ---------------------------------------------------------------------------
// screen_gpu: host wrapper. The canonical CUDA steps:
//   (1) upload the query to CONSTANT memory (cudaMemcpyToSymbol)
//   (2) allocate + copy the flat library + offsets to the device
//   (3) launch screen_kernel (timed with CUDA events)
//   (4) copy the per-molecule scores back
//   (5) free device memory
// We time ONLY step (3) so the figure reflects kernel cost, not PCIe transfers.
// ---------------------------------------------------------------------------
void screen_gpu(const ScreenData& s, double self_qq,
                std::vector<float>& scores, float* kernel_ms) {
    const int N = s.N;
    const int n_query = static_cast<int>(s.query.size());
    scores.assign(static_cast<std::size_t>(N), 0.0f);

    // Guard the constant-memory capacity (a real screen would tile larger queries
    // or store them in global memory; for teaching we cap and fail loudly).
    if (n_query > MAX_QUERY_FEATS) {
        std::fprintf(stderr, "[screen_gpu] n_query=%d exceeds MAX_QUERY_FEATS=%d\n",
                     n_query, MAX_QUERY_FEATS);
        std::exit(EXIT_FAILURE);
    }

    // (1) Upload the query (and its length) to constant memory.
    CUDA_CHECK(cudaMemcpyToSymbol(c_query, s.query.data(), n_query * sizeof(Feature)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_n_query, &n_query, sizeof(int)));

    // (2) Device buffers for the flat library + CSR offsets.
    Feature* d_lib = nullptr;
    int*     d_offset = nullptr;
    float*   d_scores = nullptr;
    const std::size_t lib_bytes    = s.lib_feats.size() * sizeof(Feature);
    const std::size_t offset_bytes = s.offset.size() * sizeof(int);
    const std::size_t score_bytes  = static_cast<std::size_t>(N) * sizeof(float);

    CUDA_CHECK(cudaMalloc(&d_lib, lib_bytes));
    CUDA_CHECK(cudaMalloc(&d_offset, offset_bytes));
    CUDA_CHECK(cudaMalloc(&d_scores, score_bytes));
    CUDA_CHECK(cudaMemcpy(d_lib, s.lib_feats.data(), lib_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset, s.offset.data(), offset_bytes, cudaMemcpyHostToDevice));

    // (3) Launch. Cover all N molecules: ceil(N / B) blocks.
    const int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    screen_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_lib, d_offset, N, n_query, self_qq, d_scores);
    *kernel_ms = timer.stop_ms();        // GPU-measured kernel time
    CUDA_CHECK_LAST("screen_kernel");    // catch launch + execution errors

    // (4) Bring the scores back.
    CUDA_CHECK(cudaMemcpy(scores.data(), d_scores, score_bytes, cudaMemcpyDeviceToHost));

    // (5) Free device memory (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_lib));
    CUDA_CHECK(cudaFree(d_offset));
    CUDA_CHECK(cudaFree(d_scores));
}
