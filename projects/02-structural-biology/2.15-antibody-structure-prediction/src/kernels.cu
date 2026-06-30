// ===========================================================================
// src/kernels.cu  --  CDR-similarity screening kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 2.15 : Antibody Structure Prediction  (reduced-scope: CDR screening)
//
// This is the GPU twin of score_cpu() in reference_cpu.cpp. main.cu runs both
// and asserts they agree. The per-pair scoring math is the SHARED core
// ab_cdr_score() from antibody.h (compiled here by nvcc as __device__, and in
// reference_cpu.cpp by the host compiler as a plain function) -> the two paths
// are guaranteed bit-for-bit identical. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "antibody.h"            // ab_cdr_score, AB_RECORD_LEN (the shared core)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

// ---------------------------------------------------------------------------
// The QUERY antibody record in CONSTANT memory.
//   * All threads read the same AB_RECORD_LEN (=144) encoded residues but none
//     writes them, and they are identical for the whole launch -> constant
//     memory is ideal: its hardware cache broadcasts one address to an entire
//     warp in a single transaction, instead of every thread re-reading the query
//     from global memory.
//   * Size is fixed at compile time (144 bytes), trivially within the 64 KB
//     constant bank. Filled by cudaMemcpyToSymbol() in score_gpu().
//   * uint8_t so it matches the encoded record layout exactly.
// ---------------------------------------------------------------------------
__constant__ uint8_t c_query[AB_RECORD_LEN];

// 128 threads/block: a multiple of the 32-lane warp with good occupancy on
// sm_75..sm_89. Each thread does a short (144-element) integer reduction, so the
// kernel is launch/bandwidth-bound on small inputs (THEORY "GPU mapping").
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// score_kernel: one logical thread per library antibody, via a grid-stride loop
//   so a fixed-size grid still covers an arbitrarily large library.
//   Thread (blockIdx.x, threadIdx.x) starts at i = block*blockDim + thread and
//   strides by the total thread count until i >= n.
//   Memory: c_query from the constant cache; library row i from global memory
//   (consecutive threads read consecutive 144-byte rows). No shared memory or
//   atomics -- outputs are fully independent, one int score per antibody.
//   The actual scoring is delegated to the shared __host__ __device__ core, so
//   there is literally one definition of "the score" shared with the CPU path.
// ---------------------------------------------------------------------------
__global__ void score_kernel(const uint8_t* __restrict__ lib, int n,
                             int32_t* __restrict__ out) {
    const int stride = blockDim.x * gridDim.x;             // total threads in grid
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        // Row i of the library: AB_RECORD_LEN encoded residues, row-major.
        const uint8_t* lib_i = lib + static_cast<std::size_t>(i) * AB_RECORD_LEN;
        // The shared core does the CDR-weighted BLOSUM62 sum. c_query lives in
        // constant memory; passing its address lets ab_cdr_score read it through
        // the broadcast cache. Integer math -> identical to the CPU reference.
        out[i] = ab_cdr_score(c_query, lib_i);
    }
}

// ---------------------------------------------------------------------------
// score_gpu: the five canonical CUDA steps, with the query going to constant
// memory instead of a global buffer. We time ONLY the kernel (CUDA events), not
// the H2D/D2H copies (those are discussed separately in THEORY).
// ---------------------------------------------------------------------------
void score_gpu(const AntibodyLibrary& ab, std::vector<int32_t>& out, float* kernel_ms) {
    const int n = ab.n;
    out.assign(static_cast<std::size_t>(n), 0);
    const std::size_t lib_bytes = static_cast<std::size_t>(n) * AB_RECORD_LEN * sizeof(uint8_t);
    const std::size_t out_bytes = static_cast<std::size_t>(n) * sizeof(int32_t);

    // (a) Upload the query to the __constant__ symbol (a special copy that
    //     targets the constant bank rather than ordinary global memory).
    CUDA_CHECK(cudaMemcpyToSymbol(c_query, ab.query.data(), AB_RECORD_LEN * sizeof(uint8_t)));

    // (b) Allocate + upload the library, and allocate the output scores.
    uint8_t* d_lib = nullptr;   // [n*AB_RECORD_LEN] device, row-major encoded
    int32_t* d_out = nullptr;   // [n] device scores
    CUDA_CHECK(cudaMalloc(&d_lib, lib_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_lib, ab.lib.data(), lib_bytes, cudaMemcpyHostToDevice));

    // (c) Launch. Enough blocks to cover n one-thread-per-antibody, but capped so
    //     the grid stays modest; the grid-stride loop handles the remainder.
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks > 1024) blocks = 1024;   // cap: grid-stride covers any larger n
    if (blocks < 1)    blocks = 1;      // always launch at least one block
    GpuTimer timer;
    timer.start();
    score_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_lib, n, d_out);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("score_kernel");    // catch launch/exec errors immediately

    // (d) Copy scores back, then (e) free device memory.
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_lib));
    CUDA_CHECK(cudaFree(d_out));
}
