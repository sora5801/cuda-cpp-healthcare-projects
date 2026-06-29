// ===========================================================================
// src/kernels.cu  --  TransE scoring kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 1.19 : Network / Polypharmacology Modeling
//
// This is the GPU twin of transe_score_cpu() in reference_cpu.cpp. main.cu runs
// both and asserts they agree EXACTLY (the per-tail math is the shared
// transe_score() from transe.h, so the float ops are identical). See
// ../THEORY.md sec "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "transe.h"              // transe_score() -- shared host/device per-tail math
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

// ---------------------------------------------------------------------------
// The query head and relation vectors in CONSTANT memory.
//   * Every thread reads all `dim` words of both vectors but NONE writes them,
//     and they are identical for the whole launch -> constant memory is the ideal
//     home: its hardware cache broadcasts one address to an entire warp in a
//     single transaction, instead of re-loading head/relation from global memory
//     per thread.
//   * Sized at the compile-time cap MAX_DIM (1 KB each), filled per launch by
//     cudaMemcpyToSymbol() with the actual `dim` floats.
// ---------------------------------------------------------------------------
__constant__ float c_head[MAX_DIM];       // the query drug embedding h
__constant__ float c_relation[MAX_DIM];   // the TARGETS relation embedding r

// 256 threads/block: a multiple of the 32-lane warp with good occupancy on
// sm_75..sm_89 (see THEORY "GPU mapping" for the occupancy reasoning).
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// transe_kernel: one logical thread per candidate tail, via a grid-stride loop
// so a fixed-size grid still covers an arbitrarily large candidate set.
//   Thread (blockIdx.x, threadIdx.x) starts at j = block*blockDim + thread and
//   strides by the total thread count until j >= n.
//   Memory: c_head / c_relation from the constant cache (broadcast warp-wide);
//   tail row j from global memory. No shared memory or atomics -- outputs are
//   fully independent (the "independent jobs" pattern).
//   It calls the SHARED transe_score() (transe.h) so the arithmetic is identical
//   to the CPU reference -> exact agreement.
// ---------------------------------------------------------------------------
__global__ void transe_kernel(const float* __restrict__ tails, int n, int dim,
                              float* __restrict__ out) {
    const int stride = blockDim.x * gridDim.x;             // total threads in grid
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < n; j += stride) {
        // Pointer to candidate tail j's embedding row (row-major, dim floats).
        const float* t = tails + static_cast<std::size_t>(j) * dim;
        // Same shared function the CPU calls; reads h, r from constant memory.
        out[j] = transe_score(c_head, c_relation, t, dim);
    }
}

// ---------------------------------------------------------------------------
// transe_score_gpu: the canonical CUDA steps, with the query vectors going to
// constant memory instead of global buffers. We time ONLY the kernel (CUDA
// events), not the H2D/D2H copies (those are discussed separately in THEORY).
// ---------------------------------------------------------------------------
void transe_score_gpu(const KnowledgeGraph& kg, std::vector<float>& out, float* kernel_ms) {
    const int n = kg.n;
    const int dim = kg.dim;
    out.assign(static_cast<std::size_t>(n), 0.0f);
    const std::size_t tail_bytes = static_cast<std::size_t>(n) * dim * sizeof(float);
    const std::size_t out_bytes  = static_cast<std::size_t>(n) * sizeof(float);
    const std::size_t vec_bytes  = static_cast<std::size_t>(dim) * sizeof(float);

    // (a) Upload head + relation to the __constant__ symbols (a special copy that
    //     targets the constant bank rather than ordinary global memory).
    CUDA_CHECK(cudaMemcpyToSymbol(c_head, kg.head.data(), vec_bytes));
    CUDA_CHECK(cudaMemcpyToSymbol(c_relation, kg.relation.data(), vec_bytes));

    // (b) Allocate + upload the candidate tails, and allocate the output scores.
    float* d_tails = nullptr;   // [n*dim] device, row-major
    float* d_out   = nullptr;   // [n] device scores
    CUDA_CHECK(cudaMalloc(&d_tails, tail_bytes));
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_tails, kg.tails.data(), tail_bytes, cudaMemcpyHostToDevice));

    // (c) Launch. Enough blocks to cover n one-thread-per-tail, but capped so the
    //     grid stays modest; the grid-stride loop handles any larger n.
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks > 1024) blocks = 1024;   // cap: grid-stride covers any larger n
    GpuTimer timer;
    timer.start();
    transe_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_tails, n, dim, d_out);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("transe_kernel");

    // (d) Copy scores back, then (e) free device memory.
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_tails));
    CUDA_CHECK(cudaFree(d_out));
}
