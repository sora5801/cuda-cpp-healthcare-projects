// ===========================================================================
// src/kernels.cu  --  Batched route-scoring kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 1.20 : Reaction Yield / Retrosynthesis Scoring
//
// This is the GPU twin of score_routes_cpu() in reference_cpu.cpp. main.cu runs
// both and asserts they agree. The per-route arithmetic is SHARED through
// route_score.h, so this file is "just" the parallel plumbing: get the model
// into constant memory, the batch into global memory, one thread per route.
// See ../THEORY.md sec "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "route_score.h"         // route_score() -- the shared __host__ __device__ core
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// ---------------------------------------------------------------------------
// The shared logistic model in CONSTANT memory.
//   * Every thread reads all NUM_FEATURES weights + the bias, NONE writes them,
//     and they are identical for the whole launch -> constant memory is ideal:
//     the constant cache broadcasts one address to an entire warp in a single
//     transaction, versus a global load per thread.
//   * Tiny and fixed-size ((NUM_FEATURES+1) * 4 bytes), trivially within the
//     64 KB constant bank. Filled by cudaMemcpyToSymbol() in score_routes_gpu().
// ---------------------------------------------------------------------------
__constant__ float c_w[NUM_FEATURES];   // shared logistic weights
__constant__ float c_b[1];              // shared logistic bias (array of 1 for symbol copy)

// 256 threads/block: a multiple of the 32-lane warp with good occupancy on
// sm_75..sm_89 (see THEORY "GPU mapping" for the occupancy reasoning).
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// score_kernel: one logical thread per candidate route, via a grid-stride loop
// so a fixed-size grid still covers an arbitrarily large batch.
//   Thread (blockIdx.x, threadIdx.x) starts at r = block*blockDim + thread and
//   strides by the total thread count until r >= n.
//   Memory: c_w/c_b from the constant cache; route r's feature block + avail[r]
//   from global memory. No shared memory or atomics: outputs are independent.
//   The actual scoring is route_score() from route_score.h -- the EXACT same
//   function the CPU reference calls, which is why the results match to ~1e-8
//   (the only difference is single-precision expf/FMA rounding; see THEORY).
// ---------------------------------------------------------------------------
__global__ void score_kernel(const float* __restrict__ feats,
                             const float* __restrict__ avail,
                             int n,
                             float* __restrict__ out) {
    const int stride = blockDim.x * gridDim.x;             // total threads in grid
    for (int r = blockIdx.x * blockDim.x + threadIdx.x; r < n; r += stride) {
        // Route r owns the feature block starting at r * ROUTE_STRIDE (row-major).
        const float* block = feats + static_cast<std::size_t>(r) * ROUTE_STRIDE;
        // Call the shared scorer with the constant-memory model. c_b is a 1-element
        // array (constant symbols are arrays); c_b[0] is the scalar bias.
        out[r] = route_score(block, avail[r], c_w, c_b[0]);
    }
}

// ---------------------------------------------------------------------------
// score_routes_gpu: the canonical CUDA steps, with the model going to constant
// memory instead of a global buffer. We time ONLY the kernel (CUDA events), not
// the H2D/D2H copies (discussed separately in THEORY).
// ---------------------------------------------------------------------------
void score_routes_gpu(const RouteSet& rs, std::vector<float>& out, float* kernel_ms) {
    const int n = rs.n;
    out.assign(static_cast<std::size_t>(n), 0.0f);
    const std::size_t feats_bytes = static_cast<std::size_t>(n) * ROUTE_STRIDE * sizeof(float);
    const std::size_t avail_bytes = static_cast<std::size_t>(n) * sizeof(float);
    const std::size_t out_bytes   = static_cast<std::size_t>(n) * sizeof(float);

    // (a) Upload the shared model to the __constant__ symbols (a special copy
    //     that targets the constant bank rather than ordinary global memory).
    CUDA_CHECK(cudaMemcpyToSymbol(c_w, rs.w, NUM_FEATURES * sizeof(float)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_b, &rs.b, sizeof(float)));

    // (b) Allocate + upload the route batch, and allocate the output scores.
    float* d_feats = nullptr;   // [n*ROUTE_STRIDE] device, row-major
    float* d_avail = nullptr;   // [n] device availability factors
    float* d_out   = nullptr;   // [n] device scores
    CUDA_CHECK(cudaMalloc(&d_feats, feats_bytes));
    CUDA_CHECK(cudaMalloc(&d_avail, avail_bytes));
    CUDA_CHECK(cudaMalloc(&d_out,   out_bytes));
    CUDA_CHECK(cudaMemcpy(d_feats, rs.feats.data(), feats_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_avail, rs.avail.data(), avail_bytes, cudaMemcpyHostToDevice));

    // (c) Launch. Enough blocks to cover n one-thread-per-route, capped so the
    //     grid stays modest; the grid-stride loop handles any larger n.
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks > 1024) blocks = 1024;   // cap: grid-stride covers any larger n
    GpuTimer timer;
    timer.start();
    score_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_feats, d_avail, n, d_out);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("score_kernel");

    // (d) Copy scores back, then (e) free device memory.
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_feats));
    CUDA_CHECK(cudaFree(d_avail));
    CUDA_CHECK(cudaFree(d_out));
}
