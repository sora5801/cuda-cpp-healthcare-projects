// ===========================================================================
// src/kernels.cu  --  Attention-MIL GPU kernels (logits + fixed-point pool)
// ---------------------------------------------------------------------------
// Project 4.11 : Digital Pathology / Whole-Slide Image Analysis
//
// GPU twin of mil_forward_cpu(): identical per-tile math (wsi.h) and the SAME
// fixed-point pooling, so main.cu's CPU-vs-GPU comparison is EXACT. The heavy
// per-tile work runs on the device; the tiny softmax + classifier reductions run
// on the host, reusing the CPU code, so both paths produce identical numbers.
// See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

#include <cmath>       // exp on device
#include <limits>      // std::numeric_limits (host softmax)
#include <vector>

// One block of 256 threads is a solid occupancy default across sm_75..sm_89.
// With N tiles we launch ceil(N/256) blocks; the last block is partly idle and
// is guarded by an `if (i >= N) return;` in each kernel.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// The frozen attention model in CONSTANT memory.
//   Every thread reads the SAME weights (V, U, w, w_c, b_c) and never writes
//   them, so constant memory is ideal: its per-SM broadcast cache serves the
//   whole warp from one fetch. We copy the host AttnParams here once per forward
//   pass with cudaMemcpyToSymbol. AttnParams is a plain-old-data struct of
//   doubles (defined in wsi.h), so it is safe to place in __constant__.
// ---------------------------------------------------------------------------
__constant__ AttnParams c_params;

// ===========================================================================
// Kernel 1 -- per-tile attention logits.
//   thread i (= blockIdx.x*blockDim.x + threadIdx.x) owns tile i:
//     logits[i] = wsi_attention_logit(features + i*FEAT_DIM, c_params)
//   Purely independent across tiles -> no shared memory, no atomics, no syncs.
//   Reads: features (global), c_params (constant). Writes: logits[i] (global).
// ===========================================================================
__global__ void attention_logits_kernel(const double* __restrict__ features,
                                        int N,
                                        double* __restrict__ logits) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's tile
    if (i >= N) return;                                   // guard ragged last block
    // The device evaluates the exact same wsi_attention_logit() the host uses.
    logits[i] = wsi_attention_logit(features + static_cast<std::size_t>(i) * FEAT_DIM, c_params);
}

// ===========================================================================
// Kernel 2 -- attention-weighted fixed-point pooling.
//   Each tile scatters its weighted feature into D shared accumulators:
//     fixed_embed[d] += wsi_quantize(attn[i] * features[i][d])   (atomicAdd)
//   Many tiles hit the same D slots -> the adds COLLIDE -> we need atomicAdd.
//   Because we quantise to INTEGERS first, the atomic adds commute, so the
//   result is order-independent (deterministic) and equals the CPU's fixed-point
//   sum bit-for-bit. atomicAdd on unsigned long long is a native integer atomic
//   (sm_35+), so no compare-and-swap loop is needed.
//   Reads: features, attn (global). Writes: fixed_embed (global, via atomics).
// ===========================================================================
__global__ void attention_pool_kernel(const double* __restrict__ features,
                                      const double* __restrict__ attn,
                                      int N,
                                      unsigned long long* __restrict__ fixed_embed) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's tile
    if (i >= N) return;
    const double a = attn[i];                             // this tile's weight
    const double* h = features + static_cast<std::size_t>(i) * FEAT_DIM;
    // Scatter D fixed-point contributions into the shared embedding accumulator.
    #pragma unroll
    for (int d = 0; d < FEAT_DIM; ++d)
        atomicAdd(&fixed_embed[d], wsi_quantize(a * h[d]));
}

// ===========================================================================
// mil_forward_gpu -- host wrapper orchestrating the two kernels + host reductions.
//   Mirrors mil_forward_cpu() step for step; the ONLY difference is WHERE each
//   step runs. Returns a MilResult identical to the CPU one.
// ===========================================================================
MilResult mil_forward_gpu(const SlideBag& bag, const AttnParams& p, float* kernel_ms) {
    const int N = bag.N;
    MilResult r;
    r.attn.assign(N, 0.0);
    r.embedding.assign(FEAT_DIM, 0.0);

    // --- Upload the frozen model to constant memory (once) -------------------
    // cudaMemcpyToSymbol copies host bytes into the __constant__ symbol c_params.
    CUDA_CHECK(cudaMemcpyToSymbol(c_params, &p, sizeof(AttnParams)));

    // --- Device buffers ------------------------------------------------------
    double* d_features = nullptr;              // [N*FEAT_DIM] tile features
    double* d_logits   = nullptr;              // [N] attention logits
    double* d_attn     = nullptr;              // [N] attention weights
    unsigned long long* d_fixed = nullptr;     // [FEAT_DIM] fixed-point embedding sum
    const std::size_t feat_bytes = static_cast<std::size_t>(N) * FEAT_DIM * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_features, feat_bytes));
    CUDA_CHECK(cudaMalloc(&d_logits,   static_cast<std::size_t>(N) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_attn,     static_cast<std::size_t>(N) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_fixed,    FEAT_DIM * sizeof(unsigned long long)));
    CUDA_CHECK(cudaMemcpy(d_features, bag.features.data(), feat_bytes, cudaMemcpyHostToDevice));

    const int grid = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    GpuTimer timer;
    timer.start();

    // --- Step 1 (device): per-tile attention logits --------------------------
    attention_logits_kernel<<<grid, THREADS_PER_BLOCK>>>(d_features, N, d_logits);
    CUDA_CHECK_LAST("attention_logits_kernel");

    // --- Step 2 (host): numerically-stable softmax over the N logits ---------
    // A tiny exact reduction on the host keeps stdout bit-reproducible and reuses
    // the very same arithmetic as the CPU reference. Bring the logits back first.
    std::vector<double> logits(N);
    CUDA_CHECK(cudaMemcpy(logits.data(), d_logits,
                          static_cast<std::size_t>(N) * sizeof(double), cudaMemcpyDeviceToHost));
    double max_logit = -std::numeric_limits<double>::infinity();
    for (int i = 0; i < N; ++i) if (logits[i] > max_logit) max_logit = logits[i];
    double denom = 0.0;
    for (int i = 0; i < N; ++i) { const double ex = std::exp(logits[i] - max_logit);
                                  r.attn[i] = ex; denom += ex; }
    for (int i = 0; i < N; ++i) r.attn[i] /= denom;

    // --- Step 3 (device): attention-weighted fixed-point pool ----------------
    CUDA_CHECK(cudaMemcpy(d_attn, r.attn.data(),
                          static_cast<std::size_t>(N) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_fixed, 0, FEAT_DIM * sizeof(unsigned long long)));
    attention_pool_kernel<<<grid, THREADS_PER_BLOCK>>>(d_features, d_attn, N, d_fixed);
    CUDA_CHECK_LAST("attention_pool_kernel");

    *kernel_ms = timer.stop_ms();   // total device time for the two kernels

    // Bring the fixed-point embedding back and dequantise (same as the CPU).
    std::vector<unsigned long long> fixed(FEAT_DIM);
    CUDA_CHECK(cudaMemcpy(fixed.data(), d_fixed,
                          FEAT_DIM * sizeof(unsigned long long), cudaMemcpyDeviceToHost));
    for (int d = 0; d < FEAT_DIM; ++d) r.embedding[d] = wsi_dequantize(fixed[d]);

    // --- Step 4 (host): classify the pooled embedding ------------------------
    r.slide_logit = wsi_slide_logit(r.embedding.data(), p);
    r.probability = wsi_sigmoid(r.slide_logit);

    // Which tile got the most attention (deterministic, ties -> lowest index).
    int best = 0;
    for (int i = 1; i < N; ++i) if (r.attn[i] > r.attn[best]) best = i;
    r.top_tile = best;

    // --- Free device memory --------------------------------------------------
    CUDA_CHECK(cudaFree(d_features));
    CUDA_CHECK(cudaFree(d_logits));
    CUDA_CHECK(cudaFree(d_attn));
    CUDA_CHECK(cudaFree(d_fixed));

    return r;
}
