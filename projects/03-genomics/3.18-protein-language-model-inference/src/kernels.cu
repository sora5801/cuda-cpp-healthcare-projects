// ===========================================================================
// src/kernels.cu  --  GPU multi-head self-attention kernels + host wrapper
// ---------------------------------------------------------------------------
// Project 3.18 : Protein Language Model Inference
//
// WHAT THIS FILE DOES
//   Implements the three device kernels that make up one self-attention block
//   (attention rows, output projection, row norms) and the host glue
//   (attention_gpu) that allocates GPU memory, launches them, times them, and
//   brings the results back. This is the GPU twin of attention_cpu() in
//   reference_cpu.cpp; main.cu runs both and compares them.
//
//   Every kernel calls the SAME per-element helpers as the CPU reference
//   (attention_math.h: proj_one, scaled_score, softmax_inplace), so the two
//   implementations run identical arithmetic and verification is meaningful.
//
// READ THIS AFTER: kernels.cuh (the thread-mapping idea), attention_math.h.
// ===========================================================================
#include "kernels.cuh"
#include "attention_math.h"      // the shared __host__ __device__ math
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cstdio>

// ---------------------------------------------------------------------------
// attention_rows_kernel
//   Block (blockIdx.x = query residue i, blockIdx.y = head h) computes attention
//   ROW i for head h and accumulates head h's output slice for residue i into Z.
//
//   THREAD MAPPING inside the block (blockDim.x == ATTN_THREADS):
//     * Phase 1 (logits): thread `t` computes the scaled logit q_i·k_j for every
//       key j it owns (j = t, t+blockDim, ...) and writes it to shared s_logit[j].
//       q_i and each k_j are recomputed on the fly via proj_one() (same as CPU).
//     * Phase 2 (softmax): the block reduces s_logit[0..L) to its max, then to
//       the sum of exp(logit-max), in shared memory -- the numerically-stable
//       softmax of attention_math.h, parallelized. After this s_logit holds the
//       attention weights A[i, j] (a probability distribution over keys).
//     * Phase 3 (value blend): thread `t` owns output dim t of this head (for
//       t < d_head) and computes  Z[i, off+t] = sum_j A[i,j] * V_h[j, t]
//       as a single sequential sum over the L keys -> deterministic, CPU-exact.
//
//   SHARED MEMORY: we keep the whole length-L logit/weight row resident (s_logit)
//   plus a small reduction scratch (s_red). At teaching scale L is tiny; a
//   production kernel (FlashAttention) tiles this so it never materializes the
//   full row -- explained in THEORY. Shared size is set at launch (see wrapper).
//
//   Only head h==0's weights are copied out to `attn` (the map we report/verify).
// ---------------------------------------------------------------------------
__global__ void attention_rows_kernel(const float* __restrict__ X, AttnConfig cfg,
                                      float* __restrict__ Z, float* __restrict__ attn) {
    const int i = blockIdx.x;   // this block's query residue
    const int h = blockIdx.y;   // this block's head
    const int L  = cfg.seq_len;
    const int D  = cfg.d_model;
    const int dh = cfg.d_head;
    const int off = h * dh;     // column offset of head h inside Q/K/V/Z
    const int t   = threadIdx.x;
    const int nth = blockDim.x;

    // Dynamic shared memory layout: [ s_logit : L floats ][ s_red : nth floats ].
    extern __shared__ float s_mem[];
    float* s_logit = s_mem;        // the length-L logit/weight row for residue i
    float* s_red   = s_mem + L;    // scratch for the block-wide reductions

    // q_i for this head lives in registers/local: recompute its d_head entries.
    // (Each thread needs the full q_i to score its keys, so every thread builds
    //  it. d_head is small at teaching scale; a tiled kernel would stage it in
    //  shared memory once -- noted in THEORY as an optimization.)
    const float* xi = &X[static_cast<std::size_t>(i) * D];

    // ---- Phase 1: logits  s_logit[j] = q_i . k_j / sqrt(d_head) -------------
    for (int j = t; j < L; j += nth) {
        const float* xj = &X[static_cast<std::size_t>(j) * D];
        // Build q_i's and k_j's d_head slices for this head via the SAME proj_one
        // the CPU uses; project column (off + c) of Wq / Wk.
        float q[64];   // d_head <= 64 at teaching scale (asserted on the host)
        float k[64];
        for (int c = 0; c < dh; ++c) {
            q[c] = proj_one(xi, D, off + c, SALT_WQ);
            k[c] = proj_one(xj, D, off + c, SALT_WK);
        }
        s_logit[j] = scaled_score(q, k, dh);
    }
    __syncthreads();   // s_logit fully populated before any reduction reads it

    // ---- Phase 2a: block max of s_logit[0..L) ------------------------------
    // Each thread first folds its strided slice into a private max, then we do a
    // tree reduction over the nth partials in s_red. This reproduces the same max
    // the serial softmax_inplace() finds (max is order-independent).
    float local = -3.4e38f;   // ~ -FLT_MAX
    for (int j = t; j < L; j += nth)
        if (s_logit[j] > local) local = s_logit[j];
    s_red[t] = local;
    __syncthreads();
    for (int stride = nth / 2; stride > 0; stride >>= 1) {
        if (t < stride && s_red[t + stride] > s_red[t]) s_red[t] = s_red[t + stride];
        __syncthreads();
    }
    const float row_max = s_red[0];   // broadcast via shared memory
    __syncthreads();

    // ---- Phase 2b: exponentiate (shift by row_max) and sum -----------------
    for (int j = t; j < L; j += nth)
        s_logit[j] = expf(s_logit[j] - row_max);   // stable: argument <= 0
    __syncthreads();
    // Sum the exponentials. We accumulate the per-thread partial in double to
    // match the CPU's double-accumulated denominator, then tree-reduce. The
    // reduction order differs from the CPU's left-to-right sum, but the values
    // are tiny exp() results so the difference is ~1e-7 (documented tolerance).
    double dlocal = 0.0;
    for (int j = t; j < L; j += nth)
        dlocal += static_cast<double>(s_logit[j]);
    s_red[t] = static_cast<float>(dlocal);
    __syncthreads();
    for (int stride = nth / 2; stride > 0; stride >>= 1) {
        if (t < stride) s_red[t] += s_red[t + stride];
        __syncthreads();
    }
    const float inv_sum = 1.0f / s_red[0];
    __syncthreads();
    // Normalize -> attention weights. s_logit now holds A[i, 0..L).
    for (int j = t; j < L; j += nth)
        s_logit[j] *= inv_sum;
    __syncthreads();

    // Export head-0's attention row for reporting/verification.
    if (h == 0)
        for (int j = t; j < L; j += nth)
            attn[static_cast<std::size_t>(i) * L + j] = s_logit[j];

    // ---- Phase 3: value blend  Z[i, off+t] = sum_j A[i,j] * V_h[j, t] -------
    // Thread t owns output dim t of this head (t < d_head). Each value V_h[j, t]
    // is projected on the fly (column off+t of Wv), exactly like the CPU.
    if (t < dh) {
        double acc = 0.0;
        for (int j = 0; j < L; ++j) {
            const float* xj = &X[static_cast<std::size_t>(j) * D];
            const float vjt = proj_one(xj, D, off + t, SALT_WV);
            acc += static_cast<double>(s_logit[j]) * vjt;
        }
        Z[static_cast<std::size_t>(i) * D + off + t] = static_cast<float>(acc);
    }
}

// ---------------------------------------------------------------------------
// output_proj_kernel: Y[i, j] = (Z row i) . Wo[:, j], one thread per element.
//   Grid is 1-D over the L*D output entries; thread `idx` owns (i, j). proj_one
//   does the dot product against column j of Wo -- the SAME call the CPU makes.
// ---------------------------------------------------------------------------
__global__ void output_proj_kernel(const float* __restrict__ Z, AttnConfig cfg,
                                   float* __restrict__ Y) {
    const int D = cfg.d_model;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(cfg.seq_len) * D;
    if (idx >= total) return;                 // guard the ragged last block
    const int i = static_cast<int>(idx / D);  // residue (row)
    const int j = static_cast<int>(idx % D);  // output feature (column)
    const float* zi = &Z[static_cast<std::size_t>(i) * D];
    Y[idx] = proj_one(zi, D, j, SALT_WO);
}

// ---------------------------------------------------------------------------
// row_norm_kernel: out_norm[i] = sqrt( sum_j Y[i,j]^2 ), one thread per residue.
//   A single thread sums a whole row sequentially in double, so the result is
//   deterministic and matches the CPU's row norm. (No atomics, no cross-thread
//   reduction order to worry about -- PATTERNS.md §3.)
// ---------------------------------------------------------------------------
__global__ void row_norm_kernel(const float* __restrict__ Y, AttnConfig cfg,
                               float* __restrict__ out_norm) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cfg.seq_len) return;
    const int D = cfg.d_model;
    const float* yi = &Y[static_cast<std::size_t>(i) * D];
    double n2 = 0.0;
    for (int j = 0; j < D; ++j)
        n2 += static_cast<double>(yi[j]) * yi[j];
    out_norm[i] = sqrtf(static_cast<float>(n2));
}

// ---------------------------------------------------------------------------
// attention_gpu: host wrapper. Uploads X, launches the three kernels, copies
// out/attn/out_norm back, computes top_attn on the host (a tiny argmax), and
// reports the summed kernel time (CUDA events). main.cu calls exactly this.
// ---------------------------------------------------------------------------
void attention_gpu(const std::vector<float>& X, const AttnConfig& cfg,
                   AttnResult& r, float* kernel_ms) {
    const int L = cfg.seq_len, D = cfg.d_model, dh = cfg.d_head;

    // The per-thread q/k register arrays in the kernel are sized 64; refuse a
    // config that would overflow them rather than corrupt memory silently.
    if (dh > 64) {
        std::fprintf(stderr, "[attention_gpu] d_head=%d exceeds the 64 cap of "
                             "the teaching kernel.\n", dh);
        std::exit(EXIT_FAILURE);
    }

    const std::size_t n_xy   = static_cast<std::size_t>(L) * D;   // X, Z, Y sizes
    const std::size_t n_attn = static_cast<std::size_t>(L) * L;   // head-0 map

    // Allocate device buffers (d_ = device pointer).
    float *d_X = nullptr, *d_Z = nullptr, *d_Y = nullptr, *d_attn = nullptr, *d_norm = nullptr;
    CUDA_CHECK(cudaMalloc(&d_X,    n_xy   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Z,    n_xy   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_Y,    n_xy   * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_attn, n_attn * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_norm, static_cast<std::size_t>(L) * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_X, X.data(), n_xy * sizeof(float), cudaMemcpyHostToDevice));

    GpuTimer timer;
    timer.start();

    // (1) Attention rows + value blend -> Z, and head-0 attention -> d_attn.
    //     Grid: (L query rows) x (H heads). Block: ATTN_THREADS threads.
    //     Dynamic shared memory: L logits + ATTN_THREADS reduction scratch.
    dim3 grid(static_cast<unsigned>(L), static_cast<unsigned>(cfg.n_heads));
    const std::size_t shmem = (static_cast<std::size_t>(L) + ATTN_THREADS) * sizeof(float);
    attention_rows_kernel<<<grid, ATTN_THREADS, shmem>>>(d_X, cfg, d_Z, d_attn);
    CUDA_CHECK_LAST("attention_rows_kernel");

    // (2) Output projection Y = Z Wo: one thread per output element.
    const int total = L * D;
    const int proj_blocks = (total + 255) / 256;
    output_proj_kernel<<<proj_blocks, 256>>>(d_Z, cfg, d_Y);
    CUDA_CHECK_LAST("output_proj_kernel");

    // (3) Per-residue output norm: one thread per residue.
    const int norm_blocks = (L + 127) / 128;
    row_norm_kernel<<<norm_blocks, 128>>>(d_Y, cfg, d_norm);
    CUDA_CHECK_LAST("row_norm_kernel");

    *kernel_ms = timer.stop_ms();   // total device time across the three kernels

    // Copy results back to the host.
    r.out.assign(n_xy, 0.0f);
    r.attn.assign(n_attn, 0.0f);
    r.out_norm.assign(L, 0.0f);
    CUDA_CHECK(cudaMemcpy(r.out.data(),      d_Y,    n_xy   * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.attn.data(),     d_attn, n_attn * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.out_norm.data(), d_norm, static_cast<std::size_t>(L) * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // top_attn[i] = argmax over keys of head-0 attention (ties -> lowest index).
    // A trivial host step (L is small); kept off the GPU for clarity.
    r.top_attn.assign(L, 0);
    for (int i = 0; i < L; ++i) {
        int best = 0;
        float bv = r.attn[static_cast<std::size_t>(i) * L + 0];
        for (int j = 1; j < L; ++j) {
            const float v = r.attn[static_cast<std::size_t>(i) * L + j];
            if (v > bv) { bv = v; best = j; }
        }
        r.top_attn[i] = best;
    }

    CUDA_CHECK(cudaFree(d_X));
    CUDA_CHECK(cudaFree(d_Z));
    CUDA_CHECK(cudaFree(d_Y));
    CUDA_CHECK(cudaFree(d_attn));
    CUDA_CHECK(cudaFree(d_norm));
}
