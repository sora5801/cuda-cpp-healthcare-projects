// ===========================================================================
// src/kernels.cu  --  Self-attention kernel + host wrapper (one head)
// ---------------------------------------------------------------------------
// Project 2.1 : Protein Structure Prediction Inference (AlphaFold-class)
//               REDUCED-SCOPE TEACHING VERSION.
//
// WHAT THIS FILE DOES
//   Implements the device kernel (attention_kernel) and the host-side glue
//   (attention_gpu) that allocates GPU memory, moves Q/K/V over, launches the
//   kernel, times it, and brings Out back. This is the GPU twin of
//   attention_cpu() in reference_cpu.cpp; main.cu runs both and compares them.
//   The per-element math (dot products, scaling, stable exp) is shared with the
//   CPU via attention_core.h, so the two are numeric twins (PATTERNS.md sec 2).
//
// THE KERNEL'S SHAPE (one block per output row)
//   Block b owns query residue i = b. Its THREADS_PER_BLOCK threads cooperate:
//     phase 1 -- each thread scores i against a strided subset of residues j and
//                writes those scores to shared memory; it also keeps the max it
//                saw, and a block reduction turns those into the ROW MAX.
//     phase 2 -- each thread exponentiates its scores (shifted by the row max)
//                and sums them; a block reduction turns those into the SOFTMAX
//                DENOMINATOR. The normalised weights now live in shared memory.
//     phase 3 -- each thread owns one output CHANNEL c and walks j=0..L-1 in
//                order, accumulating sum_j w[j]*V[j][c] -- the SAME order the CPU
//                uses, so the results match to ~FP32 epsilon (THEORY sec 6).
//
// READ THIS AFTER: kernels.cuh (the mapping idea) and attention_core.h (the math).
// ===========================================================================
#include "kernels.cuh"
#include "attention_core.h"      // D_MODEL, dot_d, scaled_score, stable_exp
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. D_MODEL (=32) is the natural width here: phase 3 maps one
// thread to one output channel, so we want at least D_MODEL threads, and a full
// warp (32) is the cleanest fit -- every lane is active, no warp divergence in
// the channel loop. We use exactly D_MODEL so threadIdx.x doubles as the channel
// index in phase 3. (For larger D_MODEL you would round up to a warp multiple.)
static constexpr int THREADS_PER_BLOCK = D_MODEL;   // = 32, one warp

// ---------------------------------------------------------------------------
// block_reduce_max / block_reduce_sum: classic shared-memory tree reductions.
//   Each thread arrives with a partial (its max / its partial sum); after the
//   call, lane 0's slot of `scratch` holds the block-wide result. We use a
//   simple, readable binary-tree reduction (halve the active set each step) and
//   __syncthreads() between steps so every thread sees the previous level's
//   writes. With THREADS_PER_BLOCK = 32 (one warp) this is short, but we keep
//   the general form because it is the canonical pattern (THEORY sec 4).
//   `scratch` must have >= blockDim.x doubles of shared memory.
// ---------------------------------------------------------------------------
__device__ inline double block_reduce_max(double val, double* scratch) {
    const int t = threadIdx.x;
    scratch[t] = val;
    __syncthreads();
    // Fold the upper half into the lower half repeatedly until one value remains.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) {
            const double other = scratch[t + stride];
            if (other > scratch[t]) scratch[t] = other;   // keep the larger
        }
        __syncthreads();   // all writes of this level must land before the next
    }
    return scratch[0];     // every thread reads the broadcast result
}

__device__ inline double block_reduce_sum(double val, double* scratch) {
    const int t = threadIdx.x;
    scratch[t] = val;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (t < stride) scratch[t] += scratch[t + stride];   // accumulate
        __syncthreads();
    }
    return scratch[0];
}

// ---------------------------------------------------------------------------
// attention_kernel: compute Out[i] for i = blockIdx.x.
//   Launch config (set in attention_gpu):
//     grid  = L blocks (one per query residue / output row)
//     block = THREADS_PER_BLOCK (= D_MODEL = 32) threads
//   Dynamic shared memory layout (one contiguous chunk, sized by the host):
//     [0            .. L-1            ]  s_w   : scores -> softmax weights (double)
//     [L            .. L+blockDim.x-1 ]  s_red : reduction scratch (double)
//   Memory: Q[i] row, all K rows, all V rows from global memory; scores and
//   reduction in shared memory; output channels accumulated in registers.
//   No atomics: each block writes a disjoint output row, each thread a disjoint
//   channel -> no write conflicts at all.
// ---------------------------------------------------------------------------
__global__ void attention_kernel(const float* __restrict__ q,
                                 const float* __restrict__ k,
                                 const float* __restrict__ v,
                                 int L,
                                 float* __restrict__ out) {
    const int i = blockIdx.x;     // this block's query residue (output row)
    const int t = threadIdx.x;    // this thread's lane in [0, blockDim.x)
    const int d = D_MODEL;

    // Carve the dynamic shared memory into the two regions described above.
    extern __shared__ double smem[];
    double* s_w   = smem;          // [L]          per-residue scores / weights
    double* s_red = smem + L;      // [blockDim.x] reduction scratch

    const float* q_i = q + static_cast<std::size_t>(i) * d;   // query row i (global)

    // ---- phase 1: scores Q[i].K[j]/sqrt(d), and the row max -----------------
    // Each thread handles residues j = t, t+blockDim.x, t+2*blockDim.x, ... (a
    // grid-stride over j) so any L is covered by blockDim.x threads.
    double local_max = -1.0e308;   // this thread's max over its strided j's
    for (int j = t; j < L; j += blockDim.x) {
        const float* k_j = k + static_cast<std::size_t>(j) * d;
        const double s = scaled_score(q_i, k_j, d);   // SAME function the CPU calls
        s_w[j] = s;                                   // stash the raw score
        if (s > local_max) local_max = s;
    }
    // Reduce the per-thread maxima into the row maximum (broadcast to all).
    const double row_max = block_reduce_max(local_max, s_red);
    __syncthreads();   // ensure all s_w[j] writes are visible before phase 2

    // ---- phase 2: exp(score - row_max), and the softmax denominator ---------
    double local_sum = 0.0;        // this thread's partial sum of exponentials
    for (int j = t; j < L; j += blockDim.x) {
        const double e = stable_exp(s_w[j], row_max);   // exp(s - max) in (0,1]
        s_w[j] = e;                                     // overwrite score with exp
        local_sum += e;
    }
    const double denom = block_reduce_sum(local_sum, s_red);   // sum_j exp(...)
    const double inv_denom = 1.0 / denom;
    __syncthreads();   // all s_w[j] now hold exponentials; safe to read in phase 3

    // ---- phase 3: out[i][c] = sum_j (exp_j/denom) * V[j][c] ------------------
    // Thread t owns output channel c = t (valid because blockDim.x == D_MODEL).
    // It walks j in INCREASING order -- identical to the CPU's accumulation
    // order -- so the two weighted sums round the same way (THEORY sec 6).
    if (t < d) {
        double acc = 0.0;          // channel accumulator, in double like the CPU
        for (int j = 0; j < L; ++j) {
            const double w = s_w[j] * inv_denom;                 // softmax weight
            const float* v_j = v + static_cast<std::size_t>(j) * d;
            acc += w * static_cast<double>(v_j[t]);              // weighted value
        }
        out[static_cast<std::size_t>(i) * d + t] = static_cast<float>(acc);
    }
}

// ---------------------------------------------------------------------------
// attention_gpu: host wrapper. The five canonical steps of a CUDA computation:
//   (1) allocate device memory  (2) copy Q/K/V host->device
//   (3) launch the kernel        (4) copy Out device->host
//   (5) free device memory
// We time ONLY step (3) with CUDA events so the figure is the kernel cost, not
// the PCIe transfer cost (discussed separately in THEORY sec 7).
// ---------------------------------------------------------------------------
void attention_gpu(const AttentionProblem& prob, std::vector<float>& out,
                   float* kernel_ms) {
    const int L = prob.L;
    const int d = prob.d;                                  // == D_MODEL
    const std::size_t mat = static_cast<std::size_t>(L) * d;
    const std::size_t bytes = mat * sizeof(float);
    out.assign(mat, 0.0f);

    // (1) Device buffers (d_ prefix = DEVICE pointer; CLAUDE.md sec 12).
    float *d_q = nullptr, *d_k = nullptr, *d_v = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_q, bytes));     // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_k, bytes));
    CUDA_CHECK(cudaMalloc(&d_v, bytes));
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // (2) Copy the three input matrices H2D (.data() is vector's backing array).
    CUDA_CHECK(cudaMemcpy(d_q, prob.q.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_k, prob.k.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_v, prob.v.data(), bytes, cudaMemcpyHostToDevice));

    // (3) Launch: one block per output row, blockDim.x = D_MODEL threads.
    //     Dynamic shared memory = L scores + blockDim.x reduction scratch, in
    //     doubles. (For very large L this could exceed the 48 KB default shared
    //     budget; the tiny teaching sample is far under it -- see THEORY sec 5.)
    const int  block = THREADS_PER_BLOCK;
    const int  grid  = L;                                  // L output rows
    const std::size_t shmem = (static_cast<std::size_t>(L) + block) * sizeof(double);
    GpuTimer timer;
    timer.start();
    attention_kernel<<<grid, block, shmem>>>(d_q, d_k, d_v, L, d_out);
    *kernel_ms = timer.stop_ms();            // GPU-measured kernel time
    CUDA_CHECK_LAST("attention_kernel");     // catch launch + execution errors

    // (4) Bring the result row-matrix back to the host.
    CUDA_CHECK(cudaMemcpy(out.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // (5) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_q));
    CUDA_CHECK(cudaFree(d_k));
    CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_out));
}
