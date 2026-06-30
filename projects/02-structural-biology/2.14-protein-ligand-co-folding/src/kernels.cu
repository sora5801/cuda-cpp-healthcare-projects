// ===========================================================================
// src/kernels.cu  --  GPU attention denoising kernel + ping-pong reverse loop
// ---------------------------------------------------------------------------
// Project 2.14 : Protein-Ligand Co-Folding (reduced-scope teaching version)
//
// WHAT THIS FILE DOES
//   Implements the per-step self-attention kernel (the co-folding bottleneck)
//   and the host wrapper that runs the T-step reverse-diffusion loop. The
//   per-token math (features, logits, softmax, DDIM blend) is the SHARED
//   cofold.h code that the CPU reference also uses, so the GPU reproduces the
//   CPU result. main.cu runs both and verifies agreement + recovered RMSD.
//
// THE MAPPING (one BLOCK per query token; FlashAttention-shaped)
//   For query token i (= blockIdx.x), the block's THREADS_PER_TOKEN threads
//   cooperatively scan all N key tokens in two passes:
//     pass 1  -- each thread takes a strided slice of keys, finds its local max
//                logit; a shared-memory reduction gives the block-wide max.
//     pass 2  -- each thread re-scans its slice, accumulating the softmax
//                denominator and the exp-weighted sum of the keys' NATIVE
//                targets; shared-memory reductions combine the partials.
//   Thread 0 then forms x0_hat = weighted_target / denom and DDIM-blends the
//   query toward it, writing pos_next. This is exactly denoise_token()'s math,
//   re-expressed as a cooperative block so it scales to long sequences.
//
//   WHY TWO PASSES + max-subtraction: the numerically stable softmax. Without
//   subtracting the max, exp(logit) can overflow to +inf for large logits.
//
//   DETERMINISM NOTE (honest, and the reason for our tolerance): the block
//   reduction sums partials in a TREE order, which differs from the CPU's strict
//   left-to-right order. Floating-point addition is not associative, so GPU and
//   CPU diverge by ~1e-13 per step; over `steps` denoising steps this can grow
//   to ~1e-4..1e-3 (also fed by FMA contraction differences). That is why
//   main.cu verifies to a 1e-3 PHYSICAL tolerance, not bit-exactness, and also
//   checks the science-level metric (recovered RMSD). See THEORY "Numerical
//   considerations" and PATTERNS.md §4.
//
// READ THIS AFTER: cofold.h (the math), kernels.cuh (the interface).
// ===========================================================================
#include "kernels.cuh"
#include "cofold.h"              // attention_logit, ddim_blend, D_POS
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads cooperating on one query token. 64 is a good teaching default: two
// warps (enough to hide latency on the strided key loads) while keeping the
// shared-memory reduction shallow. A real FlashAttention tunes this to the head
// dimension and the GPU; here clarity beats peak occupancy.
static constexpr int THREADS_PER_TOKEN = 64;

// ---------------------------------------------------------------------------
// block_reduce_max / block_reduce_sum: classic shared-memory tree reductions.
//   Each thread arrives with one partial value in s[tid]; after the call s[0]
//   holds the block-wide max / sum. We halve the active thread count each round
//   (stride = blockDim/2, /4, ... 1), syncing between rounds so every thread
//   sees its partner's freshly written value. O(log threads) steps.
//   `s` must be a __shared__ array of at least blockDim.x doubles.
// ---------------------------------------------------------------------------
__device__ inline void block_reduce_max(double* s, int tid, int nthreads) {
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        __syncthreads();                          // all partials from the prior round visible
        if (tid < stride && s[tid + stride] > s[tid]) s[tid] = s[tid + stride];
    }
    __syncthreads();
}
__device__ inline void block_reduce_sum(double* s, int tid, int nthreads) {
    for (int stride = nthreads / 2; stride > 0; stride >>= 1) {
        __syncthreads();
        if (tid < stride) s[tid] += s[tid + stride];
    }
    __syncthreads();
}

// ---------------------------------------------------------------------------
// attention_step_kernel: one denoising step for one query token (= one block).
//   Memory used:
//     * GLOBAL: pos / target / types (read), pos_next (write). Coalescing is
//       imperfect because each thread strides over whole tokens (D_POS apart);
//       fine at teaching scale -- THEORY discusses tiling keys into shared mem.
//     * SHARED: a single blockDim-length scratch array reused for the max and
//       the four sum reductions (denom + 3 target axes), one at a time.
//   No atomics: each block owns a distinct output token, so there are no
//   cross-block write conflicts (contrast the k-means flagship 11.09, which DOES
//   need atomics because many threads hit the same centroid).
// ---------------------------------------------------------------------------
__global__ void attention_step_kernel(CofoldParams P,
                                       const double* __restrict__ pos,
                                       const double* __restrict__ target,
                                       const int* __restrict__ types,
                                       double* __restrict__ pos_next) {
    const int i   = blockIdx.x;     // this block updates query token i
    const int tid = threadIdx.x;    // this thread's lane within the block
    const int nt  = blockDim.x;     // == THREADS_PER_TOKEN
    if (i >= P.n_tokens) return;    // guard (grid is sized to n_tokens, so rare)

    // Shared scratch for the reductions; sized at launch to nt doubles.
    extern __shared__ double s[];

    // Load this query's CURRENT position + type ONCE into registers (every
    // thread needs them; cheap, so each thread keeps its own copy).
    const double qx = pos[i * D_POS + 0];
    const double qy = pos[i * D_POS + 1];
    const double qz = pos[i * D_POS + 2];
    const int qtype = types[i];

    // ---- PASS 1: block-wide maximum logit (for stable softmax) -------------
    double local_max = -1.0e300;
    for (int j = tid; j < P.n_tokens; j += nt) {     // strided over keys
        const int same = (types[j] == qtype) ? 1 : 0;
        const double l = attention_logit(qx, qy, qz,
                                         pos[j * D_POS + 0], pos[j * D_POS + 1],
                                         pos[j * D_POS + 2], same, P);
        if (l > local_max) local_max = l;
    }
    s[tid] = local_max;
    block_reduce_max(s, tid, nt);
    const double max_logit = s[0];                   // broadcast via shared mem
    __syncthreads();                                 // before reusing s[] for sums

    // ---- PASS 2: softmax denominator + exp-weighted target sum -------------
    // Each thread keeps four running partials over its key slice.
    double denom = 0.0;
    double ax = 0.0, ay = 0.0, az = 0.0;             // sum_j w_j * x*_j
    for (int j = tid; j < P.n_tokens; j += nt) {
        const int same = (types[j] == qtype) ? 1 : 0;
        const double l = attention_logit(qx, qy, qz,
                                         pos[j * D_POS + 0], pos[j * D_POS + 1],
                                         pos[j * D_POS + 2], same, P);
        const double w = exp(l - max_logit);         // device exp; >0, <=1
        denom += w;
        ax += w * target[j * D_POS + 0];
        ay += w * target[j * D_POS + 1];
        az += w * target[j * D_POS + 2];
    }

    // Reduce the four partials in turn through the shared scratch. We reuse the
    // same s[] buffer sequentially (denom, then x, y, z) to keep shared memory
    // to a single blockDim array -- a common space-saving idiom.
    s[tid] = denom; block_reduce_sum(s, tid, nt); const double denom_t = s[0]; __syncthreads();
    s[tid] = ax;    block_reduce_sum(s, tid, nt); const double sx = s[0];      __syncthreads();
    s[tid] = ay;    block_reduce_sum(s, tid, nt); const double sy = s[0];      __syncthreads();
    s[tid] = az;    block_reduce_sum(s, tid, nt); const double sz = s[0];      __syncthreads();

    // ---- Thread 0 finalizes: x0_hat then DDIM blend ------------------------
    if (tid == 0) {
        const double inv = (denom_t > 0.0) ? (1.0 / denom_t) : 0.0;
        // ddim_blend is the SAME shared-math update the CPU uses (cofold.h).
        pos_next[i * D_POS + 0] = ddim_blend(qx, sx * inv, P.step_frac);
        pos_next[i * D_POS + 1] = ddim_blend(qy, sy * inv, P.step_frac);
        pos_next[i * D_POS + 2] = ddim_blend(qz, sz * inv, P.step_frac);
    }
}

// ---------------------------------------------------------------------------
// simulate_gpu: host wrapper. The canonical CUDA lifecycle, with the T-step
// reverse-diffusion loop timed by CUDA events:
//   (1) allocate device buffers (two position buffers for ping-pong, plus the
//       read-only target and types)
//   (2) copy initial positions / target / types host->device
//   (3) loop T steps: launch attention_step_kernel, swap the buffers
//   (4) copy the final positions device->host
//   (5) free device memory
// We time the whole denoising loop (step 3); transfers are excluded so the
// figure reflects compute, as discussed in THEORY.
// ---------------------------------------------------------------------------
void simulate_gpu(const Complex& C, std::vector<double>& pos, float* kernel_ms) {
    const CofoldParams& P = C.P;
    const int N = P.n_tokens;
    const std::size_t pos_bytes = (std::size_t)N * D_POS * sizeof(double);
    const std::size_t typ_bytes = (std::size_t)N * sizeof(int);

    // (1) Two position buffers (source/destination for ping-pong) + target + types.
    //     d_ prefix marks DEVICE pointers (CLAUDE.md §12): they must never be
    //     dereferenced on the host.
    double *d_pa = nullptr, *d_pb = nullptr, *d_target = nullptr;
    int    *d_types = nullptr;
    CUDA_CHECK(cudaMalloc(&d_pa, pos_bytes));      // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_pb, pos_bytes));
    CUDA_CHECK(cudaMalloc(&d_target, pos_bytes));
    CUDA_CHECK(cudaMalloc(&d_types, typ_bytes));

    // (2) Upload the initial noised positions, the native targets, and types.
    CUDA_CHECK(cudaMemcpy(d_pa, pos.data(), pos_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_target, C.target.data(), pos_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_types, C.types.data(), typ_bytes, cudaMemcpyHostToDevice));

    // Launch geometry: one block per token; shared memory = one double per thread.
    const dim3 grid(N);
    const dim3 block(THREADS_PER_TOKEN);
    const std::size_t shmem = (std::size_t)THREADS_PER_TOKEN * sizeof(double);

    double* src = d_pa;   // current (frozen) positions
    double* dst = d_pb;   // next positions
    GpuTimer timer;
    timer.start();
    for (int s = 0; s < P.steps; ++s) {
        attention_step_kernel<<<grid, block, shmem>>>(P, src, d_target, d_types, dst);
        double* tmp = src; src = dst; dst = tmp;   // ping-pong the buffers
    }
    *kernel_ms = timer.stop_ms();                  // GPU-measured loop time
    CUDA_CHECK_LAST("attention_step_kernel");      // catch launch + execution errors

    // (4) After the final swap, `src` holds the latest positions; copy them back.
    CUDA_CHECK(cudaMemcpy(pos.data(), src, pos_bytes, cudaMemcpyDeviceToHost));

    // (5) Release every device allocation (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_pa));
    CUDA_CHECK(cudaFree(d_pb));
    CUDA_CHECK(cudaFree(d_target));
    CUDA_CHECK(cudaFree(d_types));
}
