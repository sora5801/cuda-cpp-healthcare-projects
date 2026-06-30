// ===========================================================================
// src/kernels.cu  --  All-vs-all read-overlap kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 3.5 : De Novo Genome Assembly  (read-overlap stage)
//
// WHAT THIS FILE DOES
//   The GPU twin of overlap_cpu() in reference_cpu.cpp. One thread per read
//   PAIR computes that pair's shared-minimizer count by calling the SAME
//   __host__ __device__ routine count_shared_sorted() (assembly.h), so the GPU
//   and CPU results are bit-identical integers. main.cu runs both and asserts
//   agreement, then thresholds the scores into overlap-graph edges.
//
// READ THIS AFTER: assembly.h (the shared math), kernels.cuh (declarations).
// The launch-config and memory reasoning is in ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, eight warps to hide global-memory latency, and many blocks
// resident for occupancy. The work per thread is a short merge over two tiny
// sorted lists, so the kernel is latency-bound on small inputs (THEORY "timing").
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// overlap_kernel: one logical thread per read pair, via a grid-stride loop so a
// fixed grid covers any number of pairs P.
//   Thread-to-data map: flat pair index p = block*blockDim + thread (+ stride).
//   pair_to_ij(p) decodes the upper-triangle (i,j); we then read each read's
//   minimizer slice from the CSR buffers and intersect them.
//   Memory: d_mins / d_offset in GLOBAL memory (read-only, __restrict__). No
//   shared memory or atomics: each thread writes exactly one independent output
//   (out_score[p]) -> fully deterministic, no reduction (PATTERNS.md sec.3).
// ---------------------------------------------------------------------------
__global__ void overlap_kernel(const minimizer_t* __restrict__ d_mins,
                               const int* __restrict__ d_offset,
                               int n, long long P,
                               int* __restrict__ out_score) {
    const long long stride = (long long)blockDim.x * gridDim.x;  // total threads
    for (long long p = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         p < P; p += stride) {
        int i, j;
        pair_to_ij(p, n, &i, &j);                 // flat index -> (i<j); HD-shared
        // CSR slice lookups: read r's minimizers live at d_mins[off[r]..off[r+1]).
        const int oi = d_offset[i], oj = d_offset[j];
        const minimizer_t* a = d_mins + oi;
        const minimizer_t* b = d_mins + oj;
        const int na = d_offset[i + 1] - oi;
        const int nb = d_offset[j + 1] - oj;
        // THE shared per-pair math -- identical code path as the CPU reference.
        out_score[p] = count_shared_sorted(a, na, b, nb);
    }
}

// ---------------------------------------------------------------------------
// overlap_gpu: host wrapper. The five canonical CUDA steps:
//   (1) allocate device buffers  (2) copy CSR sketch H2D
//   (3) launch the kernel         (4) copy the [P] scores D2H
//   (5) free device memory
// We time ONLY step (3) with CUDA events (the kernel cost, not the PCIe copies;
// those are discussed separately in THEORY "GPU mapping").
// ---------------------------------------------------------------------------
void overlap_gpu(const ReadSet& rs, std::vector<int>& out_score, float* kernel_ms) {
    const int n = rs.n;
    const long long P = num_pairs(n);
    out_score.assign(static_cast<std::size_t>(P), 0);

    const std::size_t mins_bytes   = rs.mins.size()   * sizeof(minimizer_t);
    const std::size_t offset_bytes = rs.offset.size() * sizeof(int);
    const std::size_t score_bytes  = static_cast<std::size_t>(P) * sizeof(int);

    // (1) Device buffers. d_ prefix = DEVICE pointer (CLAUDE.md 12); never
    //     dereference on the host. mins can be empty only if every read is too
    //     short (we still allocate >=1 byte indirectly via cudaMalloc(0) guard).
    minimizer_t* d_mins   = nullptr;   // [total minimizers] CSR data
    int*         d_offset = nullptr;   // [n+1] CSR offsets
    int*         d_score  = nullptr;   // [P] per-pair shared counts (output)
    CUDA_CHECK(cudaMalloc(&d_mins,   mins_bytes ? mins_bytes : 1));
    CUDA_CHECK(cudaMalloc(&d_offset, offset_bytes));
    CUDA_CHECK(cudaMalloc(&d_score,  score_bytes ? score_bytes : 1));

    // (2) Copy the sketch H2D. (mins may be empty -> skip the zero-byte copy.)
    if (mins_bytes)
        CUDA_CHECK(cudaMemcpy(d_mins, rs.mins.data(), mins_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_offset, rs.offset.data(), offset_bytes, cudaMemcpyHostToDevice));

    // (3) Launch. Enough blocks to cover P one-thread-per-pair, capped so the
    //     grid stays modest; the grid-stride loop handles any larger P.
    long long want = (P + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    int blocks = (want < 1) ? 1 : (want > 65535 ? 65535 : (int)want);
    GpuTimer timer;
    timer.start();
    overlap_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_mins, d_offset, n, P, d_score);
    *kernel_ms = timer.stop_ms();           // GPU-measured kernel time
    CUDA_CHECK_LAST("overlap_kernel");      // catch launch + execution errors

    // (4) Bring the per-pair scores back to the host.
    if (score_bytes)
        CUDA_CHECK(cudaMemcpy(out_score.data(), d_score, score_bytes, cudaMemcpyDeviceToHost));

    // (5) Free everything (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_mins));
    CUDA_CHECK(cudaFree(d_offset));
    CUDA_CHECK(cudaFree(d_score));
}
