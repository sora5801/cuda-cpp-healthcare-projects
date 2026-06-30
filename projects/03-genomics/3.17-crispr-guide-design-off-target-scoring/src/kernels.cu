// ===========================================================================
// src/kernels.cu  --  GPU off-target scan kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 3.17 : CRISPR Guide Design & Off-Target Scoring
//
// WHAT THIS FILE DOES
//   Implements the device kernel (scan_kernel) and the host glue (scan_gpu) that
//   uploads the guide to constant memory + the genome to global memory, launches
//   the kernel, times it (CUDA events), and copies the per-window results back.
//   The per-window MATH is not here -- it is score_window() in cfd_score.h, the
//   exact same function the CPU reference uses, so GPU and CPU agree bit-for-bit.
//   main.cu runs both and verifies (PATTERNS.md §2, §4).
//
// READ THIS AFTER: kernels.cuh (the thread-mapping idea) and cfd_score.h (the
// shared scorer). Compare scan_kernel() here against scan_cpu() in
// reference_cpu.cpp -- same loop body, one on a thread, one in a serial loop.
// ===========================================================================
#include "kernels.cuh"
#include "cfd_score.h"           // GUIDE_LEN, PAM_LEN, score_window (host+device)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// ---------------------------------------------------------------------------
// The 20-base guide in CONSTANT memory.
//   * Every thread reads all GUIDE_LEN guide bases but NONE writes them, and they
//     are identical for the whole launch -> constant memory is ideal: its
//     hardware cache broadcasts one address to a whole warp in a single
//     transaction, instead of a global load per thread. (Same idea as the query
//     fingerprint in flagship 1.12.)
//   * Size is fixed at compile time (GUIDE_LEN bytes = 20 B), trivially within
//     the 64 KB constant bank. Filled by cudaMemcpyToSymbol() in scan_gpu().
// ---------------------------------------------------------------------------
__constant__ uint8_t c_guide[GUIDE_LEN];

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, 8 warps to hide memory latency, plenty of blocks for occupancy.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// scan_kernel: one logical thread per genome window, via a grid-stride loop so a
// fixed-size grid covers an arbitrarily long genome.
//   Thread (blockIdx.x, threadIdx.x) starts at i = block*blockDim + thread and
//   strides by the total thread count until i >= n_windows.
//   For window i: the protospacer is genome[i .. i+19], the PAM is
//   genome[i+20 .. i+22]; we hand both to the shared score_window() and write
//   the result. No shared memory or atomics: every window's output is
//   independent (reductions over these outputs happen later, on the host).
//   Memory: c_guide from the constant cache; genome bases from global memory
//   (consecutive threads read overlapping windows, so the genome stays hot in
//   the L2/L1 cache -- THEORY §"GPU mapping" discusses the access pattern).
// ---------------------------------------------------------------------------
__global__ void scan_kernel(const uint8_t* __restrict__ genome, int n_windows,
                            int* __restrict__ d_mismatch,
                            double* __restrict__ d_cfd) {
    const int stride = blockDim.x * gridDim.x;                 // total threads
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n_windows; i += stride) {
        const uint8_t* proto = genome + i;                    // 20 protospacer bases
        const uint8_t* pam   = genome + i + GUIDE_LEN;        // 3 PAM bases
        // Call the SAME scorer the CPU uses (cfd_score.h). Identical inputs and
        // identical IEEE-754 operations -> identical outputs (verified in main).
        WindowScore ws = score_window(c_guide, proto, pam);
        d_mismatch[i] = ws.mismatches;
        d_cfd[i]      = ws.cfd;
    }
}

// ---------------------------------------------------------------------------
// scan_gpu: the canonical CUDA steps, with the guide going to constant memory
// instead of a global buffer. We time ONLY the kernel (CUDA events), not the
// H2D/D2H copies (discussed separately in THEORY).
// ---------------------------------------------------------------------------
void scan_gpu(const CrisprProblem& prob, ScanResult& out, float* kernel_ms) {
    const int n = prob.n_windows;
    out.mismatches.assign(static_cast<std::size_t>(n), -1);
    out.cfd.assign(static_cast<std::size_t>(n), 0.0);

    const std::size_t genome_bytes = static_cast<std::size_t>(prob.genome_len) * sizeof(uint8_t);
    const std::size_t mm_bytes     = static_cast<std::size_t>(n) * sizeof(int);
    const std::size_t cfd_bytes    = static_cast<std::size_t>(n) * sizeof(double);

    // (a) Upload the guide to the __constant__ symbol (a special copy that
    //     targets the constant bank rather than ordinary global memory).
    CUDA_CHECK(cudaMemcpyToSymbol(c_guide, prob.guide.data(), GUIDE_LEN * sizeof(uint8_t)));

    // (b) Allocate + upload the genome, and allocate the two output arrays.
    uint8_t* d_genome   = nullptr;   // [genome_len] device, 2-bit base codes
    int*     d_mismatch = nullptr;   // [n] device mismatch counts
    double*  d_cfd      = nullptr;   // [n] device CFD scores
    CUDA_CHECK(cudaMalloc(&d_genome, genome_bytes));
    CUDA_CHECK(cudaMalloc(&d_mismatch, mm_bytes));
    CUDA_CHECK(cudaMalloc(&d_cfd, cfd_bytes));
    CUDA_CHECK(cudaMemcpy(d_genome, prob.genome.data(), genome_bytes, cudaMemcpyHostToDevice));

    // (c) Launch. Enough blocks to cover n windows one-thread-each, capped so the
    //     grid stays modest; the grid-stride loop handles any larger genome.
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks < 1)    blocks = 1;       // never launch a zero-block grid
    if (blocks > 1024) blocks = 1024;    // cap: grid-stride covers the remainder
    GpuTimer timer;
    timer.start();
    scan_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_genome, n, d_mismatch, d_cfd);
    *kernel_ms = timer.stop_ms();        // GPU-measured kernel time
    CUDA_CHECK_LAST("scan_kernel");      // catch launch + execution errors

    // (d) Copy both result arrays back to the host.
    CUDA_CHECK(cudaMemcpy(out.mismatches.data(), d_mismatch, mm_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(out.cfd.data(),        d_cfd,      cfd_bytes, cudaMemcpyDeviceToHost));

    // (e) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_genome));
    CUDA_CHECK(cudaFree(d_mismatch));
    CUDA_CHECK(cudaFree(d_cfd));
}
