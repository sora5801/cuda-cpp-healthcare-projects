// ===========================================================================
// src/kernels.cu  --  GPU MI kernel (one thread per column pair) + host wrapper
// ---------------------------------------------------------------------------
// Project 2.25 : Coevolutionary Contact Prediction & MSA Transformer
//
// WHAT THIS FILE DOES
//   Implements the device kernel (mi_pairs_kernel) and the host-side glue
//   (coevolution_mi_gpu) that allocates GPU memory, moves the MSA + marginals
//   over, launches the kernel, times it, and brings the raw MI matrix back.
//   This is the GPU twin of coevolution_cpu() in reference_cpu.cpp; main.cu runs
//   both and compares them. The per-pair MATH (cv_mi_from_counts) is shared with
//   the CPU via coevolution.h, so the two agree to ~1 ulp (THEORY.md section
//   "How we verify correctness").
//
// READ THIS AFTER: kernels.cuh (the thread-mapping idea) and coevolution.h.
// ===========================================================================
#include "kernels.cuh"
#include "coevolution.h"         // CV_Q, cv_mi_from_counts (shared host/device)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// 2-D block of 16x16 = 256 threads. 256 is a solid default on sm_75..sm_89 (a
// multiple of the 32-lane warp, 8 warps to hide latency). We use a SQUARE block
// because the work is a 2-D matrix of column pairs (i along x, j along y), so a
// 2-D tiling maps naturally onto the L x L output. (Tune per GPU.)
static constexpr int BLOCK_DIM = 16;

// ---------------------------------------------------------------------------
// mi_pairs_kernel: one thread computes Mutual Information for one column pair.
//
//   LAUNCH CONFIG (set in coevolution_mi_gpu):
//     block = (BLOCK_DIM, BLOCK_DIM)             = (16,16) threads
//     grid  = (ceil(L/16), ceil(L/16)) blocks    -> covers the whole L x L matrix
//   THREAD -> DATA MAP:
//     i = blockIdx.x * blockDim.x + threadIdx.x  (column index, "x" axis)
//     j = blockIdx.y * blockDim.y + threadIdx.y  (column index, "y" axis)
//   Thread (i, j) owns matrix cell (i, j). MI is SYMMETRIC, so we only do real
//   work when i < j and write BOTH mi[i*L+j] and mi[j*L+i]; threads with i >= j
//   (the lower triangle and the diagonal) return immediately. This wastes ~half
//   the threads but keeps the mapping dead simple -- a deliberate teaching
//   tradeoff (THEORY.md notes a triangular packing as an exercise).
//
//   MEMORY: `tokens` and `single` are read from GLOBAL memory; the joint-count
//   table `pair` lives in each thread's LOCAL memory (registers spilling to
//   per-thread local). No shared memory, NO ATOMICS -- every thread writes
//   disjoint output cells, so there is nothing to synchronize and nothing whose
//   result depends on thread order (the determinism guarantee, PATTERNS.md 3).
//
//   COST per thread: O(N) to build the joint table + O(Q^2) for the MI sum.
// ---------------------------------------------------------------------------
__global__ void mi_pairs_kernel(const uint8_t* __restrict__ tokens,
                                const uint32_t* __restrict__ single,
                                int N, int L,
                                double* __restrict__ mi) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // column i (x axis)
    const int j = blockIdx.y * blockDim.y + threadIdx.y;   // column j (y axis)

    // Guard: stay in bounds AND only compute the strict upper triangle (i < j).
    // Out-of-range or lower-triangle threads have nothing to do.
    if (i >= L || j >= L || i >= j) return;

    // Per-thread JOINT count table, pair[a*Q + b] = #sequences with token a in
    // column i and token b in column j. CV_Q*CV_Q = 441 uint32 (~1.7 KB) in this
    // thread's local memory. Zero it first.
    uint32_t pair[CV_Q * CV_Q];
    #pragma unroll
    for (int t = 0; t < CV_Q * CV_Q; ++t) pair[t] = 0u;

    // Build the joint counts: walk all N sequences once. Each sequence's tokens
    // in columns i and j select one cell to increment. tokens is row-major, so
    // row r starts at r*L; we stride by L between rows.
    for (int r = 0; r < N; ++r) {
        const std::size_t base = static_cast<std::size_t>(r) * L;   // start of row r
        const int a = tokens[base + i];   // this sequence's token in column i
        const int b = tokens[base + j];   // this sequence's token in column j
        pair[a * CV_Q + b] += 1u;
    }

    // The marginals were precomputed on the host and uploaded -> just point at
    // the right rows of `single` (single[c*CV_Q + a]).
    const uint32_t* ci = single + static_cast<std::size_t>(i) * CV_Q;   // col i marginal
    const uint32_t* cj = single + static_cast<std::size_t>(j) * CV_Q;   // col j marginal

    // MI from exact integer counts -- the SAME function the CPU reference calls,
    // so the floating-point result is identical to ~1 ulp.
    const double m = cv_mi_from_counts(pair, ci, cj, N);

    // Symmetric write (disjoint cells across threads -> no races).
    mi[static_cast<std::size_t>(i) * L + j] = m;
    mi[static_cast<std::size_t>(j) * L + i] = m;
}

// ---------------------------------------------------------------------------
// build_marginals_host: count, on the host, how often each token appears in each
//   column -> [L*CV_Q]. Done once on the CPU because the marginals are tiny and
//   reused by every one of the O(L^2) threads; recomputing them per thread would
//   be wasteful. (Identical to column_counts() in reference_cpu.cpp, kept local
//   here so the GPU path is self-contained.)
// ---------------------------------------------------------------------------
static std::vector<uint32_t> build_marginals_host(const Msa& msa) {
    std::vector<uint32_t> single(static_cast<std::size_t>(msa.L) * CV_Q, 0u);
    for (int r = 0; r < msa.N; ++r) {
        const uint8_t* row = &msa.token[static_cast<std::size_t>(r) * msa.L];
        for (int c = 0; c < msa.L; ++c)
            single[static_cast<std::size_t>(c) * CV_Q + row[c]] += 1u;
    }
    return single;
}

// ---------------------------------------------------------------------------
// coevolution_mi_gpu: host wrapper. The canonical CUDA steps:
//   (1) build marginals on host       (2) allocate device buffers
//   (3) copy tokens + marginals H2D    (4) launch mi_pairs_kernel (timed)
//   (5) copy MI matrix D2H             (6) free device memory
// We time ONLY the kernel (step 4) with CUDA events so the figure is compute
// cost, not PCIe transfer cost. The diagonal of `mi` stays 0 (we zero-init the
// host vector and never write the diagonal on the device).
// ---------------------------------------------------------------------------
void coevolution_mi_gpu(const Msa& msa, std::vector<double>& mi, float* kernel_ms) {
    const int N = msa.N, L = msa.L;

    // (1) Host marginals.
    const std::vector<uint32_t> single = build_marginals_host(msa);

    // Output matrix, zero-initialized (so the never-written diagonal reads 0).
    mi.assign(static_cast<std::size_t>(L) * L, 0.0);

    // (2) Device buffers.
    uint8_t*  d_tokens = nullptr;   // [N*L] MSA tokens
    uint32_t* d_single = nullptr;   // [L*CV_Q] column marginals
    double*   d_mi     = nullptr;   // [L*L] MI matrix
    const std::size_t tok_bytes = static_cast<std::size_t>(N) * L * sizeof(uint8_t);
    const std::size_t mar_bytes = static_cast<std::size_t>(L) * CV_Q * sizeof(uint32_t);
    const std::size_t mi_bytes  = static_cast<std::size_t>(L) * L * sizeof(double);
    CUDA_CHECK(cudaMalloc(&d_tokens, tok_bytes));   // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_single, mar_bytes));
    CUDA_CHECK(cudaMalloc(&d_mi,     mi_bytes));

    // (3) Upload inputs. We also upload the zero-initialized mi so the diagonal
    //     (and lower triangle written by symmetry) start clean even if any thread
    //     were skipped.
    CUDA_CHECK(cudaMemcpy(d_tokens, msa.token.data(), tok_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_single, single.data(),    mar_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mi,     mi.data(),        mi_bytes,  cudaMemcpyHostToDevice));

    // (4) Launch a 2-D grid covering the L x L matrix; time just the kernel.
    const dim3 block(BLOCK_DIM, BLOCK_DIM);
    const dim3 grid((L + BLOCK_DIM - 1) / BLOCK_DIM,
                    (L + BLOCK_DIM - 1) / BLOCK_DIM);
    GpuTimer timer;
    timer.start();
    mi_pairs_kernel<<<grid, block>>>(d_tokens, d_single, N, L, d_mi);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time
    CUDA_CHECK_LAST("mi_pairs_kernel");    // catch launch + execution errors

    // (5) Bring the MI matrix back.
    CUDA_CHECK(cudaMemcpy(mi.data(), d_mi, mi_bytes, cudaMemcpyDeviceToHost));

    // (6) Free everything (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_tokens));
    CUDA_CHECK(cudaFree(d_single));
    CUDA_CHECK(cudaFree(d_mi));
}
