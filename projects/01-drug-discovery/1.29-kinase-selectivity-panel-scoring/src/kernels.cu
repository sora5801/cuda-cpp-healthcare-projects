// ===========================================================================
// src/kernels.cu  --  Kinase panel scoring kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 1.29 : Kinase Selectivity Panel Scoring
//
// WHAT THIS FILE DOES
//   This is the GPU twin of score_panel_cpu() in reference_cpu.cpp. main.cu runs
//   both and asserts they agree EXACTLY (integer match). The kernel gives each
//   kinase one thread; the query compound rides in CONSTANT memory; the per-kinase
//   physics is the shared __host__ __device__ score_kinase() so device and host
//   compute identical integers. See ../THEORY.md "GPU mapping".
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea) and
// selectivity_core.h (the shared scoring physics this kernel calls).
// ===========================================================================
#include "kernels.cuh"
#include "selectivity_core.h"    // NFEAT, score_kinase, predicted_pK_milli, is_hit
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// ---------------------------------------------------------------------------
// The query compound's feature vector in CONSTANT memory.
//   * Every thread reads all NFEAT offers but NONE writes them, and they are the
//     same for the whole launch -> constant memory is the ideal home: its
//     hardware cache broadcasts one address to an entire warp in a single
//     transaction, versus NFEAT global loads per thread.
//   * Size is fixed at compile time (NFEAT * 4 = 32 bytes), trivially within the
//     64 KB constant bank. Filled by cudaMemcpyToSymbol() in score_panel_gpu().
// ---------------------------------------------------------------------------
__constant__ int32_t c_ligand[NFEAT];

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, gives the scheduler 8 warps to hide memory latency, and leaves
// plenty of blocks resident for occupancy. The work per thread is tiny here, so
// this kernel is latency/launch-bound on small panels -- see THEORY "honest timing".
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// score_kinase_kernel: one logical thread per kinase, via a grid-stride loop so a
// fixed-size grid still covers a panel of any size.
//   Thread (blockIdx.x, threadIdx.x) starts at i = block*blockDim + thread and
//   strides by the total thread count until i >= n.
//   Memory: c_ligand from the constant cache (broadcast); pockets[i] from global
//   memory (consecutive threads read consecutive KinasePocket structs). No shared
//   memory or atomics: each output is fully independent. The S-count is summed on
//   the host from the integer `hit` flags (deterministic; PATTERNS.md sec 3).
// ---------------------------------------------------------------------------
__global__ void score_kinase_kernel(const KinasePocket* __restrict__ pockets, int n,
                                    int32_t* __restrict__ pK_milli,
                                    int32_t* __restrict__ hit) {
    const int stride = blockDim.x * gridDim.x;             // total threads in grid
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        // Same three steps as the CPU reference, in the same order, using the same
        // shared __host__ __device__ helpers -> bit-identical integer results.
        const int32_t raw = score_kinase(c_ligand, pockets[i]);   // raw match score
        const int32_t pK  = predicted_pK_milli(raw);              // predicted pK * 1000
        pK_milli[i] = pK;
        hit[i]      = is_hit(pK) ? 1 : 0;                         // 0/1 selectivity flag
    }
}

// ---------------------------------------------------------------------------
// score_panel_gpu: the canonical CUDA steps, with the compound going to constant
// memory instead of a global buffer. We time ONLY the kernel (CUDA events), not
// the H2D/D2H copies (discussed separately in THEORY). The S-count is reduced on
// the host from the returned `hit` flags so it is exactly the CPU's integer sum.
// ---------------------------------------------------------------------------
int32_t score_panel_gpu(const KinasePanel& panel,
                        std::vector<int32_t>& pK_milli,
                        std::vector<int32_t>& hit,
                        float* kernel_ms) {
    const int n = panel.n;
    pK_milli.assign(static_cast<std::size_t>(n), 0);
    hit.assign(static_cast<std::size_t>(n), 0);
    const std::size_t pockets_bytes = static_cast<std::size_t>(n) * sizeof(KinasePocket);
    const std::size_t out_bytes     = static_cast<std::size_t>(n) * sizeof(int32_t);

    // (a) Upload the compound to the __constant__ symbol (a special copy that
    //     targets the constant bank rather than ordinary global memory).
    CUDA_CHECK(cudaMemcpyToSymbol(c_ligand, panel.ligand, NFEAT * sizeof(int32_t)));

    // (b) Allocate + upload the pockets, and allocate the two output arrays.
    KinasePocket* d_pockets = nullptr;   // [n] device, one struct per kinase
    int32_t*      d_pK      = nullptr;   // [n] device predicted pK (milli-units)
    int32_t*      d_hit     = nullptr;   // [n] device 0/1 hit flags
    CUDA_CHECK(cudaMalloc(&d_pockets, pockets_bytes));
    CUDA_CHECK(cudaMalloc(&d_pK, out_bytes));
    CUDA_CHECK(cudaMalloc(&d_hit, out_bytes));
    // KinasePocket is a POD struct, so the whole std::vector backing array copies
    // to the device verbatim in one transfer (no per-field marshalling needed).
    CUDA_CHECK(cudaMemcpy(d_pockets, panel.pockets.data(), pockets_bytes, cudaMemcpyHostToDevice));

    // (c) Launch. Enough blocks to cover n one-thread-per-kinase, capped so the
    //     grid stays modest; the grid-stride loop handles any larger panel.
    int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks > 1024) blocks = 1024;   // cap: grid-stride covers any larger n
    GpuTimer timer;
    timer.start();
    score_kinase_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_pockets, n, d_pK, d_hit);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("score_kinase_kernel");

    // (d) Copy the per-kinase results back.
    CUDA_CHECK(cudaMemcpy(pK_milli.data(), d_pK, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(hit.data(), d_hit, out_bytes, cudaMemcpyDeviceToHost));

    // (e) Free device memory (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_pockets));
    CUDA_CHECK(cudaFree(d_pK));
    CUDA_CHECK(cudaFree(d_hit));

    // Reduce the S-count on the host from the integer flags: integer addition
    // commutes, so this matches the CPU's s_count exactly regardless of order.
    int32_t s_count = 0;
    for (int i = 0; i < n; ++i) s_count += hit[static_cast<std::size_t>(i)];
    return s_count;
}
