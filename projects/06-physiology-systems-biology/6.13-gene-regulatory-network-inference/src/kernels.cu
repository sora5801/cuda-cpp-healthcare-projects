// ===========================================================================
// src/kernels.cu  --  GPU kernels + host wrapper for GRN inference (MI + DPI)
// ---------------------------------------------------------------------------
// Project 6.13 : Gene Regulatory Network Inference (ARACNE)
//
// WHAT THIS FILE DOES
//   Three device kernels, plus the host glue that ties them together:
//     discretize_kernel  one thread per GENE -> per-gene equal-width binning
//     mi_kernel          one thread per PAIR (i<j) -> mutual information (nats)
//     dpi_kernel         one thread per EDGE (i<j) -> Data-Processing-Inequality
//   Each is the GPU twin of a function in reference_cpu.cpp and shares the exact
//   per-element math via grn.h (discretize_value, mi_from_joint), so GPU and CPU
//   agree to ~1e-12. main.cu runs both paths and verifies.
//
//   No atomics anywhere: every thread owns a DISJOINT output cell, so results
//   are deterministic and no synchronization is needed (PATTERNS.md sec 3).
//
// READ THIS AFTER: kernels.cuh (the interface + the thread-mapping idea), grn.h.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, gives the scheduler 8 warps to hide latency, and keeps many
// blocks resident for occupancy. (Tune per project/GPU; see THEORY GPU mapping.)
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// discretize_kernel : one thread per gene does that gene's equal-width binning.
//   Thread g scans its S expression values for [min,max], then maps each value
//   through the SHARED discretize_value() (grn.h) -- identical to the CPU's
//   discretize_matrix(), so the two `disc` matrices are bit-identical.
//   Launch: grid = ceil(G / block); guard g >= G.
//   Memory: reads/writes only gene g's contiguous row -> naturally coalesced
//   within a row; no cross-thread sharing, so no shared memory or atomics.
// ---------------------------------------------------------------------------
__global__ void discretize_kernel(const double* __restrict__ expr,
                                   uint8_t* __restrict__ disc,
                                   int n_genes, int n_samples) {
    int g = blockIdx.x * blockDim.x + threadIdx.x;    // this thread's gene
    if (g >= n_genes) return;                         // guard ragged last block

    const double* row = expr + static_cast<size_t>(g) * n_samples;
    uint8_t*      out = disc + static_cast<size_t>(g) * n_samples;

    // Pass 1: this gene's dynamic range [lo, hi] over all samples.
    double lo = row[0], hi = row[0];
    for (int s = 1; s < n_samples; ++s) {
        double v = row[s];
        if (v < lo) lo = v;
        if (v > hi) hi = v;
    }
    // Pass 2: map every value to a bin (same routine the CPU uses).
    for (int s = 0; s < n_samples; ++s)
        out[s] = static_cast<uint8_t>(discretize_value(row[s], lo, hi));
}

// ---------------------------------------------------------------------------
// mi_kernel : one thread computes the mutual information of ONE gene pair.
//   We enumerate unordered pairs (i<j) as a flat index p in [0, n_pairs) and
//   map it to (i,j) with the triangular formula below, then grid-stride over p.
//   Each thread:
//     * builds a PRIVATE B*B joint histogram (JOINT_CELLS ints) in local memory
//       -- integer counting is order-independent -> deterministic;
//     * calls the shared mi_from_joint() (grn.h) for the nats value;
//     * writes BOTH mi[i*G+j] and mi[j*G+i] (symmetric); disjoint per thread.
//   Launch: enough blocks to cover n_pairs (capped; grid-stride handles the rest).
//   Memory: `disc` is read-only global (small; benefits from the L2/read cache);
//   the histogram lives in per-thread local memory (JOINT_CELLS*4 = 256 B here).
// ---------------------------------------------------------------------------
__global__ void mi_kernel(const uint8_t* __restrict__ disc,
                          int n_genes, int n_samples, long long n_pairs,
                          double* __restrict__ mi) {
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long p = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < n_pairs; p += stride) {
        // ---- Map the flat pair index p -> (i, j) with i < j ---------------
        // Row i of the strict upper triangle starts at offset
        //   base(i) = i*G - i*(i+1)/2, and holds (G-1-i) entries.
        // We invert this by scanning i from 0; G is tiny (tens) so the scan is
        // cheap and avoids a floating-point sqrt (which could misround the index
        // and break determinism). THEORY sec "GPU mapping" derives the closed form.
        int i = 0;
        long long rem = p;
        while (rem >= (n_genes - 1 - i)) {    // skip whole rows until p lands
            rem -= (n_genes - 1 - i);
            ++i;
        }
        int j = i + 1 + static_cast<int>(rem);

        // ---- Build this pair's B x B joint histogram ----------------------
        const uint8_t* di = disc + static_cast<size_t>(i) * n_samples;
        const uint8_t* dj = disc + static_cast<size_t>(j) * n_samples;
        int joint[JOINT_CELLS];                       // private per-thread table
        #pragma unroll
        for (int c = 0; c < JOINT_CELLS; ++c) joint[c] = 0;
        for (int s = 0; s < n_samples; ++s)
            joint[di[s] * N_BINS + dj[s]] += 1;       // one increment per sample

        // ---- Score it with the SHARED core, write symmetrically -----------
        double val = mi_from_joint(joint, n_samples); // same math as the CPU
        mi[static_cast<size_t>(i) * n_genes + j] = val;
        mi[static_cast<size_t>(j) * n_genes + i] = val;
    }
}

// ---------------------------------------------------------------------------
// dpi_kernel : one thread per candidate edge (i<j) applies the DPI.
//   An edge survives iff it is above `mi_threshold` AND is NOT the strictly
//   weakest edge (by more than `tolerance`) of any triangle it participates in.
//   We read the FULL, unmutated MI matrix (produced by mi_kernel) so the answer
//   does not depend on evaluation order -> matches the CPU exactly.
//   Launch: one thread per pair, same flat-index scheme as mi_kernel.
// ---------------------------------------------------------------------------
__global__ void dpi_kernel(const double* __restrict__ mi,
                           int n_genes, long long n_pairs,
                           double mi_threshold, double tolerance,
                           uint8_t* __restrict__ keep) {
    const long long stride = static_cast<long long>(blockDim.x) * gridDim.x;
    for (long long p = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
         p < n_pairs; p += stride) {
        // Same p -> (i,j) inversion as mi_kernel.
        int i = 0;
        long long rem = p;
        while (rem >= (n_genes - 1 - i)) { rem -= (n_genes - 1 - i); ++i; }
        int j = i + 1 + static_cast<int>(rem);

        double wij = mi[static_cast<size_t>(i) * n_genes + j];
        uint8_t k = (wij > mi_threshold) ? 1 : 0;     // significance gate

        if (k) {                                       // only test survivors
            for (int m = 0; m < n_genes; ++m) {        // m = mediator gene
                if (m == i || m == j) continue;
                double wim = mi[static_cast<size_t>(i) * n_genes + m];
                double wjm = mi[static_cast<size_t>(j) * n_genes + m];
                if (wij < wim - tolerance && wij < wjm - tolerance) {
                    k = 0;                             // (i,j) is indirect via m
                    break;
                }
            }
        }
        keep[static_cast<size_t>(i) * n_genes + j] = k;   // symmetric write
        keep[static_cast<size_t>(j) * n_genes + i] = k;   // (disjoint per thread)
    }
}

// ---------------------------------------------------------------------------
// grn_infer_gpu : host wrapper orchestrating the three kernels.
//   The canonical CUDA shape: allocate -> upload -> launch (timed) -> download
//   -> free. We time the MI and DPI kernels separately (CUDA events) because
//   they have different cost profiles (O(G^2 S) vs O(G^3)); the discretize pass
//   is trivial and folded into setup. Diagonal MI is left 0 (cudaMemset).
// ---------------------------------------------------------------------------
void grn_infer_gpu(const GrnData& data,
                   double mi_threshold, double tolerance,
                   std::vector<double>& mi, std::vector<uint8_t>& keep,
                   float* mi_ms, float* dpi_ms) {
    const int G = data.n_genes, S = data.n_samples;
    const long long n_pairs = static_cast<long long>(G) * (G - 1) / 2;  // strict upper triangle
    const size_t mat_cells  = static_cast<size_t>(G) * G;

    mi.assign(mat_cells, 0.0);
    keep.assign(mat_cells, 0);

    // ---- Allocate device buffers -----------------------------------------
    double*  d_expr = nullptr;   // [G*S] raw expression
    uint8_t* d_disc = nullptr;   // [G*S] discretized bins
    double*  d_mi   = nullptr;   // [G*G] MI matrix
    uint8_t* d_keep = nullptr;   // [G*G] direct-edge mask
    const size_t expr_bytes = static_cast<size_t>(G) * S * sizeof(double);
    const size_t disc_bytes = static_cast<size_t>(G) * S * sizeof(uint8_t);
    const size_t mi_bytes   = mat_cells * sizeof(double);
    const size_t keep_bytes = mat_cells * sizeof(uint8_t);
    CUDA_CHECK(cudaMalloc(&d_expr, expr_bytes));
    CUDA_CHECK(cudaMalloc(&d_disc, disc_bytes));
    CUDA_CHECK(cudaMalloc(&d_mi,   mi_bytes));
    CUDA_CHECK(cudaMalloc(&d_keep, keep_bytes));

    // ---- Upload expression; zero the MI matrix (fixes the diagonal to 0) --
    CUDA_CHECK(cudaMemcpy(d_expr, data.expr.data(), expr_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_mi, 0, mi_bytes));

    // ---- (1) Discretize on device: one thread per gene -------------------
    const int gene_blocks = (G + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    discretize_kernel<<<gene_blocks, THREADS_PER_BLOCK>>>(d_expr, d_disc, G, S);
    CUDA_CHECK_LAST("discretize_kernel");

    // ---- (2) MI: one thread per gene pair (timed) ------------------------
    int pair_blocks = static_cast<int>((n_pairs + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK);
    if (pair_blocks < 1)    pair_blocks = 1;
    if (pair_blocks > 1024) pair_blocks = 1024;   // cap; grid-stride covers the rest
    GpuTimer mi_timer;
    mi_timer.start();
    mi_kernel<<<pair_blocks, THREADS_PER_BLOCK>>>(d_disc, G, S, n_pairs, d_mi);
    *mi_ms = mi_timer.stop_ms();
    CUDA_CHECK_LAST("mi_kernel");

    // ---- (3) DPI prune: one thread per edge (timed) ----------------------
    GpuTimer dpi_timer;
    dpi_timer.start();
    dpi_kernel<<<pair_blocks, THREADS_PER_BLOCK>>>(d_mi, G, n_pairs,
                                                   mi_threshold, tolerance, d_keep);
    *dpi_ms = dpi_timer.stop_ms();
    CUDA_CHECK_LAST("dpi_kernel");

    // ---- Download results, free device memory ----------------------------
    CUDA_CHECK(cudaMemcpy(mi.data(),   d_mi,   mi_bytes,   cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(keep.data(), d_keep, keep_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_expr));
    CUDA_CHECK(cudaFree(d_disc));
    CUDA_CHECK(cudaFree(d_mi));
    CUDA_CHECK(cudaFree(d_keep));
}
