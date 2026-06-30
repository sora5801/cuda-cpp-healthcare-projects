// ===========================================================================
// src/kernels.cu  --  The GPU side: standardize, cuBLAS DGEMM GRM, assoc scan
// ---------------------------------------------------------------------------
// Project 3.11 : GWAS at Scale
//
// WHAT THIS FILE DOES (the GPU twin of reference_cpu.cpp)
//   * gpu_build_grm : (1) a kernel standardizes the raw genotype matrix G into
//                     Z on the device, then (2) cuBLAS DGEMM forms GRM=(1/M)·ZZᵀ.
//   * gpu_assoc_scan: one thread per SNP runs the single-marker regression and
//                     fills a gwas::AssocResult per SNP.
//   Every per-element formula is the SHARED one in gwas_core.h, so the GPU and
//   the CPU reference produce (near) identical numbers -- which is the whole
//   point of the verification in main.cu.
//
// LIBRARY, NOT BLACK BOX (CLAUDE.md §6.1.6):
//   cuBLAS DGEMM computes C = alpha·op(A)·op(B) + beta·C for double matrices.
//   We use it for Z·Zᵀ. cuBLAS is COLUMN-MAJOR (Fortran order); our Z is stored
//   ROW-MAJOR. The standard trick to avoid a transpose copy: a row-major N-row,
//   M-col matrix IS, byte-for-byte, the column-major M-row, N-col matrix. We
//   exploit that below and explain the exact argument choice at the call site.
//
// READ THIS AFTER: kernels.cuh, gwas_core.h, util/cuda_check.cuh, util/timer.cuh.
// Compare every routine with its serial twin in reference_cpu.cpp.
// ===========================================================================
#include "kernels.cuh"
#include "gwas_core.h"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <cublas_v2.h>     // cublasDgemm, handle/stream management
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

// cuBLAS has its own status enum; guard + explain every call (no black box).
// A failed BLAS call means the GRM is garbage, so we abort loudly rather than
// silently return wrong science.
#define CUBLAS_CHECK(call)                                                      \
    do {                                                                        \
        cublasStatus_t st__ = (call);                                          \
        if (st__ != CUBLAS_STATUS_SUCCESS) {                                   \
            std::fprintf(stderr, "[CUBLAS_CHECK] %s:%d -> status %d\n",        \
                         __FILE__, __LINE__, static_cast<int>(st__));          \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                       \
    } while (0)

// Threads per block for the 1D kernels. 256 is a solid default on sm_75..sm_89:
// a multiple of the 32-lane warp, 8 warps to hide latency, many blocks resident.
static constexpr int THREADS_PER_BLOCK = 256;

// ===========================================================================
// KERNEL 1 -- standardize: raw genotype matrix G -> standardized matrix Z
// ---------------------------------------------------------------------------
// We must center+scale each SNP COLUMN by its own (p, sd). The host precomputes
// per-SNP p and sd (cheap, done once) and passes them in; each thread then just
// applies the shared gwas::standardize formula to one matrix entry.
//
//   grid/block : a 2D grid over the (N rows) x (M cols) matrix. Block is
//                16x16 = 256 threads (a solid occupancy default on sm_75..89).
//   thread map : thread (col = blockIdx.x*bx+tx, row = blockIdx.y*by+ty) owns
//                entry Z[row][col]. The ragged edges are guarded.
//   memory     : reads g (int8) + p/sd (one double per column, broadcast),
//                writes one double to Z. Pure element-wise -> bandwidth-bound.
// ===========================================================================
__global__ void standardize_kernel(const signed char* __restrict__ g, // [N*M] int8
                                   const double* __restrict__ p,       // [M] allele freq
                                   const double* __restrict__ sd,      // [M] HWE sd
                                   int N, int M,
                                   double* __restrict__ Z) {           // [N*M] out
    int col = blockIdx.x * blockDim.x + threadIdx.x;   // SNP index   j
    int row = blockIdx.y * blockDim.y + threadIdx.y;   // individual  i
    if (row >= N || col >= M) return;                  // guard ragged blocks
    std::size_t idx = static_cast<std::size_t>(row) * M + col;  // row-major (i,j)
    // The ONE TRUE formula, shared with the CPU (gwas_core.h Section A).
    Z[idx] = gwas::standardize(static_cast<int>(g[idx]), p[col], sd[col]);
}

// ===========================================================================
// KERNEL 2 -- column_sums: per-SNP dosage sum (to get allele frequency p)
// ---------------------------------------------------------------------------
// One thread per SNP column adds the N dosages in that column. Tiny (M threads),
// runs once; we keep it simple rather than a fancy segmented reduction because
// it is utterly dwarfed by the DGEMM. Result feeds the host, which turns each
// sum into (p, sd) via the shared gwas helpers and uploads them for kernel 1.
//   thread j -> sum_j = Σ_i g[i][j]   (strided global reads down the column)
// ===========================================================================
__global__ void column_sums_kernel(const signed char* __restrict__ g,
                                   int N, int M,
                                   double* __restrict__ colsum) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;     // SNP index
    if (j >= M) return;
    double s = 0.0;
    for (int i = 0; i < N; ++i)                          // walk the column
        s += static_cast<double>(g[static_cast<std::size_t>(i) * M + j]);
    colsum[j] = s;
}

// ===========================================================================
// KERNEL 3 -- assoc_kernel: single-marker regression, ONE THREAD PER SNP
// ---------------------------------------------------------------------------
// This is the "independent jobs" pattern (PATTERNS.md §1). Thread j owns SNP j:
// it walks the N individuals, standardizes that SNP's dosage on the fly (so we
// never store a second copy of Z), accumulates the sufficient statistics
// (Σx², Σxy, Σy²), and calls the shared gwas::assoc_from_sufficient_stats to
// finish -- identical arithmetic to the CPU reference.
//
//   grid/block : 1D, block = 256 threads, grid = ceil(M / 256).
//   thread map : j = blockIdx.x*blockDim.x + threadIdx.x  -> SNP j.
//   memory     : reads the raw int8 column (strided), the centered phenotype y
//                (re-read by every thread -> a caching candidate, left simple
//                for teaching), p[j]/sd[j]; writes one AssocResult.
// ===========================================================================
__global__ void assoc_kernel(const signed char* __restrict__ g, // [N*M] int8
                             const double* __restrict__ y,       // [N] centered phenotype
                             const double* __restrict__ p,       // [M] allele freq
                             const double* __restrict__ sd,      // [M] HWE sd
                             int N, int M,
                             gwas::AssocResult* __restrict__ out) {
    int j = blockIdx.x * blockDim.x + threadIdx.x;     // this thread's SNP
    if (j >= M) return;
    double pj = p[j], sdj = sd[j];                      // this SNP's stats
    double sxx = 0.0, sxy = 0.0, syy = 0.0;             // sufficient statistics
    for (int i = 0; i < N; ++i) {
        int gi    = static_cast<int>(g[static_cast<std::size_t>(i) * M + j]);
        double x  = gwas::standardize(gi, pj, sdj);     // standardized genotype
        double yi = y[i];                                // centered phenotype
        sxx += x * x;
        sxy += x * yi;
        syy += yi * yi;
    }
    out[j] = gwas::assoc_from_sufficient_stats(sxx, sxy, syy, N);
}

// ---------------------------------------------------------------------------
// compute_p_sd_on_device: shared helper for both wrappers. Runs column_sums on
//   the device, pulls the M sums to the host, converts each to (p, sd) with the
//   shared gwas formulas, and uploads p/sd back. Factored out so the GRM path
//   and the association path derive IDENTICAL per-SNP statistics.
// ---------------------------------------------------------------------------
static void compute_p_sd_on_device(const signed char* d_g, int N, int M,
                                   double* d_colsum, double* d_p, double* d_sd) {
    int block = THREADS_PER_BLOCK, grid = (M + block - 1) / block;
    column_sums_kernel<<<grid, block>>>(d_g, N, M, d_colsum);
    CUDA_CHECK_LAST("column_sums_kernel");

    std::vector<double> h_colsum(M), h_p(M), h_sd(M);
    CUDA_CHECK(cudaMemcpy(h_colsum.data(), d_colsum,
                          static_cast<std::size_t>(M) * sizeof(double),
                          cudaMemcpyDeviceToHost));
    for (int j = 0; j < M; ++j) {                       // shared formulas
        h_p[j]  = gwas::allele_freq(h_colsum[j], N);
        h_sd[j] = gwas::hwe_sd(h_p[j]);
    }
    CUDA_CHECK(cudaMemcpy(d_p,  h_p.data(),  static_cast<std::size_t>(M) * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sd, h_sd.data(), static_cast<std::size_t>(M) * sizeof(double),
                          cudaMemcpyHostToDevice));
}

// ===========================================================================
// HOST WRAPPER 1 -- gpu_build_grm
// ===========================================================================
void gpu_build_grm(const std::vector<signed char>& geno, int N, int M,
                   std::vector<double>& grm,
                   float* standardize_ms, float* gemm_ms) {
    grm.assign(static_cast<std::size_t>(N) * N, 0.0);

    // ---- device buffers ---------------------------------------------------
    signed char* d_g = nullptr;       // [N*M] raw genotypes
    double* d_colsum = nullptr;       // [M]   per-SNP dosage sums
    double* d_p = nullptr;            // [M]   allele frequencies
    double* d_sd = nullptr;           // [M]   HWE scales
    double* d_Z = nullptr;            // [N*M] standardized matrix
    double* d_grm = nullptr;          // [N*N] relatedness matrix
    const std::size_t nm = static_cast<std::size_t>(N) * M;
    const std::size_t nn = static_cast<std::size_t>(N) * N;
    CUDA_CHECK(cudaMalloc(&d_g,      nm * sizeof(signed char)));
    CUDA_CHECK(cudaMalloc(&d_colsum, static_cast<std::size_t>(M) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p,      static_cast<std::size_t>(M) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sd,     static_cast<std::size_t>(M) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Z,      nm * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_grm,    nn * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_g, geno.data(), nm * sizeof(signed char),
                          cudaMemcpyHostToDevice));

    // ---- per-SNP (p, sd) --------------------------------------------------
    compute_p_sd_on_device(d_g, N, M, d_colsum, d_p, d_sd);

    // ---- standardize G -> Z (timed) ---------------------------------------
    {
        dim3 block(16, 16);                             // 256 threads/block
        dim3 grid((M + block.x - 1) / block.x,
                  (N + block.y - 1) / block.y);
        GpuTimer t;
        t.start();
        standardize_kernel<<<grid, block>>>(d_g, d_p, d_sd, N, M, d_Z);
        *standardize_ms = t.stop_ms();
        CUDA_CHECK_LAST("standardize_kernel");
    }

    // ---- GRM = (1/M) Z Zᵀ via cuBLAS DGEMM (timed) ------------------------
    // The library call, EXPLAINED (no black box):
    //   Math we want (row-major view): GRM[N×N] = (1/M) · Z[N×M] · Zᵀ[M×N].
    //   cuBLAS is COLUMN-MAJOR, but our Z is ROW-MAJOR. A row-major [N×M] buffer
    //   is bit-identical to a column-major [M×N] buffer; call that column-major
    //   view Zc (M rows, N cols). Then in cuBLAS's world we want
    //       GRMc[N×N] = (1/M) · Zcᵀ[N×M] · Zc[M×N]
    //   which is exactly DGEMM with op(A)=T, op(B)=N, A=B=Zc, m=n=N, k=M.
    //   Because GRM is symmetric, its row-major and column-major layouts are the
    //   same, so GRMc copies straight back into our row-major grm. lda=ldb=M
    //   (leading dim of the column-major Zc), ldc=N. The 1/M lives in alpha.
    //   Hand-rolling this would mean a tiled shared-memory GEMM with register
    //   blocking and conflict-free loads; cuBLAS already does all of that.
    {
        cublasHandle_t handle = nullptr;
        CUBLAS_CHECK(cublasCreate(&handle));
        const double alpha = 1.0 / static_cast<double>(M);  // the 1/M scaling
        const double beta  = 0.0;
        GpuTimer t;
        t.start();
        CUBLAS_CHECK(cublasDgemm(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,   // Zcᵀ · Zc
                                 N, N, M,                    // m, n, k
                                 &alpha,
                                 d_Z, M,                     // A = Zc, lda = M
                                 d_Z, M,                     // B = Zc, ldb = M
                                 &beta,
                                 d_grm, N));                 // C = GRMc, ldc = N
        *gemm_ms = t.stop_ms();
        CUDA_CHECK(cudaDeviceSynchronize());                // make timing honest
        cublasDestroy(handle);
    }

    CUDA_CHECK(cudaMemcpy(grm.data(), d_grm, nn * sizeof(double),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_g); cudaFree(d_colsum); cudaFree(d_p);
    cudaFree(d_sd); cudaFree(d_Z); cudaFree(d_grm);
}

// ===========================================================================
// HOST WRAPPER 2 -- gpu_assoc_scan
// ===========================================================================
void gpu_assoc_scan(const std::vector<signed char>& geno,
                    const std::vector<double>& y_centered, int N, int M,
                    std::vector<gwas::AssocResult>& out, float* kernel_ms) {
    out.assign(static_cast<std::size_t>(M), gwas::AssocResult{});

    signed char* d_g = nullptr;         // [N*M] raw genotypes
    double* d_y = nullptr;              // [N]   centered phenotype
    double* d_colsum = nullptr;         // [M]   per-SNP dosage sums
    double* d_p = nullptr;              // [M]   allele frequencies
    double* d_sd = nullptr;             // [M]   HWE scales
    gwas::AssocResult* d_out = nullptr; // [M]   per-SNP results
    const std::size_t nm = static_cast<std::size_t>(N) * M;
    CUDA_CHECK(cudaMalloc(&d_g,      nm * sizeof(signed char)));
    CUDA_CHECK(cudaMalloc(&d_y,      static_cast<std::size_t>(N) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_colsum, static_cast<std::size_t>(M) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_p,      static_cast<std::size_t>(M) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_sd,     static_cast<std::size_t>(M) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_out,    static_cast<std::size_t>(M) * sizeof(gwas::AssocResult)));
    CUDA_CHECK(cudaMemcpy(d_g, geno.data(), nm * sizeof(signed char),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, y_centered.data(),
                          static_cast<std::size_t>(N) * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Per-SNP (p, sd) -- same helper as gpu_build_grm so both agree exactly.
    compute_p_sd_on_device(d_g, N, M, d_colsum, d_p, d_sd);

    // The association scan: one thread per SNP (timed).
    {
        int block = THREADS_PER_BLOCK, grid = (M + block - 1) / block;
        GpuTimer t;
        t.start();
        assoc_kernel<<<grid, block>>>(d_g, d_y, d_p, d_sd, N, M, d_out);
        *kernel_ms = t.stop_ms();
        CUDA_CHECK_LAST("assoc_kernel");
    }

    CUDA_CHECK(cudaMemcpy(out.data(), d_out,
                          static_cast<std::size_t>(M) * sizeof(gwas::AssocResult),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_g); cudaFree(d_y); cudaFree(d_colsum);
    cudaFree(d_p); cudaFree(d_sd); cudaFree(d_out);
}
