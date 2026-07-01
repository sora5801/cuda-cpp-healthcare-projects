// ===========================================================================
// src/kernels.cu  --  The GPU side: build dictionary, normalize, SGEMM, argmax
// ---------------------------------------------------------------------------
// Project 4.21 : MR Fingerprinting Reconstruction
//
// WHAT THIS FILE DOES (the GPU twin of reference_cpu.cpp)
//   gpu_reconstruct runs the four-stage MRF pipeline on the device:
//     (1) build_dict_kernel   : one thread per atom simulates + normalizes it.
//     (2) norm_sig_kernel     : one thread per voxel normalizes its signal.
//     (3) cuBLAS SGEMM        : form the whole V×D cosine-score matrix at once.
//     (4) argmax_kernel       : one thread per voxel picks the best-scoring atom.
//   Every per-element formula is the SHARED one in mrf_core.h, so the GPU builds
//   the same dictionary and normalized signals as the CPU -- which is what makes
//   the verification in main.cu meaningful.
//
// LIBRARY, NOT BLACK BOX (CLAUDE.md §6.1.6):
//   cuBLAS SGEMM computes C = alpha·op(A)·op(B) + beta·C for single-precision
//   matrices. We use it for the match step: the V×D matrix of inner products
//   between every voxel's (normalized) time course and every dictionary atom's.
//   This is the ~10^11-inner-product bottleneck the catalog names, expressed as
//   ONE dense matrix multiply -- the single most optimized routine on the GPU.
//   cuBLAS is COLUMN-MAJOR (Fortran order); our signal and dictionary matrices
//   are ROW-MAJOR. The call site below explains the exact op/leading-dim choice
//   that lets us feed row-major buffers to a column-major library with no
//   transpose copy, and what hand-rolling this GEMM would take.
//
// READ THIS AFTER: kernels.cuh, mrf_core.h, util/cuda_check.cuh, util/timer.cuh.
// Compare every routine with its serial twin in reference_cpu.cpp.
// ===========================================================================
#include "kernels.cuh"
#include "mrf_core.h"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <cublas_v2.h>     // cublasSgemm, handle/stream management
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

// cuBLAS has its own status enum; guard + explain every call (no black box).
// A failed BLAS call means the score matrix is garbage, so we abort loudly
// rather than silently returning a wrong tissue map.
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
// KERNEL 1 -- build_dict_kernel: simulate + normalize one atom per thread
// ---------------------------------------------------------------------------
// Thread d owns dictionary atom d. It runs the SHARED Bloch simulator over the
// (constant, shared-by-all) schedule for that atom's (T1, T2), writing the
// length-T fingerprint into row d of dict_norm, then L2-normalizes it in place.
// This is the classic "independent jobs" map (PATTERNS.md §1): the D atoms have
// no data dependencies, so each gets its own thread.
//
//   grid/block : 1D, block = 256 threads, grid = ceil(D / 256).
//   thread map : d = blockIdx.x*blockDim.x + threadIdx.x  -> atom d.
//   memory     : reads the schedule (alpha/tr/te, T doubles each, re-read by
//                every thread -> a constant-memory candidate; left in global for
//                teaching simplicity), the atom's (T1,T2); writes T floats to
//                its own contiguous dict row.
// ===========================================================================
__global__ void build_dict_kernel(const double* __restrict__ alpha,  // [T]
                                  const double* __restrict__ tr,      // [T]
                                  const double* __restrict__ te,      // [T]
                                  const double* __restrict__ t1,      // [D]
                                  const double* __restrict__ t2,      // [D]
                                  int T, int D,
                                  float* __restrict__ dict_norm) {    // [D*T]
    int d = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's atom
    if (d >= D) return;                              // guard the ragged block
    float* atom = dict_norm + static_cast<std::size_t>(d) * T;  // row d
    // The ONE TRUE simulator + normalize, shared with the CPU (mrf_core.h).
    mrf::simulate_atom(alpha, tr, te, T, t1[d], t2[d], atom);
    mrf::normalize_inplace(atom, T);
}

// ===========================================================================
// KERNEL 2 -- norm_sig_kernel: L2-normalize one voxel signal per thread
// ---------------------------------------------------------------------------
// Thread v owns voxel v. It reads that voxel's length-T measured signal, records
// its L2 norm (the proton-density scale) into sig_scale[v], and normalizes the
// signal in place so the upcoming SGEMM yields true cosines.
//
//   grid/block : 1D, block = 256, grid = ceil(V / 256).
//   thread map : v = blockIdx.x*blockDim.x + threadIdx.x  -> voxel v.
// ===========================================================================
__global__ void norm_sig_kernel(float* __restrict__ sig_norm,        // [V*T] in/out
                                float* __restrict__ sig_scale,        // [V] out
                                int T, int V) {
    int v = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's voxel
    if (v >= V) return;
    float* sig = sig_norm + static_cast<std::size_t>(v) * T;  // row v
    sig_scale[v] = mrf::l2_norm(sig, T);             // remember scale for PD map
    mrf::normalize_inplace(sig, T);                  // unit-energy shape
}

// ===========================================================================
// KERNEL 3 -- argmax_kernel: per voxel, pick the best-scoring dictionary atom
// ---------------------------------------------------------------------------
// After SGEMM we have the score matrix S. We stored it COLUMN-MAJOR with V rows
// and D columns (leading dimension ldc = V), so the score of (voxel v, atom d)
// lives at S[d*V + v]. Thread v walks the D scores in its row and keeps the
// argmax, breaking ties by SMALLEST atom index (strict '>') to match the CPU
// reference exactly. It then fills the MatchResult fields from the winning atom.
//
//   grid/block : 1D, block = 256, grid = ceil(V / 256).
//   thread map : v = blockIdx.x*blockDim.x + threadIdx.x  -> voxel v.
//   memory     : reads its column-strided slice of S (stride V) and the winning
//                atom's (T1,T2); writes one MatchResult. The strided read is the
//                price of the column-major SGEMM layout; on this problem size it
//                is negligible next to the SGEMM itself (THEORY.md §"GPU mapping").
// ===========================================================================
__global__ void argmax_kernel(const float* __restrict__ S,           // [V*D] col-major (ldc=V)
                             const float* __restrict__ sig_scale,     // [V]
                             const double* __restrict__ t1,           // [D]
                             const double* __restrict__ t2,           // [D]
                             int V, int D,
                             MatchResult* __restrict__ out) {         // [V]
    int v = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's voxel
    if (v >= V) return;
    int   best_atom = 0;
    float best_cos  = -2.0f;                         // below the [-1,1] range
    for (int d = 0; d < D; ++d) {
        float c = S[static_cast<std::size_t>(d) * V + v];  // col-major (v,d)
        if (c > best_cos) { best_cos = c; best_atom = d; } // strict > -> smallest-index tie-break
    }
    MatchResult r;
    r.best_atom = best_atom;
    r.score     = best_cos;
    r.t1_ms     = t1[best_atom];
    r.t2_ms     = t2[best_atom];
    r.pd        = sig_scale[v] * best_cos;           // recovered proton density
    out[v] = r;
}

// ===========================================================================
// HOST WRAPPER -- gpu_reconstruct
// ===========================================================================
void gpu_reconstruct(const MrfProblem& p,
                     std::vector<MatchResult>& out,
                     std::vector<float>& dict_norm,
                     std::vector<float>& score_row_v0,
                     GpuTimings* timings) {
    const int T = p.T, D = p.D, V = p.V;
    const std::size_t DT = static_cast<std::size_t>(D) * T;   // dictionary size
    const std::size_t VT = static_cast<std::size_t>(V) * T;   // signal size
    const std::size_t VD = static_cast<std::size_t>(V) * D;   // score size

    out.assign(static_cast<std::size_t>(V), MatchResult{});
    dict_norm.assign(DT, 0.0f);
    score_row_v0.assign(static_cast<std::size_t>(D), 0.0f);

    // ---- device buffers ---------------------------------------------------
    double *d_alpha = nullptr, *d_tr = nullptr, *d_te = nullptr;   // [T] schedule
    double *d_t1 = nullptr, *d_t2 = nullptr;                        // [D] grid
    float  *d_dict = nullptr;      // [D*T] normalized dictionary (row-major)
    float  *d_sig  = nullptr;      // [V*T] normalized voxel signals (row-major)
    float  *d_scale = nullptr;     // [V]   per-voxel signal norm (PD scale)
    float  *d_S = nullptr;         // [V*D] score matrix (column-major, ldc=V)
    MatchResult* d_out = nullptr;  // [V]   per-voxel result
    CUDA_CHECK(cudaMalloc(&d_alpha, static_cast<std::size_t>(T) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_tr,    static_cast<std::size_t>(T) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_te,    static_cast<std::size_t>(T) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_t1,    static_cast<std::size_t>(D) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_t2,    static_cast<std::size_t>(D) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_dict,  DT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sig,   VT * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_scale, static_cast<std::size_t>(V) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_S,     VD * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_out,   static_cast<std::size_t>(V) * sizeof(MatchResult)));

    // ---- upload the inputs ------------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_alpha, p.alpha.data(), static_cast<std::size_t>(T) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_tr,    p.tr.data(),    static_cast<std::size_t>(T) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_te,    p.te.data(),    static_cast<std::size_t>(T) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t1,    p.dict_t1.data(), static_cast<std::size_t>(D) * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_t2,    p.dict_t2.data(), static_cast<std::size_t>(D) * sizeof(double), cudaMemcpyHostToDevice));
    // The raw (un-normalized) voxel signals go straight into d_sig; the kernel
    // normalizes them in place (mirroring normalize_signals_cpu).
    CUDA_CHECK(cudaMemcpy(d_sig, p.signal.data(), VT * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Stage 1: build the dictionary (timed) ----------------------------
    {
        int grid = (D + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        GpuTimer t;
        t.start();
        build_dict_kernel<<<grid, THREADS_PER_BLOCK>>>(d_alpha, d_tr, d_te,
                                                       d_t1, d_t2, T, D, d_dict);
        timings->build_ms = t.stop_ms();
        CUDA_CHECK_LAST("build_dict_kernel");
    }

    // ---- Stage 2: normalize the voxel signals (timed) ---------------------
    {
        int grid = (V + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        GpuTimer t;
        t.start();
        norm_sig_kernel<<<grid, THREADS_PER_BLOCK>>>(d_sig, d_scale, T, V);
        timings->normsig_ms = t.stop_ms();
        CUDA_CHECK_LAST("norm_sig_kernel");
    }

    // ---- Stage 3a: cuBLAS SGEMM -> the whole V×D score matrix (timed) -----
    // The library call, EXPLAINED (no black box):
    //   Math we want (row-major view): S[v][d] = Σ_t sig[v][t] · dict[d][t],
    //   i.e. S[V×D] = Sig[V×T] · Dictᵀ[T×D]. Both Sig and Dict are ROW-MAJOR.
    //   cuBLAS is COLUMN-MAJOR. A row-major [V×T] buffer is BIT-IDENTICAL to a
    //   column-major [T×V] buffer; call those column-major views SigC (T rows,
    //   V cols) and DictC (T rows, D cols). In cuBLAS's world we then want
    //       Sc[V×D] = SigCᵀ[V×T] · DictC[T×D]
    //   which is SGEMM with op(A)=T, op(B)=N, A=SigC (lda=T), B=DictC (ldb=T),
    //   m=V, n=D, k=T, and C = Sc stored COLUMN-MAJOR with ldc=V. We deliberately
    //   KEEP Sc column-major (V rows, D cols): the argmax kernel then reads
    //   S[d*V + v], which is fine. alpha=1, beta=0 (pure product, overwrite C).
    //   Hand-rolling this would mean a tiled shared-memory GEMM with register
    //   blocking and bank-conflict-free loads; cuBLAS already does all of that.
    {
        cublasHandle_t handle = nullptr;
        CUBLAS_CHECK(cublasCreate(&handle));
        const float alpha = 1.0f, beta = 0.0f;
        GpuTimer t;
        t.start();
        CUBLAS_CHECK(cublasSgemm(handle,
                                 CUBLAS_OP_T, CUBLAS_OP_N,   // SigCᵀ · DictC
                                 V, D, T,                    // m, n, k
                                 &alpha,
                                 d_sig,  T,                  // A = SigC,  lda = T
                                 d_dict, T,                  // B = DictC, ldb = T
                                 &beta,
                                 d_S,    V));                // C = Sc, ldc = V (col-major)
        timings->sgemm_ms = t.stop_ms();
        CUDA_CHECK(cudaDeviceSynchronize());                // make the timing honest
        cublasDestroy(handle);
    }

    // ---- Stage 3b: per-voxel argmax over the score matrix (timed) ---------
    {
        int grid = (V + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        GpuTimer t;
        t.start();
        argmax_kernel<<<grid, THREADS_PER_BLOCK>>>(d_S, d_scale, d_t1, d_t2,
                                                   V, D, d_out);
        timings->argmax_ms = t.stop_ms();
        CUDA_CHECK_LAST("argmax_kernel");
    }

    // ---- download results for verification + reporting --------------------
    CUDA_CHECK(cudaMemcpy(out.data(), d_out,
                          static_cast<std::size_t>(V) * sizeof(MatchResult),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dict_norm.data(), d_dict, DT * sizeof(float),
                          cudaMemcpyDeviceToHost));
    // Voxel 0's full score row lives in column-major column stride V: element d
    // is at d_S[d*V + 0]. Copy it out so main.cu can spot-check the SGEMM.
    {
        std::vector<float> tmp(VD);
        CUDA_CHECK(cudaMemcpy(tmp.data(), d_S, VD * sizeof(float), cudaMemcpyDeviceToHost));
        for (int d = 0; d < D; ++d)
            score_row_v0[d] = tmp[static_cast<std::size_t>(d) * V + 0];
    }

    cudaFree(d_alpha); cudaFree(d_tr); cudaFree(d_te);
    cudaFree(d_t1); cudaFree(d_t2); cudaFree(d_dict); cudaFree(d_sig);
    cudaFree(d_scale); cudaFree(d_S); cudaFree(d_out);
}
