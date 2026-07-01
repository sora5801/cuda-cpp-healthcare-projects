// ===========================================================================
// src/kernels.cu  --  The GPU side: build the lead field, then cuBLAS DGEMM it
// ---------------------------------------------------------------------------
// Project 6.18 : ECG Forward Problem & Body-Surface Potential Mapping
//
// WHAT THIS FILE DOES (the GPU twin of reference_cpu.cpp)
//   * gpu_build_lead_field : a kernel fills the transfer matrix A [L x S], one
//     GPU thread per (electrode, source) entry, each calling the SHARED
//     ecg::dipole_potential (ecg_core.h) -- so A_gpu matches A_cpu bit-for-bit.
//   * gpu_apply_forward : cuBLAS DGEMM computes Phi [L x T] = A [L x S] * X [S x T]
//     in ONE dense multiply (a batched "DGEMV per time step" = a single DGEMM).
//
//   Every per-entry formula is the SHARED one in ecg_core.h, so the GPU and CPU
//   produce (near) identical numbers -- the whole point of main.cu's verify.
//
// LIBRARY, NOT BLACK BOX (CLAUDE.md §6.1.6):
//   cuBLAS DGEMM computes C = alpha*op(A)*op(B) + beta*C for double matrices. We
//   use it for A*X. cuBLAS is COLUMN-MAJOR (Fortran order); our matrices are
//   ROW-MAJOR. The standard no-copy trick: a row-major [m x n] buffer is, byte
//   for byte, the column-major [n x m] buffer (its transpose). We exploit that
//   and explain the exact argument choice at the call site -- no transpose copies.
//
// READ THIS AFTER: kernels.cuh, ecg_core.h, util/cuda_check.cuh, util/timer.cuh.
// Compare each routine with its serial twin in reference_cpu.cpp.
// ===========================================================================
#include "kernels.cuh"
#include "ecg_core.h"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <cublas_v2.h>     // cublasDgemm, handle/stream management
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>

// cuBLAS has its own status enum; guard + explain every call (no black box).
// A failed BLAS call means Phi is garbage, so we abort loudly rather than
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

// ===========================================================================
// KERNEL -- build_lead_field_kernel: one thread per lead-field entry A[e][s]
// ---------------------------------------------------------------------------
// The build is "independent jobs": every entry of A is a self-contained
// evaluation of the dipole Green's function, reading only electrode e's position
// and source s's (position, direction). So we map a 2-D grid over the L x S
// output and let thread (col=s, row=e) own A[e][s].
//
//   grid/block : block is 16x16 = 256 threads (a solid occupancy default on
//                sm_75..sm_89). The grid covers ceil(S/16) x ceil(L/16) blocks.
//   thread map : s = blockIdx.x*blockDim.x + threadIdx.x  (source, matrix column)
//                e = blockIdx.y*blockDim.y + threadIdx.y  (electrode, matrix row)
//                -> writes A[e*S + s]. Ragged edges are guarded.
//   memory     : reads electrode[e] and src_pos[s]/src_dir[s] from global memory
//                (each a small Vec3); writes one double. Pure element-wise, so
//                the kernel is memory/compute-light -- this build is cheap and
//                the DGEMM that follows is the real work on large inputs.
// ===========================================================================
__global__ void build_lead_field_kernel(const ecg::Vec3* __restrict__ electrode, // [L]
                                         const ecg::Vec3* __restrict__ src_pos,    // [S]
                                         const ecg::Vec3* __restrict__ src_dir,    // [S]
                                         int L, int S,
                                         double* __restrict__ A) {                 // [L*S]
    int s = blockIdx.x * blockDim.x + threadIdx.x;   // source index  (column)
    int e = blockIdx.y * blockDim.y + threadIdx.y;   // electrode idx (row)
    if (e >= L || s >= S) return;                    // guard ragged blocks
    // The ONE TRUE formula, shared with the CPU reference (ecg_core.h).
    A[static_cast<std::size_t>(e) * S + s] =
        ecg::dipole_potential(electrode[e], src_pos[s], src_dir[s]);
}

// ===========================================================================
// HOST WRAPPER 1 -- gpu_build_lead_field
// ===========================================================================
void gpu_build_lead_field(const std::vector<ecg::Vec3>& electrode,
                          const std::vector<ecg::Vec3>& src_pos,
                          const std::vector<ecg::Vec3>& src_dir,
                          int L, int S,
                          std::vector<double>& A,
                          float* kernel_ms) {
    A.assign(static_cast<std::size_t>(L) * S, 0.0);

    // ---- device buffers ---------------------------------------------------
    ecg::Vec3* d_elec = nullptr;   // [L] electrode positions
    ecg::Vec3* d_spos = nullptr;   // [S] source positions
    ecg::Vec3* d_sdir = nullptr;   // [S] source directions
    double*    d_A    = nullptr;   // [L*S] lead field (output)
    const std::size_t ls = static_cast<std::size_t>(L) * S;
    CUDA_CHECK(cudaMalloc(&d_elec, static_cast<std::size_t>(L) * sizeof(ecg::Vec3)));
    CUDA_CHECK(cudaMalloc(&d_spos, static_cast<std::size_t>(S) * sizeof(ecg::Vec3)));
    CUDA_CHECK(cudaMalloc(&d_sdir, static_cast<std::size_t>(S) * sizeof(ecg::Vec3)));
    CUDA_CHECK(cudaMalloc(&d_A,    ls * sizeof(double)));

    // Copy geometry H2D. Vec3 is plain-old-data (three doubles), so a flat memcpy
    // of the vector's backing array is exactly the device layout we index above.
    CUDA_CHECK(cudaMemcpy(d_elec, electrode.data(),
                          static_cast<std::size_t>(L) * sizeof(ecg::Vec3),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_spos, src_pos.data(),
                          static_cast<std::size_t>(S) * sizeof(ecg::Vec3),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sdir, src_dir.data(),
                          static_cast<std::size_t>(S) * sizeof(ecg::Vec3),
                          cudaMemcpyHostToDevice));

    // ---- launch the build kernel (timed) ----------------------------------
    {
        dim3 block(16, 16);                            // 256 threads/block
        dim3 grid((S + block.x - 1) / block.x,         // cover S columns
                  (L + block.y - 1) / block.y);        // cover L rows
        GpuTimer t;
        t.start();
        build_lead_field_kernel<<<grid, block>>>(d_elec, d_spos, d_sdir, L, S, d_A);
        *kernel_ms = t.stop_ms();
        CUDA_CHECK_LAST("build_lead_field_kernel");
    }

    CUDA_CHECK(cudaMemcpy(A.data(), d_A, ls * sizeof(double),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_elec); cudaFree(d_spos); cudaFree(d_sdir); cudaFree(d_A);
}

// ===========================================================================
// HOST WRAPPER 2 -- gpu_apply_forward  (cuBLAS DGEMM: Phi = A * X)
// ===========================================================================
void gpu_apply_forward(const std::vector<double>& A,
                       const std::vector<double>& X,
                       int L, int S, int T,
                       std::vector<double>& Phi,
                       float* gemm_ms) {
    Phi.assign(static_cast<std::size_t>(L) * T, 0.0);

    // ---- device buffers ---------------------------------------------------
    double* d_A   = nullptr;   // [L*S] row-major lead field
    double* d_X   = nullptr;   // [S*T] row-major source strengths
    double* d_Phi = nullptr;   // [L*T] row-major body-surface potentials (out)
    const std::size_t ls = static_cast<std::size_t>(L) * S;
    const std::size_t st = static_cast<std::size_t>(S) * T;
    const std::size_t lt = static_cast<std::size_t>(L) * T;
    CUDA_CHECK(cudaMalloc(&d_A,   ls * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_X,   st * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_Phi, lt * sizeof(double)));
    CUDA_CHECK(cudaMemcpy(d_A, A.data(), ls * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_X, X.data(), st * sizeof(double), cudaMemcpyHostToDevice));

    // ---- Phi = A * X via cuBLAS DGEMM (timed), EXPLAINED (no black box) ----
    //   Math we want (row-major view): Phi[L x T] = A[L x S] * X[S x T].
    //   cuBLAS is COLUMN-MAJOR. A row-major [m x n] buffer IS a column-major
    //   [n x m] buffer (the transpose in memory). So, as cuBLAS sees them:
    //       our A  (row-major L x S)  ==  column-major S x L  =: Ac
    //       our X  (row-major S x T)  ==  column-major T x S  =: Xc
    //       our Phi(row-major L x T)  ==  column-major T x L  =: Phic
    //   Taking the transpose of the target identity: Phi^T = X^T * A^T, i.e. in
    //   column-major storage  Phic[T x L] = Xc[T x S] * Ac[S x L]. That is a
    //   plain DGEMM with op(A)=N, op(B)=N:
    //       m = T, n = L, k = S,
    //       first  matrix = Xc, lda = T   (leading dim of the T-row col-major Xc)
    //       second matrix = Ac, ldb = S   (leading dim of the S-row col-major Ac)
    //       output        = Phic, ldc = T.
    //   Because Phic is exactly our row-major Phi in memory, it copies straight
    //   back with no rearrangement. alpha = 1 (no scaling), beta = 0 (overwrite).
    //   Hand-rolling this would mean a tiled, register-blocked shared-memory GEMM
    //   with conflict-free loads; cuBLAS already does all of that, tuned per arch.
    {
        cublasHandle_t handle = nullptr;
        CUBLAS_CHECK(cublasCreate(&handle));
        const double alpha = 1.0;
        const double beta  = 0.0;
        GpuTimer t;
        t.start();
        CUBLAS_CHECK(cublasDgemm(handle,
                                 CUBLAS_OP_N, CUBLAS_OP_N,   // Xc * Ac
                                 T, L, S,                    // m, n, k
                                 &alpha,
                                 d_X, T,                     // A_arg = Xc, lda = T
                                 d_A, S,                     // B_arg = Ac, ldb = S
                                 &beta,
                                 d_Phi, T));                 // C = Phic, ldc = T
        *gemm_ms = t.stop_ms();
        CUDA_CHECK(cudaDeviceSynchronize());                // make timing honest
        cublasDestroy(handle);
    }

    CUDA_CHECK(cudaMemcpy(Phi.data(), d_Phi, lt * sizeof(double),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_A); cudaFree(d_X); cudaFree(d_Phi);
}
