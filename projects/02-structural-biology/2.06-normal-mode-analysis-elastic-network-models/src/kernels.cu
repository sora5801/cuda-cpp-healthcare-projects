// ===========================================================================
// src/kernels.cu  --  cuSOLVER symmetric eigendecomposition wrapper
// ---------------------------------------------------------------------------
// Project 2.06 : Normal Mode Analysis / Elastic Network Models
//
// Diagonalizes the Hessian with cuSOLVER. main.cu checks the eigenvalues against
// the CPU Jacobi reference. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <cusolverDn.h>
#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

// cuSOLVER has its own status type; guard + explain every call (no black box).
#define CUSOLVER_CHECK(call)                                                    \
    do {                                                                        \
        cusolverStatus_t st__ = (call);                                        \
        if (st__ != CUSOLVER_STATUS_SUCCESS) {                                 \
            std::fprintf(stderr, "[CUSOLVER_CHECK] %s:%d -> status %d\n",      \
                         __FILE__, __LINE__, static_cast<int>(st__));          \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                       \
    } while (0)

void cusolver_eigen(const std::vector<double>& H, int n, std::vector<double>& eig,
                    std::vector<double>& evec, float* kernel_ms) {
    eig.assign(n, 0.0);
    evec.assign(static_cast<std::size_t>(n) * n, 0.0);

    cusolverDnHandle_t handle = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    double* d_A = nullptr;    // [n*n] the matrix; on output, its eigenvectors (columns)
    double* d_W = nullptr;    // [n]   eigenvalues (ascending)
    int*    d_info = nullptr; // convergence/info flag
    CUDA_CHECK(cudaMalloc(&d_A, static_cast<std::size_t>(n) * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, static_cast<std::size_t>(n) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    // H is SYMMETRIC, so its row-major and column-major layouts are identical --
    // we can upload it straight into cuSOLVER's column-major A.
    CUDA_CHECK(cudaMemcpy(d_A, H.data(), static_cast<std::size_t>(n) * n * sizeof(double),
                          cudaMemcpyHostToDevice));

    // ---- The library call, NOT a black box -------------------------------
    // cusolverDnDsyevd solves A·x = λ·x for a real symmetric A by the divide-and-
    // conquer method (O(n^3)). jobz=VECTOR also overwrites A with the orthonormal
    // eigenvectors (as columns); uplo=LOWER says read the lower triangle. It first
    // asks how much workspace it needs (bufferSize), which we allocate.
    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    const cublasFillMode_t  uplo = CUBLAS_FILL_MODE_LOWER;
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(handle, jobz, uplo, n, d_A, n, d_W, &lwork));
    double* d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, static_cast<std::size_t>(lwork) * sizeof(double)));

    GpuTimer timer;
    timer.start();
    CUSOLVER_CHECK(cusolverDnDsyevd(handle, jobz, uplo, n, d_A, n, d_W, d_work, lwork, d_info));
    *kernel_ms = timer.stop_ms();

    int info = 0;
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        std::fprintf(stderr, "[cusolver_eigen] Dsyevd failed to converge (info=%d)\n", info);
        std::exit(EXIT_FAILURE);
    }

    CUDA_CHECK(cudaMemcpy(eig.data(), d_W, static_cast<std::size_t>(n) * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(evec.data(), d_A, static_cast<std::size_t>(n) * n * sizeof(double),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_work);
    cudaFree(d_A);
    cudaFree(d_W);
    cudaFree(d_info);
    cusolverDnDestroy(handle);
}
