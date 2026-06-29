// ===========================================================================
// src/kernels.cu  --  GPU batch: build Hamiltonians + batched cuSOLVER eigensolve
// ---------------------------------------------------------------------------
// Project 1.8 : Semi-Empirical & Tight-Binding Quantum Methods
//
// WHAT THIS FILE DOES (the GPU half of the pipeline)
//   1. build_hamiltonians_kernel : one thread per matrix element fills the whole
//      batch's padded Huckel matrices, calling the SHARED tb_hamiltonian_entry()
//      so the device matrices equal the host's bit-for-bit.
//   2. tb_solve_batch_gpu : host wrapper that uploads the adjacency cube, runs
//      the build kernel, then diagonalises EVERY matrix in the batch with ONE
//      cuSOLVER call (cusolverDnDsyevjBatched), and copies the eigenvalues back.
//
//   This combines two patterns from docs/PATTERNS.md: a custom batched kernel
//   (independent jobs) and a batched dense-linear-algebra library call.
//
// READ THIS AFTER: kernels.cuh, tight_binding.h.  Pairs with: main.cu.
// ===========================================================================
#include "kernels.cuh"
#include "tight_binding.h"        // tb_hamiltonian_entry (shared __host__ __device__)
#include "util/cuda_check.cuh"    // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"         // GpuTimer

#include <cusolverDn.h>           // cusolverDnDsyevjBatched and friends
#include <cstddef>                // std::size_t
#include <cstdio>                 // std::fprintf
#include <cstdlib>                // std::exit
#include <vector>

// ---------------------------------------------------------------------------
// CUSOLVER_CHECK : the cuSOLVER analogue of CUDA_CHECK. cuSOLVER returns its own
//   cusolverStatus_t (NOT a cudaError_t), so it needs its own guard. We print
//   the failing file/line/status and abort -- a failed factorisation makes every
//   downstream number meaningless, so continuing would only mislead the learner.
// ---------------------------------------------------------------------------
#define CUSOLVER_CHECK(call)                                                     \
    do {                                                                         \
        cusolverStatus_t st__ = (call);                                          \
        if (st__ != CUSOLVER_STATUS_SUCCESS) {                                   \
            std::fprintf(stderr, "[CUSOLVER_CHECK] %s:%d -> status %d\n",        \
                         __FILE__, __LINE__, static_cast<int>(st__));            \
            std::exit(EXIT_FAILURE);                                             \
        }                                                                        \
    } while (0)

// ===========================================================================
// build_hamiltonians_kernel  --  one thread fills one matrix element
// ---------------------------------------------------------------------------
// LAUNCH CONFIG (chosen in tb_solve_batch_gpu):
//   block = (16, 16, 1)  -> 256 threads, a good occupancy default on sm_75..89.
//   grid  = (ceil(n/16), ceil(n/16), num_mol) -> blockIdx.z picks the molecule,
//           the (x,y) tile covers the n x n matrix.
//   Thread (mol = blockIdx.z, i = y, j = x) writes d_H[mol*n*n + i*n + j].
//
// MEMORY: pure global-memory reads (adjacency) and writes (matrix). No shared
//   memory or atomics are needed -- every output element is written by exactly
//   one thread, so there are no races and the result is deterministic.
//
// WHY a kernel at all (vs. building on the host and copying): building on-device
//   demonstrates the catalog's "parallelise the sparse Hamiltonian construction
//   over molecule batches" and avoids a large host->device matrix copy; we only
//   upload the much smaller byte adjacency. It also leaves the matrices on the
//   GPU right where cuSOLVER wants them.
// ===========================================================================
__global__ void build_hamiltonians_kernel(const unsigned char* __restrict__ d_adj,
                                          const int* __restrict__ d_nreal,
                                          double* __restrict__ d_H,
                                          int num_mol, int n) {
    const int j   = blockIdx.x * blockDim.x + threadIdx.x;   // column
    const int i   = blockIdx.y * blockDim.y + threadIdx.y;   // row
    const int mol = blockIdx.z;                              // which molecule
    if (mol >= num_mol || i >= n || j >= n) return;          // guard ragged edges

    // This molecule's adjacency block starts at mol*n*n; tb_hamiltonian_entry()
    // is the exact same function the CPU calls, so H matches the reference. We
    // pass n_real so the kernel applies the big padding diagonal to isolated
    // padding atoms (index >= n_real), identically to the host build.
    const unsigned char* A = d_adj + (std::size_t)mol * n * n;
    const int n_real = d_nreal[mol];
    d_H[(std::size_t)mol * n * n + (std::size_t)i * n + j] =
        tb_hamiltonian_entry(i, j, A, n, n_real);
}

// ===========================================================================
// tb_solve_batch_gpu  --  the host entry point main.cu calls
// ===========================================================================
void tb_solve_batch_gpu(const std::vector<unsigned char>& adj,
                        const std::vector<int>& n_real, int num_mol, int n,
                        std::vector<double>& eval,
                        float* build_ms, float* solve_ms) {
    const std::size_t mat_elems = (std::size_t)n * n;        // elements per matrix
    const std::size_t cube      = (std::size_t)num_mol * mat_elems;

    // ---- 1. Device buffers -------------------------------------------------
    unsigned char* d_adj   = nullptr; // [num_mol*n*n] adjacency bytes
    int*           d_nreal = nullptr; // [num_mol]     true atom count per molecule
    double*        d_H     = nullptr; // [num_mol*n*n] matrices; OVERWRITTEN with
                                      //   eigenvectors by the batched solver
    double*        d_W     = nullptr; // [num_mol*n]   eigenvalues (ascending)
    int*           d_info  = nullptr; // [num_mol]     per-matrix convergence flag
    CUDA_CHECK(cudaMalloc(&d_adj,   adj.size() * sizeof(unsigned char)));
    CUDA_CHECK(cudaMalloc(&d_nreal, (std::size_t)num_mol * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_H,     cube       * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W,     (std::size_t)num_mol * n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info,  (std::size_t)num_mol * sizeof(int)));

    // Upload the small byte adjacency + per-molecule atom counts (the big
    // matrices themselves are built on-device by the kernel below).
    CUDA_CHECK(cudaMemcpy(d_adj, adj.data(), adj.size() * sizeof(unsigned char),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nreal, n_real.data(), (std::size_t)num_mol * sizeof(int),
                          cudaMemcpyHostToDevice));

    // ---- 2. Build every Hamiltonian on the GPU (timed) --------------------
    dim3 block(16, 16, 1);
    dim3 grid((n + block.x - 1) / block.x,
              (n + block.y - 1) / block.y,
              (unsigned)num_mol);
    GpuTimer build_timer;
    build_timer.start();
    build_hamiltonians_kernel<<<grid, block>>>(d_adj, d_nreal, d_H, num_mol, n);
    *build_ms = build_timer.stop_ms();
    CUDA_CHECK_LAST("build_hamiltonians_kernel");

    // ---- 3. Batched symmetric eigensolve via cuSOLVER (NOT a black box) ----
    // cusolverDnDsyevjBatched solves A_k x = lambda x for EACH symmetric matrix
    // A_k (k = 0..num_mol-1) in one launch, using the Jacobi method (the same
    // family as our CPU reference -> the two agree to ~machine precision). With
    // jobz=VECTOR it also overwrites each A_k with that matrix's orthonormal
    // eigenvectors as columns; uplo=LOWER tells it to read the lower triangle.
    //
    // Data layout it expects: the batch is a contiguous array of column-major
    // n x n matrices, stride n*n between them. Our matrices are SYMMETRIC, so
    // row-major == column-major and we pass d_H directly with lda = n. The
    // eigenvalues come back as num_mol contiguous length-n ascending blocks in
    // d_W. Hand-rolling this would mean writing a batched Jacobi kernel with
    // shared-memory rotations and a parallel off-diagonal reduction -- a real
    // project in itself, which is exactly why we lean on the tuned library.
    cusolverDnHandle_t handle = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    // syevjInfo carries Jacobi tolerances/iteration caps. We set an explicit
    // tolerance and a sweep cap so the behaviour is reproducible and documented.
    syevjInfo_t params = nullptr;
    CUSOLVER_CHECK(cusolverDnCreateSyevjInfo(&params));
    CUSOLVER_CHECK(cusolverDnXsyevjSetTolerance(params, 1e-14));   // convergence target
    CUSOLVER_CHECK(cusolverDnXsyevjSetMaxSweeps(params, 100));     // hard iteration cap

    const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;       // want vectors too
    const cublasFillMode_t  uplo = CUBLAS_FILL_MODE_LOWER;         // read lower triangle

    // Ask the library how much scratch it needs for this (n, batch) shape.
    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDsyevjBatched_bufferSize(
        handle, jobz, uplo, n, d_H, n, d_W, &lwork, params, num_mol));
    double* d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, (std::size_t)lwork * sizeof(double)));

    GpuTimer solve_timer;
    solve_timer.start();
    CUSOLVER_CHECK(cusolverDnDsyevjBatched(
        handle, jobz, uplo, n, d_H, n, d_W, d_work, lwork, d_info, params, num_mol));
    *solve_ms = solve_timer.stop_ms();
    // The solver is asynchronous; sync so the timing and d_info are valid.
    CUDA_CHECK(cudaDeviceSynchronize());

    // ---- 4. Check per-matrix convergence ----------------------------------
    std::vector<int> info(num_mol, 0);
    CUDA_CHECK(cudaMemcpy(info.data(), d_info, (std::size_t)num_mol * sizeof(int),
                          cudaMemcpyDeviceToHost));
    for (int m = 0; m < num_mol; ++m) {
        if (info[m] != 0) {
            std::fprintf(stderr,
                "[tb_solve_batch_gpu] molecule %d: Jacobi did not converge (info=%d)\n",
                m, info[m]);
            std::exit(EXIT_FAILURE);
        }
    }

    // ---- 5. Copy eigenvalues back (num_mol blocks of n, ascending) --------
    eval.assign((std::size_t)num_mol * n, 0.0);
    CUDA_CHECK(cudaMemcpy(eval.data(), d_W, (std::size_t)num_mol * n * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // ---- 6. Tear down ------------------------------------------------------
    cudaFree(d_work);
    cudaFree(d_adj);
    cudaFree(d_nreal);
    cudaFree(d_H);
    cudaFree(d_W);
    cudaFree(d_info);
    cusolverDnDestroySyevjInfo(params);
    cusolverDnDestroy(handle);
}
