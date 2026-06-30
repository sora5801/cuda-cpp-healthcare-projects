// ===========================================================================
// src/kernels.cu  --  GPU 3D-Variability (3DVA): per-element kernels + cuSOLVER
// ---------------------------------------------------------------------------
// Project 2.20 : Heterogeneous Cryo-EM Reconstruction (3D Variability)
//
// WHAT THIS FILE DOES (the GPU mirror of reference_cpu.cpp)
//   run_3dva_gpu() drives the whole PCA pipeline on the device:
//     (1) mean_kernel      -> per-voxel mean volume               (1 thread/voxel)
//     (2) gram_kernel      -> N x N covariance (Gram) matrix       (1 thread/(i,j))
//     (3) cuSOLVER Dsyevd  -> eigenvalues + eigenvectors of Gram  (library call)
//     (4) lift_kernel      -> volume-space principal component PC1 (1 thread/voxel)
//     (5) project_kernel   -> per-particle latent coordinate z     (1 thread/particle)
//   Every per-element kernel calls the SAME __host__ __device__ helpers from
//   reference_cpu.h that the CPU reference uses, so GPU and CPU compute identical
//   arithmetic (PATTERNS.md §2) and main.cu can verify to ~machine precision.
//
//   The eigenvalue problem is delegated to cuSOLVER -- documented, not a black
//   box (CLAUDE.md §6.1.6), exactly like flagship 2.06.
//
// READ THIS AFTER: kernels.cuh (the interface + the two-pattern idea),
//   reference_cpu.h (the shared math). Compare with reference_cpu.cpp.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cusolverDn.h>          // dense symmetric eigensolver (Dsyevd)
#include <cublas_v2.h>           // CUBLAS_FILL_MODE_* enum used by cuSOLVER
#include <cstdio>
#include <cstdlib>
#include <cmath>                 // sqrt, fabs on the host side
#include <vector>

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, enough warps to hide global-memory latency, and small enough to
// keep many blocks resident. (Tune per GPU; see THEORY "GPU mapping".)
static constexpr int TPB = 256;

// cuSOLVER has its own status type; guard + explain every call (no black box).
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
// (1) mean_kernel -- per-voxel mean volume, mean[v] = (1/N) sum_p vox[p][v].
//   THREAD MAP: one thread per voxel v = blockIdx.x*blockDim.x + threadIdx.x.
//   Each thread strides DOWN its column of the N x D matrix (vox[p*D + v]),
//   summing the N particle values, then divides by N. The reads are strided by
//   D (one column), which is the natural layout here; D = G^3 is small in the
//   teaching sample so this is cheap. Writes mean[v] once -- no atomics needed
//   because distinct threads own distinct voxels.
// ===========================================================================
__global__ void mean_kernel(const double* __restrict__ vox,  // [N*D] row-major
                            int N, int D,
                            double* __restrict__ mean) {      // [D] output
    const int v = blockIdx.x * blockDim.x + threadIdx.x;      // this thread's voxel
    if (v >= D) return;                                       // guard ragged block
    double acc = 0.0;                                         // sum over particles
    for (int p = 0; p < N; ++p)
        acc += vox[(size_t)p * D + v];                        // column-v entry of row p
    mean[v] = acc / (double)N;                                // mean of this voxel
}

// ===========================================================================
// (2) gram_kernel -- one thread per (i,j) entry of the N x N Gram matrix.
//   THREAD MAP: a 2-D grid; thread (col=i, row=j). Each thread computes
//   Gtil[i][j] = (1/N) * <centered particle i, centered particle j>  via the
//   SHARED gram_entry() helper -- so this matches the CPU exactly. Entries are
//   independent (PATTERNS.md §1), so no synchronization. We compute the full
//   matrix (both triangles) for code clarity; gram_entry is symmetric so the two
//   triangles agree to the bit, keeping the matrix exactly symmetric for cuSOLVER.
//   Output is row-major; since Gram is symmetric, row-major == column-major, so
//   cuSOLVER (which expects column-major) reads it correctly with no transpose.
// ===========================================================================
__global__ void gram_kernel(const double* __restrict__ vox,   // [N*D]
                            const double* __restrict__ mean,   // [D]
                            int N, int D,
                            double* __restrict__ gram) {       // [N*N] output
    const int i = blockIdx.x * blockDim.x + threadIdx.x;       // column (particle i)
    const int j = blockIdx.y * blockDim.y + threadIdx.y;       // row    (particle j)
    if (i >= N || j >= N) return;                              // guard 2-D border
    // gram_entry() is the single source of truth shared with the CPU reference.
    gram[(size_t)j * N + i] = gram_entry(vox, mean, i, j, N, D);
}

// ===========================================================================
// (4) lift_kernel -- one thread per voxel: u[v] = sum_p Xc[p][v] * w[p].
//   Given the chosen Gram eigenvector w (length N, in d_w), build the
//   (un-normalized) volume-space principal component u = X^T w voxel by voxel.
//   THREAD MAP: one thread per voxel v. Each thread loops over the N particles,
//   accumulating centered_value(p,v) * w[p]. Normalization + the sign convention
//   are done on the host afterwards (cheap, and keeps the convention identical to
//   the CPU's lift_to_volume_pc).
// ===========================================================================
__global__ void lift_kernel(const double* __restrict__ vox,   // [N*D]
                            const double* __restrict__ mean,   // [D]
                            const double* __restrict__ w,      // [N] Gram eigenvector
                            int N, int D,
                            double* __restrict__ u) {          // [D] output (un-normalized)
    const int v = blockIdx.x * blockDim.x + threadIdx.x;       // this thread's voxel
    if (v >= D) return;
    double acc = 0.0;
    for (int p = 0; p < N; ++p)
        acc += centered_value(vox, mean, p, v, D) * w[p];      // (X^T w)[v]
    u[v] = acc;
}

// ===========================================================================
// (5) project_kernel -- one thread per particle: z[p] = <Xc[p], u>.
//   Uses the SHARED project_particle() so it matches the CPU bit-for-bit.
//   THREAD MAP: one thread per particle p. Reads the whole length-D PC `u` and
//   particle p's centered row; writes one latent coordinate z[p].
// ===========================================================================
__global__ void project_kernel(const double* __restrict__ vox,  // [N*D]
                               const double* __restrict__ mean,  // [D]
                               const double* __restrict__ u,     // [D] PC (unit)
                               int N, int D,
                               double* __restrict__ z) {         // [N] output
    const int p = blockIdx.x * blockDim.x + threadIdx.x;         // this thread's particle
    if (p >= N) return;
    z[p] = project_particle(vox, mean, u, p, D);
}

// ===========================================================================
// run_3dva_gpu -- orchestrate the five stages. Allocate once, reuse buffers,
//   time each stage with CUDA events. See kernels.cuh for the output contract.
// ===========================================================================
void run_3dva_gpu(const VolumeSet& vs,
                  std::vector<double>& mean,
                  std::vector<double>& eval,
                  std::vector<double>& pc1,
                  std::vector<double>& z,
                  double& var_explained_pc1,
                  GpuTimings& t) {
    const int N = vs.N, D = vs.D;
    const size_t voxBytes  = (size_t)N * D * sizeof(double);
    const size_t meanBytes = (size_t)D * sizeof(double);
    const size_t gramBytes = (size_t)N * N * sizeof(double);

    // ---- Device buffers (d_ prefix = DEVICE memory; never deref on host) ----
    double *d_vox = nullptr, *d_mean = nullptr, *d_gram = nullptr;
    double *d_u = nullptr, *d_z = nullptr, *d_w = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vox,  voxBytes));     // the N x D volume matrix
    CUDA_CHECK(cudaMalloc(&d_mean, meanBytes));    // [D] mean volume
    CUDA_CHECK(cudaMalloc(&d_gram, gramBytes));    // [N*N] Gram matrix / eigenvectors
    CUDA_CHECK(cudaMalloc(&d_u,    meanBytes));    // [D] volume PC
    CUDA_CHECK(cudaMalloc(&d_z,    (size_t)N * sizeof(double)));  // [N] latent coords
    CUDA_CHECK(cudaMalloc(&d_w,    (size_t)N * sizeof(double)));  // [N] chosen eigenvector
    CUDA_CHECK(cudaMemcpy(d_vox, vs.vox.data(), voxBytes, cudaMemcpyHostToDevice));

    GpuTimer timer;

    // ---- (1) mean volume -------------------------------------------------
    {
        const int blocks = (D + TPB - 1) / TPB;     // cover all D voxels
        timer.start();
        mean_kernel<<<blocks, TPB>>>(d_vox, N, D, d_mean);
        t.mean_ms = timer.stop_ms();
        CUDA_CHECK_LAST("mean_kernel");
    }

    // ---- (2) N x N Gram matrix (the heavy step) --------------------------
    {
        // 2-D launch: a 16x16 block of threads, grid covers the N x N matrix.
        dim3 block(16, 16);
        dim3 grid((N + block.x - 1) / block.x, (N + block.y - 1) / block.y);
        timer.start();
        gram_kernel<<<grid, block>>>(d_vox, d_mean, N, D, d_gram);
        t.gram_ms = timer.stop_ms();
        CUDA_CHECK_LAST("gram_kernel");
    }

    // ---- (3) eigendecomposition of the N x N Gram via cuSOLVER -----------
    // THE LIBRARY CALL, NOT A BLACK BOX:
    //   cusolverDnDsyevd solves A x = lambda x for a real SYMMETRIC A by the
    //   divide-and-conquer method (O(N^3)). jobz=VECTOR overwrites A (d_gram)
    //   with the orthonormal eigenvectors as COLUMNS; uplo=LOWER says read the
    //   lower triangle (our matrix is exactly symmetric, so either works).
    //   Eigenvalues come back ASCENDING in d_W. It first reports the workspace it
    //   needs (bufferSize), which we allocate. Hand-rolling this would mean coding
    //   tridiagonalization + a QR/divide-and-conquer iteration -- exactly why we
    //   lean on the library (and document it).
    {
        cusolverDnHandle_t handle = nullptr;
        CUSOLVER_CHECK(cusolverDnCreate(&handle));

        double* d_W = nullptr;     // [N] eigenvalues (ascending)
        int*    d_info = nullptr;  // convergence flag
        CUDA_CHECK(cudaMalloc(&d_W, (size_t)N * sizeof(double)));
        CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));

        const cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;  // want vectors too
        const cublasFillMode_t  uplo = CUBLAS_FILL_MODE_LOWER;
        int lwork = 0;
        CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(handle, jobz, uplo, N, d_gram, N, d_W, &lwork));
        double* d_work = nullptr;
        CUDA_CHECK(cudaMalloc(&d_work, (size_t)lwork * sizeof(double)));

        timer.start();
        CUSOLVER_CHECK(cusolverDnDsyevd(handle, jobz, uplo, N, d_gram, N, d_W, d_work, lwork, d_info));
        t.eigen_ms = timer.stop_ms();

        int info = 0;
        CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
        if (info != 0) {
            std::fprintf(stderr, "[run_3dva_gpu] Dsyevd failed to converge (info=%d)\n", info);
            std::exit(EXIT_FAILURE);
        }

        // Pull eigenvalues to the host.
        eval.assign(N, 0.0);
        CUDA_CHECK(cudaMemcpy(eval.data(), d_W, (size_t)N * sizeof(double), cudaMemcpyDeviceToHost));

        // Variance explained by PC1 = largest eigenvalue / sum of eigenvalues.
        // (Eigenvalues are the per-mode variances; ascending order => last is largest.)
        double total = 0.0;
        for (int i = 0; i < N; ++i) total += (eval[i] > 0.0 ? eval[i] : 0.0);
        var_explained_pc1 = (total > 0.0) ? (eval[N - 1] / total) : 0.0;

        // The top eigenvector w (PC1 in Gram space) is the LAST column of d_gram
        // (column-major), i.e. starts at offset (N-1)*N. Copy it into d_w for the
        // lift kernel.
        CUDA_CHECK(cudaMemcpy(d_w, d_gram + (size_t)(N - 1) * N,
                              (size_t)N * sizeof(double), cudaMemcpyDeviceToDevice));

        cudaFree(d_work);
        cudaFree(d_W);
        cudaFree(d_info);
        cusolverDnDestroy(handle);
    }

    // ---- (4) lift Gram eigenvector -> volume-space PC1 -------------------
    {
        const int blocks = (D + TPB - 1) / TPB;
        timer.start();
        lift_kernel<<<blocks, TPB>>>(d_vox, d_mean, d_w, N, D, d_u);
        t.lift_ms = timer.stop_ms();
        CUDA_CHECK_LAST("lift_kernel");
    }

    // Bring the un-normalized PC to the host to normalize + fix the sign with the
    // SAME convention as the CPU reference (largest-magnitude voxel positive).
    pc1.assign(D, 0.0);
    CUDA_CHECK(cudaMemcpy(pc1.data(), d_u, meanBytes, cudaMemcpyDeviceToHost));
    double nrm = 0.0;
    for (int v = 0; v < D; ++v) nrm += pc1[v] * pc1[v];
    nrm = std::sqrt(nrm);
    if (nrm > 0.0) { const double inv = 1.0 / nrm; for (int v = 0; v < D; ++v) pc1[v] *= inv; }
    int amax = 0;
    for (int v = 1; v < D; ++v) if (std::fabs(pc1[v]) > std::fabs(pc1[amax])) amax = v;
    if (pc1[amax] < 0.0) for (int v = 0; v < D; ++v) pc1[v] = -pc1[v];
    // Push the normalized, sign-fixed PC back to the device for the projection.
    CUDA_CHECK(cudaMemcpy(d_u, pc1.data(), meanBytes, cudaMemcpyHostToDevice));

    // ---- (5) per-particle latent coordinate ------------------------------
    {
        const int blocks = (N + TPB - 1) / TPB;
        timer.start();
        project_kernel<<<blocks, TPB>>>(d_vox, d_mean, d_u, N, D, d_z);
        t.proj_ms = timer.stop_ms();
        CUDA_CHECK_LAST("project_kernel");
    }
    z.assign(N, 0.0);
    CUDA_CHECK(cudaMemcpy(z.data(), d_z, (size_t)N * sizeof(double), cudaMemcpyDeviceToHost));

    // Also return the mean volume (used by main.cu's verification).
    mean.assign(D, 0.0);
    CUDA_CHECK(cudaMemcpy(mean.data(), d_mean, meanBytes, cudaMemcpyDeviceToHost));

    // ---- free everything (no GPU garbage collector) ----------------------
    CUDA_CHECK(cudaFree(d_vox));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_gram));
    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_z));
    CUDA_CHECK(cudaFree(d_w));
}
