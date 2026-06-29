// ===========================================================================
// src/kernels.cu  --  GPU ERI kernel (O(N^4)) + cuSOLVER generalized eigensolve
// ---------------------------------------------------------------------------
// Project 1.7 : Quantum Chemistry / DFT  (reduced-scope RHF/SCF -- see THEORY.md)
//
// Two GPU jobs live here (see kernels.cuh for the interface and the big picture):
//   (1) eri_kernel / build_eri_gpu -- the two-electron repulsion tensor, ONE
//       THREAD PER (i,j,k,l) integral. This is the catalog's named O(N^4)
//       bottleneck and the project's headline kernel.
//   (2) cusolver_generalized -- the per-SCF-cycle generalized eigenproblem
//       F C = S C eps, handed to cuSOLVER (documented, not a black box).
//
// The kernel calls the SAME eri_primitive() inline as the CPU (gaussian_integrals.h),
// so the GPU and CPU tensors are bitwise identical -- verification is exact.
// ===========================================================================
#include "kernels.cuh"
#include "gaussian_integrals.h"        // eri_primitive (HD: runs on the device too)
#include "util/cuda_check.cuh"         // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"              // GpuTimer

#include <cusolverDn.h>                // dense cuSOLVER (the eigensolver)
#include <cublas_v2.h>                 // cublasFillMode_t enum used by cuSOLVER
#include <cstdio>
#include <cstdlib>
#include <vector>

// We cap STO-3G primitive count at 3 per contracted function; the kernel keeps a
// fixed-size scratch per basis function so it needs no dynamic device allocation.
#define MAX_PRIM 3

// cuSOLVER has its own status enum; guard + explain every call (no black box).
#define CUSOLVER_CHECK(call)                                                    \
    do {                                                                        \
        cusolverStatus_t st__ = (call);                                         \
        if (st__ != CUSOLVER_STATUS_SUCCESS) {                                  \
            std::fprintf(stderr, "[CUSOLVER_CHECK] %s:%d -> status %d\n",       \
                         __FILE__, __LINE__, static_cast<int>(st__));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// ---------------------------------------------------------------------------
// Flat, device-friendly view of the contracted basis.
//   STL vectors cannot cross to the device, so we copy the basis into plain
//   arrays the kernel can index:
//     cx,cy,cz [N]            : centers of each contracted function
//     nprim    [N]            : how many primitives each function has (<= MAX_PRIM)
//     pexp,pcoef [N*MAX_PRIM] : per-function primitive exponents and coefficients
//   Row-major: primitive p of function i lives at i*MAX_PRIM + p.
// ---------------------------------------------------------------------------
struct FlatBasis {
    std::vector<double> cx, cy, cz;
    std::vector<int>    nprim;
    std::vector<double> pexp, pcoef;
};

static FlatBasis flatten(const Basis& bs, int N) {
    FlatBasis fb;
    fb.cx.resize(N); fb.cy.resize(N); fb.cz.resize(N);
    fb.nprim.resize(N);
    fb.pexp.assign(static_cast<size_t>(N) * MAX_PRIM, 0.0);
    fb.pcoef.assign(static_cast<size_t>(N) * MAX_PRIM, 0.0);
    for (int i = 0; i < N; ++i) {
        fb.cx[i] = bs[i].x; fb.cy[i] = bs[i].y; fb.cz[i] = bs[i].z;
        const int np = static_cast<int>(bs[i].exp.size());
        fb.nprim[i] = np;
        for (int p = 0; p < np; ++p) {
            fb.pexp [static_cast<size_t>(i) * MAX_PRIM + p] = bs[i].exp[p];
            fb.pcoef[static_cast<size_t>(i) * MAX_PRIM + p] = bs[i].coef[p];
        }
    }
    return fb;
}

// ---------------------------------------------------------------------------
// eri_kernel: compute ONE contracted two-electron integral (ij|kl) per thread.
//   launch config:
//     total threads = N^4 (one per integral); block = 256; grid = ceil(N^4/256).
//   thread t -> the flat integral index, decoded back into (i,j,k,l) by repeated
//   div/mod with the slowest-varying index first (matches the CPU memory layout
//   eri[((i*N+j)*N+k)*N+l], so the two tensors line up element-for-element).
//
//   Each thread loops over the (<=81) primitive quartets of its four functions and
//   sums coef*coef*coef*coef * eri_primitive(...). eri_primitive is the shared
//   __host__ __device__ formula -- identical arithmetic to the CPU reference.
//   memory: pure global-memory reads of the small flat-basis arrays (they fit in
//   L2 and are reused across threads); no shared memory needed at this size. The
//   real win is the N^4 independence: thousands of integrals computed at once.
// ---------------------------------------------------------------------------
__global__ void eri_kernel(int N,
                           const double* __restrict__ cx,
                           const double* __restrict__ cy,
                           const double* __restrict__ cz,
                           const int*    __restrict__ nprim,
                           const double* __restrict__ pexp,
                           const double* __restrict__ pcoef,
                           double* __restrict__ eri) {
    // Global linear thread id = which integral this thread owns.
    const long long tid = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long total = static_cast<long long>(N) * N * N * N;
    if (tid >= total) return;                  // guard the ragged last block

    // Decode tid -> (i,j,k,l). Layout: index = ((i*N + j)*N + k)*N + l.
    const int l = static_cast<int>(tid % N);
    const int k = static_cast<int>((tid / N) % N);
    const int j = static_cast<int>((tid / (static_cast<long long>(N) * N)) % N);
    const int i = static_cast<int>(tid / (static_cast<long long>(N) * N * N));

    // Centers of the four functions.
    const double Ax = cx[i], Ay = cy[i], Az = cz[i];
    const double Bx = cx[j], By = cy[j], Bz = cz[j];
    const double Cx = cx[k], Cy = cy[k], Cz = cz[k];
    const double Dx = cx[l], Dy = cy[l], Dz = cz[l];
    const int npA = nprim[i], npB = nprim[j], npC = nprim[k], npD = nprim[l];

    double val = 0.0;
    // Contract over every primitive quartet, just like build_eri_cpu.
    for (int pa = 0; pa < npA; ++pa) {
        const double a  = pexp [static_cast<size_t>(i) * MAX_PRIM + pa];
        const double ca = pcoef[static_cast<size_t>(i) * MAX_PRIM + pa];
        for (int pb = 0; pb < npB; ++pb) {
            const double b  = pexp [static_cast<size_t>(j) * MAX_PRIM + pb];
            const double cb = pcoef[static_cast<size_t>(j) * MAX_PRIM + pb];
            for (int pc = 0; pc < npC; ++pc) {
                const double c  = pexp [static_cast<size_t>(k) * MAX_PRIM + pc];
                const double cc = pcoef[static_cast<size_t>(k) * MAX_PRIM + pc];
                for (int pd = 0; pd < npD; ++pd) {
                    const double d  = pexp [static_cast<size_t>(l) * MAX_PRIM + pd];
                    const double cd = pcoef[static_cast<size_t>(l) * MAX_PRIM + pd];
                    val += ca * cb * cc * cd *
                           eri_primitive(a, Ax, Ay, Az, b, Bx, By, Bz,
                                         c, Cx, Cy, Cz, d, Dx, Dy, Dz);
                }
            }
        }
    }
    eri[tid] = val;
}

// ---------------------------------------------------------------------------
// build_eri_gpu: host wrapper -- flatten the basis, upload, launch eri_kernel,
//   download the N^4 tensor, time the kernel. Mirrors build_eri_cpu() so main.cu
//   can diff the two element-by-element.
// ---------------------------------------------------------------------------
void build_eri_gpu(const Basis& bs, int N, std::vector<double>& eri, float* kernel_ms) {
    const long long total = static_cast<long long>(N) * N * N * N;
    eri.assign(static_cast<size_t>(total), 0.0);

    FlatBasis fb = flatten(bs, N);

    // Device buffers for the flat basis + the output tensor.
    double *d_cx=nullptr,*d_cy=nullptr,*d_cz=nullptr,*d_pexp=nullptr,*d_pcoef=nullptr,*d_eri=nullptr;
    int    *d_nprim=nullptr;
    CUDA_CHECK(cudaMalloc(&d_cx,    N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cy,    N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_cz,    N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nprim, N * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_pexp,  static_cast<size_t>(N) * MAX_PRIM * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_pcoef, static_cast<size_t>(N) * MAX_PRIM * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_eri,   static_cast<size_t>(total) * sizeof(double)));

    CUDA_CHECK(cudaMemcpy(d_cx,    fb.cx.data(),    N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cy,    fb.cy.data(),    N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cz,    fb.cz.data(),    N * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_nprim, fb.nprim.data(), N * sizeof(int),    cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pexp,  fb.pexp.data(),
                          static_cast<size_t>(N) * MAX_PRIM * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pcoef, fb.pcoef.data(),
                          static_cast<size_t>(N) * MAX_PRIM * sizeof(double), cudaMemcpyHostToDevice));

    // Launch: 256 threads/block is a solid occupancy default on sm_75..sm_89.
    const int block = 256;
    const long long grid = (total + block - 1) / block;
    GpuTimer timer;
    timer.start();
    eri_kernel<<<static_cast<unsigned int>(grid), block>>>(
        N, d_cx, d_cy, d_cz, d_nprim, d_pexp, d_pcoef, d_eri);
    CUDA_CHECK_LAST("eri_kernel");      // catch launch + execution errors
    *kernel_ms = timer.stop_ms();

    CUDA_CHECK(cudaMemcpy(eri.data(), d_eri,
                          static_cast<size_t>(total) * sizeof(double), cudaMemcpyDeviceToHost));

    cudaFree(d_cx); cudaFree(d_cy); cudaFree(d_cz);
    cudaFree(d_nprim); cudaFree(d_pexp); cudaFree(d_pcoef); cudaFree(d_eri);
}

// ---------------------------------------------------------------------------
// cusolver_generalized: solve F C = S C eps with cuSOLVER -- NOT a black box.
//   cusolverDnDsygvd solves the GENERALIZED symmetric-definite eigenproblem
//   A x = lambda B x by divide-and-conquer (O(N^3)). We pass:
//     itype = CUSOLVER_EIG_TYPE_1   -> the form A x = lambda B x (our F C = eps S C)
//     jobz  = CUSOLVER_EIG_MODE_VECTOR -> also return eigenvectors (the MO coeffs)
//     uplo  = LOWER                 -> read the lower triangle (matrices symmetric)
//   On output d_A holds the eigenvectors as COLUMNS and d_W the eigenvalues
//   (ascending) -- exactly the C (column-major) and eps our SCF loop expects.
//   By hand this would mean: Cholesky-factor S, reduce to a standard eigenproblem,
//   tridiagonalize, run divide-and-conquer, back-transform -- hundreds of careful
//   lines. The library does it correctly and fast; we only document what it means.
// ---------------------------------------------------------------------------
void cusolver_generalized(const std::vector<double>& F, const std::vector<double>& S,
                          int N, std::vector<double>& C, std::vector<double>& eps) {
    C.assign(static_cast<size_t>(N) * N, 0.0);
    eps.assign(N, 0.0);

    cusolverDnHandle_t handle = nullptr;
    CUSOLVER_CHECK(cusolverDnCreate(&handle));

    double *d_A=nullptr, *d_B=nullptr, *d_W=nullptr;     // A=F, B=S, W=eigenvalues
    int    *d_info=nullptr;
    CUDA_CHECK(cudaMalloc(&d_A, static_cast<size_t>(N) * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_B, static_cast<size_t>(N) * N * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_W, static_cast<size_t>(N) * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_info, sizeof(int)));
    // F and S are symmetric, so their row-major and column-major layouts are the
    // same bytes -- we upload them straight into cuSOLVER's column-major A, B.
    CUDA_CHECK(cudaMemcpy(d_A, F.data(), static_cast<size_t>(N) * N * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, S.data(), static_cast<size_t>(N) * N * sizeof(double),
                          cudaMemcpyHostToDevice));

    const cusolverEigType_t  itype = CUSOLVER_EIG_TYPE_1;       // A x = lambda B x
    const cusolverEigMode_t  jobz  = CUSOLVER_EIG_MODE_VECTOR;  // want eigenvectors
    const cublasFillMode_t   uplo  = CUBLAS_FILL_MODE_LOWER;

    int lwork = 0;
    CUSOLVER_CHECK(cusolverDnDsygvd_bufferSize(handle, itype, jobz, uplo, N,
                                               d_A, N, d_B, N, d_W, &lwork));
    double* d_work = nullptr;
    CUDA_CHECK(cudaMalloc(&d_work, static_cast<size_t>(lwork) * sizeof(double)));

    CUSOLVER_CHECK(cusolverDnDsygvd(handle, itype, jobz, uplo, N,
                                    d_A, N, d_B, N, d_W, d_work, lwork, d_info));

    int info = 0;
    CUDA_CHECK(cudaMemcpy(&info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (info != 0) {
        std::fprintf(stderr, "[cusolver_generalized] Dsygvd failed (info=%d)\n", info);
        std::exit(EXIT_FAILURE);
    }

    // d_A now holds eigenvectors as columns (column-major) = our C; d_W = eps.
    CUDA_CHECK(cudaMemcpy(C.data(),   d_A, static_cast<size_t>(N) * N * sizeof(double),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(eps.data(), d_W, static_cast<size_t>(N) * sizeof(double),
                          cudaMemcpyDeviceToHost));

    cudaFree(d_work); cudaFree(d_A); cudaFree(d_B); cudaFree(d_W); cudaFree(d_info);
    cusolverDnDestroy(handle);
}
