// ===========================================================================
// src/kernels.cu  --  GPU QSM dipole inversion via cuFFT (3-D spectral solve)
// ---------------------------------------------------------------------------
// Project 4.22 : Quantitative Susceptibility Mapping (QSM)
//
// GPU twin of the reconstruct_*_cpu() references. The CPU reference does its 3-D
// Fourier transforms with a naive O(N^2) direct DFT; here we do the IDENTICAL
// transform in O(N log N) with cuFFT (double-precision complex-to-complex, Z2Z).
// The per-bin dipole/inversion math is the SAME qsm_core.h code the CPU calls, so
// the only numerical difference is "direct DFT vs cuFFT" -- see THEORY.md
// "How we verify correctness".
//
// TWO RECONSTRUCTIONS, ONE PIPELINE
//   * TKD (direct, one-shot):
//         FFT3(field) -> multiply each bin by 1/D_thr(k) -> IFFT3 -> chi
//   * Tikhonov (iterative):
//         FFT3(field) = Ffield (once) ; Fchi = 0
//         repeat `iters` times: Fchi <- gradient_step(Fchi, Ffield, D, alpha, step)
//         IFFT3(Fchi) -> chi
//   The iterative path is the catalog's headline pattern: O(100) iterations of
//   3-D FFT + gradient updates. Here the FFT is done ONCE (the objective is
//   diagonal in k-space, so the whole iteration stays in the frequency domain);
//   THEORY.md explains why a real edge-regularized MEDI solver instead needs an
//   FFT *inside* every iteration, and how that changes the GPU cost.
//
// cuFFT is UNNORMALIZED: a forward+inverse round trip multiplies by N = nx*ny*nz,
// so the final inverse result is divided by N (folded into the last kernel).
//
// Read kernels.cuh first for the big idea; util/cuda_check.cuh for the macros.
// ===========================================================================
#include "kernels.cuh"
#include "qsm_core.h"            // dipole_kernel(), tkd_reciprocal(), grad step
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cufft.h>               // cufftHandle, cufftExecZ2Z, cufftDoubleComplex
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
// cuFFT has its own status enum (cufftResult), so it needs its own check macro
// that mirrors CUDA_CHECK. Every cuFFT call is guarded and the failure is printed
// with file/line so a bad plan or exec is never silent (CLAUDE.md 6.1 rule 7).
// ---------------------------------------------------------------------------
#define CUFFT_CHECK(call)                                                       \
    do {                                                                        \
        cufftResult st__ = (call);                                              \
        if (st__ != CUFFT_SUCCESS) {                                            \
            std::fprintf(stderr, "[CUFFT_CHECK] %s:%d -> cuFFT error %d\n",     \
                         __FILE__, __LINE__, static_cast<int>(st__));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// Threads per block for the element-wise k-space kernels. 256 is a solid default
// on sm_75..sm_89 (multiple of the 32-lane warp, 8 warps to hide latency).
static constexpr int THREADS_PER_BLOCK = 256;

// cufftDoubleComplex is layout-identical to our Complex (two doubles: .x=re,
// .y=im). We work entirely in DOUBLE precision (Z2Z): QSM inversion divides by
// tiny dipole values near the magic cone, so single precision would lose bits
// exactly where it hurts, and the teaching volume is small enough that double
// is free.

// ---------------------------------------------------------------------------
// device_signed_freq: the DEVICE copy of reference_cpu.h's signed_freq().
//   Maps FFT bin index i in [0,n) to its signed frequency: bins above n/2 are
//   negative frequencies (i - n). We duplicate it here (rather than include the
//   host header's inline) so the device sees a __device__ version; it is the
//   identical formula, so CPU and GPU place every bin at the same frequency.
// ---------------------------------------------------------------------------
__device__ __forceinline__ int device_signed_freq(int i, int n) {
    return (i <= n / 2) ? i : (i - n);
}

// ---------------------------------------------------------------------------
// bin_dipole: compute the dipole-kernel value D(k) for a linear bin index.
//   Recovers (kx,ky,kz) from the flat index (x fastest, then y, then z), maps
//   each to its signed, dimension-scaled frequency, and calls the SHARED
//   dipole_kernel(). Used by both k-space kernels below so the frequency->D map
//   is defined in exactly one place.
//     i          : flat bin index in [0, nx*ny*nz)
//     nx,ny,nz   : grid dimensions
// ---------------------------------------------------------------------------
__device__ __forceinline__ double bin_dipole(int i, int nx, int ny, int nz) {
    const int kx = i % nx;               // x fastest
    const int ky = (i / nx) % ny;        // then y
    const int kz = i / (nx * ny);        // then z
    // Scale each signed frequency by its axis length so |k| is dimensionless and
    // the kernel matches the CPU reference's apply_kspace_weight() exactly.
    const double fx = static_cast<double>(device_signed_freq(kx, nx)) / nx;
    const double fy = static_cast<double>(device_signed_freq(ky, ny)) / ny;
    const double fz = static_cast<double>(device_signed_freq(kz, nz)) / nz;
    return dipole_kernel(fx, fy, fz);    // shared host+device math (qsm_core.h)
}

// ===========================================================================
// Custom element-wise kernels -- each is "one GPU thread per k-space bin".
//   grid  = ceil(N / THREADS_PER_BLOCK), block = THREADS_PER_BLOCK.
//   thread i = blockIdx.x*blockDim.x + threadIdx.x owns bin i; the `if (i<N)`
//   guard handles the ragged final block. No shared memory / atomics needed: each
//   bin is fully independent (the dipole operator is diagonal in k-space).
// ===========================================================================

// --- weight_tkd_kernel -----------------------------------------------------
// Apply the TKD inverse weight 1/D_thr(k) to each bin AND fold in the 1/N inverse-
// FFT normalization, in one pass. spec[i] <- spec[i] * (w * inv_n), where
// w = tkd_reciprocal(D(k), thr) from the shared header.
//   spec  : in/out complex spectrum (length N)
//   nx..nz: dimensions (for bin_dipole)
//   thr   : TKD threshold
//   inv_n : 1.0 / N, pre-multiplied so the later IFFT output is already scaled
__global__ void weight_tkd_kernel(cufftDoubleComplex* __restrict__ spec,
                                  int nx, int ny, int nz, int N,
                                  double thr, double inv_n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const double D = bin_dipole(i, nx, ny, nz);
    const double w = tkd_reciprocal(D, thr) * inv_n;   // bounded inverse * 1/N
    spec[i].x *= w;                                     // real part
    spec[i].y *= w;                                     // imag part
}

// --- grad_step_kernel ------------------------------------------------------
// One Tikhonov gradient-descent step for each bin, using the SHARED
// tikhonov_grad_step() so the GPU iteration is byte-for-byte the CPU iteration.
//   Fchi   : in/out current chi-spectrum estimate (length N)
//   Ffield : the fixed data spectrum (length N)
//   nx..nz : dimensions
//   alpha  : Tikhonov weight
//   step   : gradient-descent step size
// The bin's D(k) is recomputed each call (cheap: a few flops) rather than stored,
// keeping the kernel memory-light -- it reads/writes only its own two complex bins.
__global__ void grad_step_kernel(cufftDoubleComplex* __restrict__ Fchi,
                                 const cufftDoubleComplex* __restrict__ Ffield,
                                 int nx, int ny, int nz, int N,
                                 double alpha, double step) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    const double D = bin_dipole(i, nx, ny, nz);
    // Load this bin's current estimate and the data into the shared Complex type.
    const Complex cur = cplx(Fchi[i].x,   Fchi[i].y);
    const Complex dat = cplx(Ffield[i].x, Ffield[i].y);
    const Complex nxt = tikhonov_grad_step(cur, dat, D, alpha, step);  // shared
    Fchi[i].x = nxt.re;
    Fchi[i].y = nxt.im;
}

// --- scale_kernel ----------------------------------------------------------
// Multiply every bin by a real scalar (used to fold the 1/N inverse-FFT
// normalization into the iterative path, where the weighting happened in the
// gradient loop rather than in a single weight kernel).
//   spec : in/out complex spectrum (length N);  s : the scalar (e.g. 1/N)
__global__ void scale_kernel(cufftDoubleComplex* __restrict__ spec, int N, double s) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    spec[i].x *= s;
    spec[i].y *= s;
}

// ---------------------------------------------------------------------------
// upload_real_as_complex: copy a real host Volume into a device complex buffer,
// with the imaginary part zeroed (the field map is real). We stage it on the host
// into a std::vector<cufftDoubleComplex> once, then a single cudaMemcpy. Small and
// one-time, so simplicity beats a custom "real->complex" kernel here.
// ---------------------------------------------------------------------------
static void upload_real_as_complex(const Volume& v, cufftDoubleComplex* d_dst) {
    const int N = v.size();
    std::vector<cufftDoubleComplex> host(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i) {
        host[static_cast<std::size_t>(i)].x = v.vox[static_cast<std::size_t>(i)];
        host[static_cast<std::size_t>(i)].y = 0.0;   // real input -> zero imag
    }
    CUDA_CHECK(cudaMemcpy(d_dst, host.data(),
                          static_cast<std::size_t>(N) * sizeof(cufftDoubleComplex),
                          cudaMemcpyHostToDevice));
}

// ---------------------------------------------------------------------------
// download_real_part: copy a device complex buffer back to the host and keep only
// the REAL part into a real Volume (the reconstructed chi is real; the imaginary
// part is ~0 round-off). `dims` supplies the output shape.
// ---------------------------------------------------------------------------
static void download_real_part(const cufftDoubleComplex* d_src, const Volume& dims,
                               Volume& out) {
    const int N = dims.size();
    std::vector<cufftDoubleComplex> host(static_cast<std::size_t>(N));
    CUDA_CHECK(cudaMemcpy(host.data(), d_src,
                          static_cast<std::size_t>(N) * sizeof(cufftDoubleComplex),
                          cudaMemcpyDeviceToHost));
    out.nx = dims.nx; out.ny = dims.ny; out.nz = dims.nz;
    out.vox.resize(static_cast<std::size_t>(N));
    for (int i = 0; i < N; ++i)
        out.vox[static_cast<std::size_t>(i)] = host[static_cast<std::size_t>(i)].x;
}

// ---------------------------------------------------------------------------
// make_plan3d: build a single in-place 3-D double-complex FFT plan for this grid.
//   cuFFT takes dimensions in "slowest to fastest" order (nz, ny, nx) to match
//   our storage (x fastest). CUFFT_Z2Z is the double-precision complex->complex
//   transform; the SAME plan does forward and inverse (the direction is a flag on
//   cufftExecZ2Z). cufftExecZ2Z computes, for input f:
//       F[k] = sum_r f[r] * exp(-2*pi*i * (k.r)/dims)   (CUFFT_FORWARD)
//   i.e. the standard DFT -- identical to the CPU reference's direct sum, just via
//   the fast FFT algorithm. cuFFT does NOT normalize (see the 1/N folding above).
// ---------------------------------------------------------------------------
static cufftHandle make_plan3d(const Volume& dims) {
    cufftHandle plan;
    CUFFT_CHECK(cufftPlan3d(&plan, dims.nz, dims.ny, dims.nx, CUFFT_Z2Z));
    return plan;
}

// ===========================================================================
// reconstruct_tkd_gpu: FFT3(field) -> * 1/D_thr(k) (+1/N) -> IFFT3 -> chi.
// (Contract documented in kernels.cuh.)
// ===========================================================================
void reconstruct_tkd_gpu(const Volume& field, double thr,
                         Volume& out, float* kernel_ms) {
    const int N = field.size();
    const double inv_n = 1.0 / static_cast<double>(N);
    const std::size_t cbytes = static_cast<std::size_t>(N) * sizeof(cufftDoubleComplex);

    // ---- Device buffer + plan --------------------------------------------
    cufftDoubleComplex* d_spec = nullptr;   // in-place working spectrum
    CUDA_CHECK(cudaMalloc(&d_spec, cbytes));
    upload_real_as_complex(field, d_spec);  // field map -> complex (imag=0)
    cufftHandle plan = make_plan3d(field);

    const int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // ---- Timed GPU work: FFT -> weight -> IFFT ---------------------------
    GpuTimer timer;
    timer.start();
    CUFFT_CHECK(cufftExecZ2Z(plan, d_spec, d_spec, CUFFT_FORWARD));   // FFT3(field)
    weight_tkd_kernel<<<blocks, THREADS_PER_BLOCK>>>(                 // * 1/D_thr * 1/N
        d_spec, field.nx, field.ny, field.nz, N, thr, inv_n);
    CUFFT_CHECK(cufftExecZ2Z(plan, d_spec, d_spec, CUFFT_INVERSE));   // IFFT3 -> chi
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("weight_tkd_kernel");

    // ---- Result + teardown -----------------------------------------------
    download_real_part(d_spec, field, out);
    cufftDestroy(plan);
    CUDA_CHECK(cudaFree(d_spec));
}

// ===========================================================================
// reconstruct_tikhonov_iter_gpu: iterative gradient descent, one thread per bin.
//   FFT3(field) -> Ffield (once). Fchi = 0. Loop `iters`: grad_step over all bins.
//   Then scale Fchi by 1/N and IFFT3 -> chi. (Contract in kernels.cuh.)
// ===========================================================================
void reconstruct_tikhonov_iter_gpu(const Volume& field, double alpha,
                                   double step, int iters,
                                   Volume& out, float* kernel_ms) {
    const int N = field.size();
    const double inv_n = 1.0 / static_cast<double>(N);
    const std::size_t cbytes = static_cast<std::size_t>(N) * sizeof(cufftDoubleComplex);

    // ---- Device buffers + plan -------------------------------------------
    cufftDoubleComplex* d_field = nullptr;  // holds field, then Ffield (in-place FFT)
    cufftDoubleComplex* d_chi   = nullptr;  // the evolving chi-spectrum estimate
    CUDA_CHECK(cudaMalloc(&d_field, cbytes));
    CUDA_CHECK(cudaMalloc(&d_chi,   cbytes));
    upload_real_as_complex(field, d_field);
    CUDA_CHECK(cudaMemset(d_chi, 0, cbytes));   // Fchi starts at 0 (all bins)
    cufftHandle plan = make_plan3d(field);

    const int blocks = (N + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

    // ---- Timed GPU work --------------------------------------------------
    GpuTimer timer;
    timer.start();
    // (1) Transform the field ONCE into its k-space spectrum Ffield (in place).
    //     The objective is diagonal in k-space, so the whole gradient descent can
    //     run in the frequency domain without another FFT per iteration.
    CUFFT_CHECK(cufftExecZ2Z(plan, d_field, d_field, CUFFT_FORWARD));
    // (2) `iters` gradient steps over every bin (the parallel loop).
    for (int it = 0; it < iters; ++it) {
        grad_step_kernel<<<blocks, THREADS_PER_BLOCK>>>(
            d_chi, d_field, field.nx, field.ny, field.nz, N, alpha, step);
    }
    // (3) Fold the 1/N inverse-FFT normalization, then inverse-transform to chi.
    scale_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_chi, N, inv_n);
    CUFFT_CHECK(cufftExecZ2Z(plan, d_chi, d_chi, CUFFT_INVERSE));
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("grad_step_kernel loop");

    // ---- Result + teardown -----------------------------------------------
    download_real_part(d_chi, field, out);
    cufftDestroy(plan);
    CUDA_CHECK(cudaFree(d_field));
    CUDA_CHECK(cudaFree(d_chi));
}
