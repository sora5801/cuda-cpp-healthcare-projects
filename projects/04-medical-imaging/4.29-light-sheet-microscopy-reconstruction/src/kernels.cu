// ===========================================================================
// src/kernels.cu  --  cuFFT Fourier-domain Richardson-Lucy deconvolution (GPU)
// ---------------------------------------------------------------------------
// Project 4.29 : Light-Sheet Microscopy Reconstruction
//
// WHAT THIS FILE DOES
//   The GPU twin of deconvolve_cpu(). It runs the identical Richardson-Lucy (RL)
//   loop, but performs the two convolutions per iteration in the FREQUENCY domain
//   with cuFFT (double precision D2Z/Z2D). main.cu runs both and compares them.
//
//   The convolution theorem is the whole trick:  (a conv b) = IFFT( FFT(a).FFT(b) ).
//   We precompute FFT(psf) once; each RL iteration then costs 2 FFTs, 2 IFFTs,
//   and a handful of element-wise kernel launches -- all on the GPU.
//
// FILE MAP
//   * CUFFT_CHECK              -- cuFFT's own error-check macro (mirrors CUDA_CHECK)
//   * complex_mul_scaled/ratio/update  -- tiny per-pixel device kernels
//   * fft_r2c_pow2 helpers via cuFFT plans (D2Z forward, Z2D inverse)
//   * deconvolve_gpu           -- the host wrapper: plans, buffers, the RL loop
//
// READ THIS AFTER: kernels.cuh (the pattern), rl_core.h (the per-pixel math).
// ===========================================================================
#include "kernels.cuh"
#include "rl_core.h"             // rl_ratio, rl_apply  (shared with the CPU)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cufft.h>               // cufftHandle, cufftExecD2Z / Z2D, cufftDoubleComplex
#include <cstdio>
#include <cstdlib>

// Threads per block. 256 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, 8 warps to hide latency, many blocks resident for occupancy.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// CUFFT_CHECK(call): cuFFT returns its own status type (cufftResult), not
//   cudaError_t, so it needs a parallel macro to CUDA_CHECK. Every cuFFT call is
//   guarded; on failure we print file/line/code and abort (a failed transform
//   makes the whole result meaningless in teaching code).
// ---------------------------------------------------------------------------
#define CUFFT_CHECK(call)                                                        \
    do {                                                                         \
        cufftResult st__ = (call);                                              \
        if (st__ != CUFFT_SUCCESS) {                                            \
            std::fprintf(stderr, "[CUFFT_CHECK] %s:%d -> cuFFT error %d\n",     \
                         __FILE__, __LINE__, static_cast<int>(st__));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                        \
    } while (0)

// ===========================================================================
// Device kernels -- each is "one thread per element", the fundamental mapping.
// ===========================================================================

// complex_mul_scaled: out[i] = a[i] * (conj_b ? conj(b[i]) : b[i]) * scale.
//   The frequency-domain heart of convolution. cufftDoubleComplex is laid out as
//   { double x (real), double y (imag) }, identical to CUDA's double2, so we take
//   double2* here. `scale` folds in cuFFT's normalization: an unnormalized
//   forward+inverse pair multiplies the data by N = H*W, so we divide by N once.
//   `conj_b` picks CORRELATION over convolution (the flipped-PSF adjoint step).
//   grid = ceil(n/256), block = 256; thread i owns frequency bin i (of n = H*(W/2+1)).
__global__ void complex_mul_scaled(const double2* __restrict__ a,
                                   const double2* __restrict__ b,
                                   int n, double scale, bool conj_b,
                                   double2* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's bin
    if (i >= n) return;                              // guard the ragged last block
    double ar = a[i].x, ai = a[i].y;                 // a = ar + i*ai
    double br = b[i].x, bi = b[i].y;                 // b = br + i*bi
    if (conj_b) bi = -bi;                            // conj(b) = br - i*bi
    // Complex product (ar+i ai)(br+i bi) = (ar br - ai bi) + i(ar bi + ai br).
    out[i].x = (ar * br - ai * bi) * scale;
    out[i].y = (ar * bi + ai * br) * scale;
}

// ratio_kernel: out[i] = rl_ratio(measured[i], reblurred[i]).  Per-pixel RL
//   correction ratio, using the SHARED rl_core.h function so it matches the CPU
//   bit-for-bit at the arithmetic level. One thread per spatial pixel.
__global__ void ratio_kernel(const double* __restrict__ measured,
                             const double* __restrict__ reblurred,
                             int n, double* __restrict__ out) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = rl_ratio(measured[i], reblurred[i]);
}

// update_kernel: est[i] = rl_apply(est[i], correction[i]).  The multiplicative
//   RL update, in place, again via shared rl_core.h. One thread per pixel.
__global__ void update_kernel(double* __restrict__ est,
                             const double* __restrict__ correction, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) est[i] = rl_apply(est[i], correction[i]);
}

// ---------------------------------------------------------------------------
// A tiny host helper to launch a 1D grid over `n` elements at 256 threads/block.
//   Returns the block count (ceiling division "round up").
// ---------------------------------------------------------------------------
static inline int grid_for(int n) {
    return (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
}

// ===========================================================================
// deconvolve_gpu -- the host wrapper: plan cuFFT, allocate device buffers, run
// the RL loop, copy the result back. Mirrors deconvolve_cpu() step for step.
// ===========================================================================
void deconvolve_gpu(const LsfmData& d, std::vector<double>& estimate,
                    float* kernel_ms) {
    const int H = d.H, W = d.W;
    const int n = H * W;                          // real-space pixel count
    // Real-to-complex FFT stores only the non-redundant half along the last axis
    // (Hermitian symmetry of a real signal's FFT), so the complex spectrum has
    // W/2+1 columns. nc = number of complex frequency bins.
    const int wc = W / 2 + 1;
    const int nc = H * wc;
    const double inv_n = 1.0 / static_cast<double>(n);   // 1/N normalization
    estimate.assign(static_cast<std::size_t>(n), 0.0);

    // ---- Build the PSF on the host (shared gaussian_psf) and its mean-init ----
    // We reuse the CPU's PSF builder so both paths convolve with the identical
    // kernel. The estimate is initialized flat at the mean of the measurement,
    // exactly like deconvolve_cpu().
    const std::vector<double> h_psf = gaussian_psf(H, W, d.sigma);
    double mean = 0.0;
    for (double v : d.measured) mean += v;        // fixed-order sum (deterministic)
    mean /= static_cast<double>(n);

    // ---- Device buffers ------------------------------------------------------
    // Real-space (double): the measurement b, current estimate x, PSF, and two
    // scratch images (reblur / correction share `d_tmp_real`; ratio uses d_ratio).
    double *d_meas = nullptr, *d_est = nullptr, *d_psf = nullptr;
    double *d_tmp_real = nullptr, *d_ratio = nullptr;
    // Frequency-space (complex, double): FFT(psf), FFT(current), and the product.
    cufftDoubleComplex *d_psf_f = nullptr, *d_tmp_f = nullptr, *d_prod_f = nullptr;

    const std::size_t rbytes = static_cast<std::size_t>(n) * sizeof(double);
    const std::size_t cbytes = static_cast<std::size_t>(nc) * sizeof(cufftDoubleComplex);
    CUDA_CHECK(cudaMalloc(&d_meas,     rbytes));
    CUDA_CHECK(cudaMalloc(&d_est,      rbytes));
    CUDA_CHECK(cudaMalloc(&d_psf,      rbytes));
    CUDA_CHECK(cudaMalloc(&d_tmp_real, rbytes));
    CUDA_CHECK(cudaMalloc(&d_ratio,    rbytes));
    CUDA_CHECK(cudaMalloc(&d_psf_f,    cbytes));
    CUDA_CHECK(cudaMalloc(&d_tmp_f,    cbytes));
    CUDA_CHECK(cudaMalloc(&d_prod_f,   cbytes));

    // Copy the measurement and PSF up; fill the estimate with the flat mean.
    CUDA_CHECK(cudaMemcpy(d_meas, d.measured.data(), rbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_psf,  h_psf.data(),      rbytes, cudaMemcpyHostToDevice));
    {
        // Initialize the estimate to `mean` everywhere. A one-line kernel would do,
        // but a host-filled vector copied once is simplest and equally deterministic.
        std::vector<double> flat(static_cast<std::size_t>(n), mean);
        CUDA_CHECK(cudaMemcpy(d_est, flat.data(), rbytes, cudaMemcpyHostToDevice));
    }

    // ---- cuFFT plans ---------------------------------------------------------
    // cufftPlan2d(&plan, H, W, TYPE) builds a plan for a single 2D transform of an
    // H-by-W array. D2Z = real (double) -> complex (double), forward. Z2D =
    // complex -> real, inverse. These are the double-precision analogues of R2C /
    // C2R. cuFFT does NOT normalize, so a D2Z followed by Z2D scales data by N --
    // we divide by N once inside complex_mul_scaled (the `scale` argument).
    //
    // Hand-rolling this would mean writing a radix-mixed 2D FFT with bit-reversal
    // and twiddle factors -- hundreds of lines and far slower than the vendor's
    // tuned kernels. That is precisely why we call the library (CLAUDE.md 6.1.6).
    cufftHandle plan_fwd, plan_inv;
    CUFFT_CHECK(cufftPlan2d(&plan_fwd, H, W, CUFFT_D2Z));   // real image  -> spectrum
    CUFFT_CHECK(cufftPlan2d(&plan_inv, H, W, CUFFT_Z2D));   // spectrum    -> real image

    const int cgrid = grid_for(nc);   // launch size over complex bins
    const int rgrid = grid_for(n);    // launch size over real pixels

    GpuTimer timer;
    timer.start();

    // Precompute FFT(psf) ONCE -- the PSF is constant across all iterations.
    CUFFT_CHECK(cufftExecD2Z(plan_fwd, d_psf, d_psf_f));

    // ---- The Richardson-Lucy loop (mirrors deconvolve_cpu) -------------------
    for (int it = 0; it < d.iters; ++it) {
        // 1) reblur = IFFT( FFT(est) . FFT(psf) )   -- forward model (convolution)
        CUFFT_CHECK(cufftExecD2Z(plan_fwd, d_est, d_tmp_f));           // FFT(est)
        // product = FFT(est) . FFT(psf) / N   (convolution, NOT conjugated)
        complex_mul_scaled<<<cgrid, THREADS_PER_BLOCK>>>(
            d_tmp_f, d_psf_f, nc, inv_n, /*conj_b=*/false, d_prod_f);
        CUDA_CHECK_LAST("complex_mul_scaled(reblur)");
        CUFFT_CHECK(cufftExecZ2D(plan_inv, d_prod_f, d_tmp_real));      // -> reblur

        // 2) ratio = measured / reblur      (shared rl_core.h, per pixel)
        ratio_kernel<<<rgrid, THREADS_PER_BLOCK>>>(d_meas, d_tmp_real, n, d_ratio);
        CUDA_CHECK_LAST("ratio_kernel");

        // 3) correct = IFFT( FFT(ratio) . conj(FFT(psf)) )  -- adjoint (correlation)
        CUFFT_CHECK(cufftExecD2Z(plan_fwd, d_ratio, d_tmp_f));         // FFT(ratio)
        complex_mul_scaled<<<cgrid, THREADS_PER_BLOCK>>>(
            d_tmp_f, d_psf_f, nc, inv_n, /*conj_b=*/true, d_prod_f);    // conj -> correlation
        CUDA_CHECK_LAST("complex_mul_scaled(correct)");
        CUFFT_CHECK(cufftExecZ2D(plan_inv, d_prod_f, d_tmp_real));      // -> correction

        // 4) est = est * correction         (shared rl_core.h, per pixel, in place)
        update_kernel<<<rgrid, THREADS_PER_BLOCK>>>(d_est, d_tmp_real, n);
        CUDA_CHECK_LAST("update_kernel");
    }

    *kernel_ms = timer.stop_ms();   // total GPU time for all iterations

    // ---- Bring the deblurred estimate back to the host -----------------------
    CUDA_CHECK(cudaMemcpy(estimate.data(), d_est, rbytes, cudaMemcpyDeviceToHost));

    // ---- Tear down (no GPU garbage collector; free everything) ---------------
    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);
    CUDA_CHECK(cudaFree(d_meas));
    CUDA_CHECK(cudaFree(d_est));
    CUDA_CHECK(cudaFree(d_psf));
    CUDA_CHECK(cudaFree(d_tmp_real));
    CUDA_CHECK(cudaFree(d_ratio));
    CUDA_CHECK(cudaFree(d_psf_f));
    CUDA_CHECK(cudaFree(d_tmp_f));
    CUDA_CHECK(cudaFree(d_prod_f));
}
