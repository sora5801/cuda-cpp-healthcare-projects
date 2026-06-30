// ===========================================================================
// src/kernels.cu  --  GPU Richardson-Lucy via cuFFT (FFT convolution)
// ---------------------------------------------------------------------------
// Project 4.30 : Deconvolution Microscopy
//
// GPU twin of richardson_lucy_cpu(). The CPU reference does each RL convolution
// directly in space; here we do the IDENTICAL circular convolution in FREQUENCY
// space with cuFFT (the convolution theorem). The per-pixel ratio/update use the
// SAME rl_core.h functions the CPU uses, so the only numerical difference is
// "direct convolution vs FFT convolution" -- see THEORY.md "verify correctness".
//
// PIPELINE per RL iteration (all on the device):
//   1. FFT(estimate)                          -> ESTf            (D2Z, real->complex)
//   2. ESTf .*= PSFf                           (apply blur in frequency)
//   3. IFFT(ESTf) / N                          -> blurred        (Z2D, complex->real)
//   4. ratio = observed / blurred              (rl_ratio, per pixel)
//   5. FFT(ratio)                              -> RATf
//   6. RATf .*= conj(PSFf)                     (adjoint = flipped PSF in frequency)
//   7. IFFT(RATf) / N                          -> correction
//   8. estimate *= correction                  (rl_update, per pixel)
//
// cuFFT is UNNORMALIZED: a forward+inverse round trip multiplies by N = w*h, so
// every IFFT result is divided by N (folded into the complex-multiply kernels).
//
// Read kernels.cuh first for the big idea; util/cuda_check.cuh for the macros.
// ===========================================================================
#include "kernels.cuh"
#include "rl_core.h"               // rl_ratio(), rl_update()  (shared with CPU)
#include "util/cuda_check.cuh"     // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"          // GpuTimer (CUDA-event timing)

#include <cufft.h>                 // cufftHandle, cufftExecD2Z/Z2D, cufftDoubleComplex
#include <cstdio>
#include <cstdlib>
#include <vector>

// ---------------------------------------------------------------------------
// cuFFT has its own status enum (cufftResult), so it needs its own check macro
// that mirrors CUDA_CHECK. Every cuFFT call is guarded and the failure is
// printed with file/line so a bad plan or exec is never silent.
// ---------------------------------------------------------------------------
#define CUFFT_CHECK(call)                                                       \
    do {                                                                        \
        cufftResult st__ = (call);                                              \
        if (st__ != CUFFT_SUCCESS) {                                            \
            std::fprintf(stderr, "[CUFFT_CHECK] %s:%d -> cuFFT error %d\n",      \
                         __FILE__, __LINE__, static_cast<int>(st__));           \
            std::exit(EXIT_FAILURE);                                            \
        }                                                                       \
    } while (0)

// We work in DOUBLE precision throughout (cufftDoubleReal == double,
// cufftDoubleComplex == double2 with .x=real, .y=imag). RL runs many iterations
// and the images are small, so double precision is cheap and keeps the GPU
// result close to the double-precision CPU reference.

// ===========================================================================
// Custom element-wise kernels -- each is "one GPU thread per element".
// grid = ceil(count / block), block = 256 (a good occupancy default sm_75..89).
// thread i = blockIdx.x*blockDim.x + threadIdx.x owns element i; the `if (i<...)`
// guard handles the ragged final block.
// ===========================================================================

// --- complex_mul_psf -------------------------------------------------------
// Apply the PSF (or its adjoint) in the frequency domain AND fold in the 1/N
// inverse-FFT normalization, all in one pass over the half-spectrum.
//   spec    : in/out half-complex spectrum of the image, length `nf = h*(w/2+1)`
//   psf     : the PSF's half-complex spectrum, same length (precomputed once)
//   conj_psf: if nonzero, multiply by conj(psf) instead of psf. Multiplying by
//             the conjugate spectrum is EXACTLY convolving with the flipped
//             (adjoint) PSF in space -- the RL back-projection step -- so we get
//             H and H^T from a SINGLE stored spectrum (no second FFT of a
//             flipped kernel).
//   inv_n   : 1.0 / (w*h); pre-multiplied here so the later IFFT output is
//             already correctly scaled (cuFFT does not normalize).
// Complex multiply (a+bi)(c+di) = (ac-bd) + (ad+bc)i; with conj we negate d.
__global__ void complex_mul_psf(cufftDoubleComplex* __restrict__ spec,
                                const cufftDoubleComplex* __restrict__ psf,
                                int nf, int conj_psf, double inv_n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nf) return;
    const double a = spec[i].x, b = spec[i].y;     // image spectrum bin
    const double c = psf[i].x;                      // PSF spectrum bin (real part)
    const double d = conj_psf ? -psf[i].y : psf[i].y;  // imag part (conjugated?)
    spec[i].x = (a * c - b * d) * inv_n;            // real, with 1/N folded in
    spec[i].y = (a * d + b * c) * inv_n;            // imag, with 1/N folded in
}

// --- ratio_kernel ----------------------------------------------------------
// Per-pixel data-fidelity ratio, using the SHARED rl_ratio() so the GPU and CPU
// compute the identical guarded division.
//   observed : the fixed blurry input (device)
//   blurred  : H(estimate) for this iteration (device)
//   ratio    : output = observed / blurred (guarded), device
__global__ void ratio_kernel(const double* __restrict__ observed,
                             const double* __restrict__ blurred,
                             double* __restrict__ ratio, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) ratio[i] = rl_ratio(observed[i], blurred[i]);
}

// --- update_kernel ---------------------------------------------------------
// Multiplicative RL update in place, using the SHARED rl_update() (clamped >=0).
//   est        : current estimate (in/out, device)
//   correction : H^T(ratio) back-projected term (device)
__global__ void update_kernel(double* __restrict__ est,
                              const double* __restrict__ correction, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) est[i] = rl_update(est[i], correction[i]);
}

// ---------------------------------------------------------------------------
// embed_psf_host: lay the small (d x d) PSF into a full w*h image so its FFT is
// the transfer function for CIRCULAR convolution -- matching the CPU reference
// exactly. The trick: the PSF's CENTER tap (offset 0,0) must sit at pixel (0,0),
// and a negative offset must WRAP to the high index. Then FFT(image).*FFT(this)
// reproduces out[y,x] = sum src[(y+dy)%h,(x+dx)%w] * psf(dx,dy).
//
// This is done on the host (it runs once, it is tiny) and the result is copied
// to the device and transformed a single time; its spectrum is reused for every
// iteration (and conjugated on the fly for the adjoint step).
// ---------------------------------------------------------------------------
static std::vector<double> embed_psf_host(const Psf& psf, int w, int h) {
    std::vector<double> img(static_cast<std::size_t>(w) * h, 0.0);
    const int r = psf.r, d = psf.d();
    for (int dy = -r; dy <= r; ++dy) {
        const int yy = ((dy) % h + h) % h;                 // wrap negative -> high index
        for (int dx = -r; dx <= r; ++dx) {
            const int xx = ((dx) % w + w) % w;
            img[static_cast<std::size_t>(yy) * w + xx] =
                psf.k[static_cast<std::size_t>(dy + r) * d + (dx + r)];
        }
    }
    return img;
}

// ===========================================================================
// deconvolve_rl_gpu: the host wrapper -- set up cuFFT, run the RL loop, return.
// (Contract documented in kernels.cuh.)
// ===========================================================================
void deconvolve_rl_gpu(const Image& observed, const Psf& psf, int iters,
                       Image& out, float* kernel_ms) {
    const int w = observed.w, h = observed.h;
    const int n  = w * h;                 // real-space pixel count
    const int nf = h * (w / 2 + 1);       // half-complex bins (R2C Hermitian sym.)
    const double inv_n = 1.0 / static_cast<double>(n);

    // ---- Device buffers --------------------------------------------------
    // Real-space (double): observed, estimate, scratch (blurred/ratio/correction).
    // Frequency-space (double-complex): the working spectrum + the PSF spectrum.
    double* d_obs  = nullptr;   // [n]  fixed blurry input
    double* d_est  = nullptr;   // [n]  current estimate (updated in place)
    double* d_real = nullptr;   // [n]  scratch real image (blurred, then ratio, ...)
    cufftDoubleComplex* d_spec    = nullptr;  // [nf] working spectrum
    cufftDoubleComplex* d_psf_spec= nullptr;  // [nf] PSF transfer function (computed once)
    CUDA_CHECK(cudaMalloc(&d_obs,  static_cast<std::size_t>(n)  * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_est,  static_cast<std::size_t>(n)  * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_real, static_cast<std::size_t>(n)  * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_spec,     static_cast<std::size_t>(nf) * sizeof(cufftDoubleComplex)));
    CUDA_CHECK(cudaMalloc(&d_psf_spec, static_cast<std::size_t>(nf) * sizeof(cufftDoubleComplex)));

    // ---- cuFFT plans -----------------------------------------------------
    // Two 2-D plans on a w-by-h image (cuFFT takes dimensions as (rows, cols) =
    // (h, w)):
    //   plan_fwd : CUFFT_D2Z  real(h x w)         -> complex(h x (w/2+1))
    //   plan_inv : CUFFT_Z2D  complex(h x (w/2+1))-> real(h x w)
    // R2C/C2R exploit the Hermitian symmetry of a real signal's spectrum, so we
    // store only the non-redundant half (w/2+1 columns) -- half the memory and
    // work of a full complex FFT. cufftExecD2Z computes, for the input f:
    //     F[ky,kx] = sum_{y,x} f[y,x] * exp(-2*pi*i*(ky*y/h + kx*x/w))
    // i.e. the standard 2-D DFT. cuFFT does NOT normalize, so a forward+inverse
    // round trip scales by N = w*h; we divide by N inside complex_mul_psf.
    cufftHandle plan_fwd, plan_inv;
    CUFFT_CHECK(cufftPlan2d(&plan_fwd, h, w, CUFFT_D2Z));
    CUFFT_CHECK(cufftPlan2d(&plan_inv, h, w, CUFFT_Z2D));

    // ---- One-time setup: copy inputs, build the PSF transfer function -----
    CUDA_CHECK(cudaMemcpy(d_obs, observed.pix.data(),
                          static_cast<std::size_t>(n) * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Flat initial estimate = mean of observed (same as the CPU reference).
    double mean = 0.0;
    for (double v : observed.pix) mean += v;
    mean = (n > 0) ? mean / n : 0.0;
    std::vector<double> est0(static_cast<std::size_t>(n), mean);
    CUDA_CHECK(cudaMemcpy(d_est, est0.data(),
                          static_cast<std::size_t>(n) * sizeof(double),
                          cudaMemcpyHostToDevice));

    // Embed the PSF for circular convolution, copy to d_real, and FFT it ONCE
    // into d_psf_spec. This spectrum is the microscope's transfer function H(k);
    // multiplying by it blurs, multiplying by its conjugate back-projects.
    std::vector<double> psf_img = embed_psf_host(psf, w, h);
    CUDA_CHECK(cudaMemcpy(d_real, psf_img.data(),
                          static_cast<std::size_t>(n) * sizeof(double),
                          cudaMemcpyHostToDevice));
    CUFFT_CHECK(cufftExecD2Z(plan_fwd, d_real, d_psf_spec));   // H(k), kept fixed

    // ---- Launch geometry --------------------------------------------------
    const int block = 256;
    const int grid_n  = (n  + block - 1) / block;   // over real-space pixels
    const int grid_nf = (nf + block - 1) / block;   // over half-spectrum bins

    // ---- The RL iteration loop (timed as the "GPU work") ------------------
    GpuTimer timer;
    timer.start();
    for (int it = 0; it < iters; ++it) {
        // (1) FFT the current estimate into the working spectrum.
        CUFFT_CHECK(cufftExecD2Z(plan_fwd, d_est, d_spec));
        // (2) Apply the blur H(k) and fold in 1/N (conj_psf = 0 -> forward blur).
        complex_mul_psf<<<grid_nf, block>>>(d_spec, d_psf_spec, nf, /*conj=*/0, inv_n);
        // (3) Inverse FFT -> blurred estimate in real space (d_real).
        CUFFT_CHECK(cufftExecZ2D(plan_inv, d_spec, d_real));
        // (4) Per-pixel ratio observed / blurred -> overwrite d_real with ratio.
        //     (Reusing d_real is safe: every pixel reads its own index only.)
        ratio_kernel<<<grid_n, block>>>(d_obs, d_real, d_real, n);

        // (5) FFT the ratio.
        CUFFT_CHECK(cufftExecD2Z(plan_fwd, d_real, d_spec));
        // (6) Multiply by conj(H) = back-project through the flipped PSF, +1/N.
        complex_mul_psf<<<grid_nf, block>>>(d_spec, d_psf_spec, nf, /*conj=*/1, inv_n);
        // (7) Inverse FFT -> correction image in real space (d_real).
        CUFFT_CHECK(cufftExecZ2D(plan_inv, d_spec, d_real));
        // (8) Multiplicative update of the estimate (clamped non-negative).
        update_kernel<<<grid_n, block>>>(d_est, d_real, n);
    }
    *kernel_ms = timer.stop_ms();        // blocks until the GPU finishes the loop
    CUDA_CHECK_LAST("RL iteration loop");

    // ---- Copy the deconvolved estimate back to the host ------------------
    out.w = w; out.h = h;
    out.pix.resize(static_cast<std::size_t>(n));
    CUDA_CHECK(cudaMemcpy(out.pix.data(), d_est,
                          static_cast<std::size_t>(n) * sizeof(double),
                          cudaMemcpyDeviceToHost));

    // ---- Teardown ---------------------------------------------------------
    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);
    CUDA_CHECK(cudaFree(d_obs));
    CUDA_CHECK(cudaFree(d_est));
    CUDA_CHECK(cudaFree(d_real));
    CUDA_CHECK(cudaFree(d_spec));
    CUDA_CHECK(cudaFree(d_psf_spec));
}
