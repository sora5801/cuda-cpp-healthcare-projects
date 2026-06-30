// ===========================================================================
// src/kernels.cu  --  cuFFT ramp filter + per-pixel back-projection gather
// ---------------------------------------------------------------------------
// Project 2.31 : Cryo-EM Tilt-Series Alignment & Tomogram Reconstruction
//
// This file holds the two GPU teaching points:
//   * ramp_filter_gpu  -- batched R2C FFT (cuFFT) -> ramp multiply -> C2R FFT.
//   * backproject_gpu  -- one thread per output pixel gathers from every tilt.
// Both are GPU twins of routines in reference_cpu.cpp; main.cu runs both and
// checks they agree. The per-sample sampling math is shared via wbp_core.h, so
// the GPU and CPU back-projections run identical float arithmetic. See
// ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "wbp_core.h"            // sample_projection_hd, WBP_PI_F (shared core)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cufft.h>               // cufftHandle, cufftExecR2C/C2R, cufftPlan1d
#include <cmath>                 // std::cos
#include <cstdio>                // std::fprintf
#include <cstdlib>               // std::exit, EXIT_FAILURE
#include <vector>

// ---------------------------------------------------------------------------
// CUFFT_CHECK: cuFFT has its own status enum (cufftResult), so it needs its own
//   guard macro that mirrors CUDA_CHECK. Every cuFFT call is wrapped: a failed
//   plan or exec means the filtered sinogram is meaningless, so we abort loudly
//   rather than back-project garbage.
// ---------------------------------------------------------------------------
#define CUFFT_CHECK(call)                                                       \
    do {                                                                        \
        cufftResult st__ = (call);                                             \
        if (st__ != CUFFT_SUCCESS) {                                           \
            std::fprintf(stderr, "[CUFFT_CHECK] %s:%d -> cufft error %d\n",    \
                         __FILE__, __LINE__, static_cast<int>(st__));          \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                       \
    } while (0)

// 16x16 = 256 threads/block: a square tile matching the 2-D slice, good
// occupancy on sm_75..sm_89 (the repo's target arch range).
static constexpr int TILE = 16;
// 1-D block size for the element-wise ramp-multiply kernel.
static constexpr int RAMP_BLOCK = 256;

// ===========================================================================
// STEP 2 (GPU): ramp filter in the frequency domain with cuFFT
// ===========================================================================

// ---------------------------------------------------------------------------
// ramp_apply_kernel: multiply each complex spectral bin by its real ramp weight.
//   THREAD MAP: i = blockIdx.x*blockDim.x + threadIdx.x indexes the flattened
//   [n_tilts * nf] spectrum; the ramp weight depends only on the bin number
//   within a projection (i % nf), broadcast across all tilts. cufftComplex is
//   float2 (.x real, .y imag), so we scale both components by the same real
//   weight. No atomics, no shared memory -- a pure element-wise map.
// ---------------------------------------------------------------------------
__global__ void ramp_apply_kernel(float2* __restrict__ spec,
                                  const float* __restrict__ ramp,
                                  int nf, int total) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;                 // guard the ragged last block
    const int f = i % nf;                   // spectral-bin index within this proj
    const float w = ramp[f];                // real ramp weight |f|*apodization
    float2 v = spec[i];
    v.x *= w;                               // scale real part
    v.y *= w;                               // scale imaginary part
    spec[i] = v;
}

// ---------------------------------------------------------------------------
// ramp_filter_gpu: the whole FFT-domain ramp filter.
//
//   PIPELINE (per projection row, all batched together):
//     1. Forward R2C FFT  : real row[n_det] -> complex spectrum[nf], nf=n_det/2+1.
//        cuFFT computes X[f] = sum_j row[j] exp(-2*pi*i*f*j/n_det) in O(n log n).
//     2. Ramp multiply    : X[f] *= ramp[f], where ramp[f] = |f| with a
//        raised-cosine roll-off near Nyquist (apodization tames high-frequency
//        noise; pure |f| is the unapodized Ram-Lak filter).
//     3. Inverse C2R FFT  : spectrum -> real filtered row. cuFFT's inverse is
//        UN-normalized, so we divide by n_det (folded into the ramp scale below).
//
//   WHAT HAND-ROLLING WOULD TAKE: a radix-2/mixed-radix FFT with bit-reversal
//   and twiddle factors per length -- exactly the kind of solved primitive a
//   library should own (CLAUDE.md sec.6.1.6). We keep the *ramp* by hand (it is
//   the physics) and let cuFFT own the transform.
//
//   The ramp scale is chosen so the result matches ramp_filter_cpu()'s spatial
//   Ram-Lak filter up to the documented edge tolerance (see main.cu / THEORY).
// ---------------------------------------------------------------------------
void ramp_filter_gpu(const TiltSeries& ts, const std::vector<float>& aligned,
                     std::vector<float>& filtered, float* kernel_ms) {
    const int K = ts.n_tilts, n = ts.n_det;
    const int nf = n / 2 + 1;                       // R2C output length per row
    const int total = K * nf;
    filtered.assign(aligned.size(), 0.0f);

    // ---- Build the ramp weight table on the host, then upload --------------
    // We use the SHARED ramp_weight_hd() (wbp_core.h) -- the very same |f|*apod
    // ramp the CPU reference DFT applies -- so the two filters are identical math
    // and verify tightly. The only GPU-specific factor is inv_n: cuFFT's forward
    // then inverse transform multiplies the data by n (its inverse is
    // un-normalized), so we divide the ramp by n to cancel it.
    std::vector<float> h_ramp(static_cast<std::size_t>(nf));
    const float inv_n = 1.0f / static_cast<float>(n);          // cuFFT inverse norm
    for (int f = 0; f < nf; ++f) {
        h_ramp[static_cast<std::size_t>(f)] = ramp_weight_hd(f, nf, n, ts.ds) * inv_n;
    }

    // ---- Device buffers ----------------------------------------------------
    // cufftReal == float, cufftComplex == float2.
    cufftReal*    d_in   = nullptr;   // [K*n]   real input (aligned projections)
    cufftComplex* d_spec = nullptr;   // [K*nf]  complex spectra
    cufftReal*    d_out  = nullptr;   // [K*n]   real filtered output
    float*        d_ramp = nullptr;   // [nf]    ramp weights
    CUDA_CHECK(cudaMalloc(&d_in,   static_cast<std::size_t>(K) * n  * sizeof(cufftReal)));
    CUDA_CHECK(cudaMalloc(&d_spec, static_cast<std::size_t>(K) * nf * sizeof(cufftComplex)));
    CUDA_CHECK(cudaMalloc(&d_out,  static_cast<std::size_t>(K) * n  * sizeof(cufftReal)));
    CUDA_CHECK(cudaMalloc(&d_ramp, static_cast<std::size_t>(nf)     * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_in, aligned.data(),
                          static_cast<std::size_t>(K) * n * sizeof(cufftReal),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ramp, h_ramp.data(),
                          static_cast<std::size_t>(nf) * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ---- Plans: K independent length-n transforms, contiguous batch --------
    // cufftPlan1d(plan, n, type, batch) lays out `batch` rows back-to-back with
    // input stride n and output stride nf -- exactly our row-major sinogram.
    cufftHandle plan_fwd, plan_inv;
    CUFFT_CHECK(cufftPlan1d(&plan_fwd, n, CUFFT_R2C, K));   // real -> complex
    CUFFT_CHECK(cufftPlan1d(&plan_inv, n, CUFFT_C2R, K));   // complex -> real

    const int grid = (total + RAMP_BLOCK - 1) / RAMP_BLOCK;

    GpuTimer timer;
    timer.start();
    CUFFT_CHECK(cufftExecR2C(plan_fwd, d_in, d_spec));      // (1) forward FFT
    ramp_apply_kernel<<<grid, RAMP_BLOCK>>>(                 // (2) ramp multiply
        reinterpret_cast<float2*>(d_spec), d_ramp, nf, total);
    CUDA_CHECK_LAST("ramp_apply_kernel");
    CUFFT_CHECK(cufftExecC2R(plan_inv, d_spec, d_out));      // (3) inverse FFT
    *kernel_ms = timer.stop_ms();

    CUDA_CHECK(cudaMemcpy(filtered.data(), d_out,
                          static_cast<std::size_t>(K) * n * sizeof(cufftReal),
                          cudaMemcpyDeviceToHost));

    cufftDestroy(plan_fwd);
    cufftDestroy(plan_inv);
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_spec));
    CUDA_CHECK(cudaFree(d_out));
    CUDA_CHECK(cudaFree(d_ramp));
}

// ===========================================================================
// STEP 4 (GPU): weighted back-projection as a per-pixel gather
// ===========================================================================

// ---------------------------------------------------------------------------
// backproject_kernel: thread (px,py) owns slice pixel (px,py).
//   It walks every tilt angle, asks the SHARED sampler sample_projection_hd()
//   for this pixel's contribution from that tilt (ray hits detector at
//   s = wx*cos + wy*sin; linear interpolation), and accumulates. Each pixel is
//   independent -> no shared memory, no atomics; the loop is byte-for-byte the
//   same one backproject_cpu() runs, which is why CPU and GPU agree tightly.
// ---------------------------------------------------------------------------
__global__ void backproject_kernel(const float* __restrict__ filtered,
                                   const float* __restrict__ cosv,
                                   const float* __restrict__ sinv,
                                   int n_tilts, int n_det, int N,
                                   float ds, float center, float W,
                                   float pix, float scale,
                                   float* __restrict__ slice) {
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= N || py >= N) return;                  // guard ragged edge tiles

    const float wx = -W + px * pix;                  // world coords of this pixel
    const float wy = -W + py * pix;

    float acc = 0.0f;
    for (int k = 0; k < n_tilts; ++k) {
        const float* row = filtered + (size_t)k * n_det;
        acc += sample_projection_hd(row, n_det, wx, wy, cosv[k], sinv[k], ds, center);
    }
    slice[(size_t)py * N + px] = acc * scale;        // scale = pi/n_tilts (d-theta)
}

// ---------------------------------------------------------------------------
// backproject_gpu: upload inputs, launch the 2-D grid, copy the slice back.
//   Mirrors backproject_cpu()'s setup so the two share every constant.
// ---------------------------------------------------------------------------
void backproject_gpu(const TiltSeries& ts, const std::vector<float>& filtered,
                     const std::vector<float>& cosv, const std::vector<float>& sinv,
                     std::vector<float>& slice, float* kernel_ms) {
    const int N = ts.img, n_det = ts.n_det, K = ts.n_tilts;
    const std::size_t cells = static_cast<std::size_t>(N) * N;
    slice.assign(cells, 0.0f);

    float *d_filtered = nullptr, *d_cos = nullptr, *d_sin = nullptr, *d_slice = nullptr;
    CUDA_CHECK(cudaMalloc(&d_filtered, filtered.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cos, cosv.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, sinv.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_slice, cells * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_filtered, filtered.data(),
                          filtered.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, cosv.data(), cosv.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, sinv.data(), sinv.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    const float center = 0.5f * (n_det - 1);                  // detector index of s=0
    const float W   = ts.world_half;
    const float pix = (N > 1) ? (2.0f * W / (N - 1)) : 0.0f;   // world units / pixel
    const float scale = WBP_PI_F / static_cast<float>(K);      // d(theta) ~ pi/n_tilts

    // 2-D grid of TILE x TILE blocks covering the N x N slice.
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    GpuTimer timer;
    timer.start();
    backproject_kernel<<<grid, block>>>(d_filtered, d_cos, d_sin, K, n_det, N,
                                        ts.ds, center, W, pix, scale, d_slice);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("backproject_kernel");

    CUDA_CHECK(cudaMemcpy(slice.data(), d_slice, cells * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_filtered));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_slice));
}
