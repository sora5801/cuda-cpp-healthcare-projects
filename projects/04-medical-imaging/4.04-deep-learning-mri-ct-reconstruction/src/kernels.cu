// ===========================================================================
// src/kernels.cu  --  GPU unrolled reconstruction: kernels + host driver loop
// ---------------------------------------------------------------------------
// Project 4.4 : Deep-Learning MRI/CT Reconstruction  (REDUCED-SCOPE TEACHING VERSION)
//
// WHAT THIS FILE DOES
//   Implements the GPU twin of recon_cpu(). One unrolled reconstruction is a
//   host-driven loop over `stages` cascade steps; each step is three kernels:
//     (A) regularize_kernel : image-domain denoiser step  x <- x + l*(D(x)-x)
//     (B) dft_forward_kernel : image -> k-space (direct 2-D DFT)
//     (C) dc_idft_kernel     : overwrite SAMPLED k-space bins with the measured
//                              values (data consistency), then k-space -> image
//   Kernels (A) and (C) each read one buffer and write another (no in-place
//   hazards), so we PING-PONG two image buffers -- exactly the double-buffer
//   idiom from the stencil flagship 6.04.
//
//   All per-element arithmetic is delegated to the SHARED __host__ __device__
//   cores (recon_core.h, dft_core.h) so this GPU path and the CPU reference
//   compute identical sums -- the basis for verification. See ../THEORY.md.
//
// READ THIS AFTER: kernels.cuh (types), recon_core.h + dft_core.h (the math).
// ===========================================================================
#include "kernels.cuh"
#include "recon_core.h"          // denoise/regularize per-pixel stencil (shared)
#include "dft_core.h"            // forward/inverse DFT per-output (shared)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// 16x16 = 256 threads per block over the 2-D image. 256 is a warp multiple that
// hides latency well on sm_75..sm_89; a 2-D block matches the 2-D data so the
// thread indices map straight onto (x,y) pixel coordinates.
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// (A) regularize_kernel: one thread per pixel applies ONE denoiser step.
//   Launch: grid = ceil(nx/16) x ceil(ny/16) blocks, block = 16x16 threads.
//   Thread-to-data map: pixel (x,y) = (blockIdx.x*16+threadIdx.x,
//                                      blockIdx.y*16+threadIdx.y).
//   Memory: reads the 3x3 neighbourhood of `in` from global memory, writes one
//   pixel of `out`. Neighbours are re-read by adjacent threads; on a real net we
//   would stage a tile in shared memory (as flagship 7.10 does), but at this
//   teaching size the L1/L2 cache already absorbs the reuse -- we keep the code
//   simple and note the optimization in THEORY. No atomics: each output pixel is
//   independent, so there are no races.
// ---------------------------------------------------------------------------
__global__ void regularize_kernel(const float* __restrict__ in, int ny, int nx,
                                  float lambda, float* __restrict__ out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's column
    const int y = blockIdx.y * blockDim.y + threadIdx.y;   // this thread's row
    if (x >= nx || y >= ny) return;                        // guard ragged tiles
    const float self = in[img_idx(y, x, nx)];              // current pixel value
    // The whole update is the shared HD function -> identical to the CPU loop.
    out[img_idx(y, x, nx)] = regularize_pixel(in, self, y, x, ny, nx, lambda);
}

// ---------------------------------------------------------------------------
// (B) dft_forward_kernel: one thread per OUTPUT FREQUENCY (v,u).
//   Launch: same 16x16 grid over the ny x nx frequency grid.
//   Each thread reduces over ALL image pixels to produce one complex k-space
//   sample -- a "gather + reduce per output". Reads `img` (global), writes one
//   (re,im) pair. Independent outputs -> no atomics. This is the O(N^2) direct
//   transform; production uses cuFFT (O(N log N)) -- see dft_core.h header.
// ---------------------------------------------------------------------------
__global__ void dft_forward_kernel(const float* __restrict__ img, int ny, int nx,
                                   float* __restrict__ kre, float* __restrict__ kim) {
    const int u = blockIdx.x * blockDim.x + threadIdx.x;   // frequency column
    const int v = blockIdx.y * blockDim.y + threadIdx.y;   // frequency row
    if (u >= nx || v >= ny) return;
    float re, im;
    dft_forward_pixel(img, v, u, ny, nx, &re, &im);        // shared reduction
    const std::size_t k = kidx(v, u, nx);
    kre[k] = re;
    kim[k] = im;
}

// ---------------------------------------------------------------------------
// (C) dc_idft_kernel: DATA CONSISTENCY then inverse transform, fused.
//   Step 1 (data consistency): for every SAMPLED frequency (mask==1), replace
//     the current estimate's k-space value with the MEASURED value. This is the
//     projection that keeps the reconstruction faithful to the scan. We apply it
//     to a per-thread LOCAL copy of the needed bin -- but because the inverse
//     transform reads ALL bins, we instead pre-apply DC to the whole k-space
//     buffer in a tiny separate kernel (dc_apply_kernel) BEFORE this one. That
//     keeps each kernel single-purpose and race-free.
//   Step 2 (inverse DFT): one thread per output pixel reduces over all (already
//     data-consistent) frequencies to reconstruct the image pixel.
//   Launch: 16x16 grid over the ny x nx image. Reads kre/kim (global), writes
//   one pixel of `img_out`. Independent outputs -> no atomics.
// ---------------------------------------------------------------------------
__global__ void dc_idft_kernel(const float* __restrict__ kre, const float* __restrict__ kim,
                               int ny, int nx, float* __restrict__ img_out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    img_out[img_idx(y, x, nx)] = idft_pixel(kre, kim, y, x, ny, nx);  // shared iDFT
}

// ---------------------------------------------------------------------------
// dc_apply_kernel: the data-consistency projection itself, one thread per bin.
//   For each frequency (v,u): if it was measured (mask==1), OVERWRITE the current
//   estimate's k-space with the measured value; otherwise leave the estimate's
//   value untouched (that is what the denoiser is allowed to fill in). In-place
//   on the estimate's k-space buffers; each bin is independent -> no races.
// ---------------------------------------------------------------------------
__global__ void dc_apply_kernel(const int* __restrict__ mask,
                                const float* __restrict__ meas_re,
                                const float* __restrict__ meas_im,
                                int n,
                                float* __restrict__ est_re, float* __restrict__ est_im) {
    const int k = blockIdx.x * blockDim.x + threadIdx.x;   // flat frequency index
    if (k >= n) return;
    if (mask[k]) {                    // measured -> trust the scan, replace
        est_re[k] = meas_re[k];
        est_im[k] = meas_im[k];
    }
    // else: keep the estimate (denoiser-filled) value for this skipped frequency.
}

// ---------------------------------------------------------------------------
// recon_gpu: orchestrate the whole unrolled reconstruction on the GPU.
//   The initial estimate is the ZERO-FILLED reconstruction: put the measured
//   k-space (zeros where unsampled) straight through the inverse transform. Then
//   run `stages` cascade steps of (denoise) -> (forward DFT) -> (data-consistency
//   + inverse DFT). We ping-pong two image buffers so no kernel writes a buffer
//   it is also reading. All stage kernels are timed together with CUDA events.
// ---------------------------------------------------------------------------
void recon_gpu(const Acquisition& acq, const ReconParams& p,
               std::vector<float>& recon, float* kernel_ms) {
    const int ny = acq.ny, nx = acq.nx, n = acq.n();
    const std::size_t fbytes = static_cast<std::size_t>(n) * sizeof(float);
    const std::size_t ibytes = static_cast<std::size_t>(n) * sizeof(int);
    recon.assign(static_cast<std::size_t>(n), 0.0f);

    // --- Device buffers -----------------------------------------------------
    //   d_imgA / d_imgB : ping-pong image buffers (current / next estimate).
    //   d_kre / d_kim   : the estimate's k-space (reused every stage).
    //   d_mre / d_mim   : the MEASURED k-space (constant across the recon).
    //   d_mask          : the binary sampling mask (constant).
    float *d_imgA=nullptr, *d_imgB=nullptr, *d_kre=nullptr, *d_kim=nullptr;
    float *d_mre=nullptr, *d_mim=nullptr;
    int   *d_mask=nullptr;
    CUDA_CHECK(cudaMalloc(&d_imgA, fbytes));
    CUDA_CHECK(cudaMalloc(&d_imgB, fbytes));
    CUDA_CHECK(cudaMalloc(&d_kre,  fbytes));
    CUDA_CHECK(cudaMalloc(&d_kim,  fbytes));
    CUDA_CHECK(cudaMalloc(&d_mre,  fbytes));
    CUDA_CHECK(cudaMalloc(&d_mim,  fbytes));
    CUDA_CHECK(cudaMalloc(&d_mask, ibytes));

    // Upload the measurement + mask (they never change during the recon).
    CUDA_CHECK(cudaMemcpy(d_mre,  acq.kmeas_re.data(), fbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mim,  acq.kmeas_im.data(), fbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mask, acq.mask.data(),     ibytes, cudaMemcpyHostToDevice));

    // 2-D launch geometry over the image/frequency grid, plus a 1-D geometry for
    // the per-bin data-consistency kernel.
    dim3 block2d(TILE, TILE);
    dim3 grid2d((nx + TILE - 1) / TILE, (ny + TILE - 1) / TILE);
    const int block1d = 256;
    const int grid1d  = (n + block1d - 1) / block1d;

    GpuTimer timer;
    timer.start();

    // --- Initial estimate: zero-filled iDFT of the measured k-space ---------
    // Copy the measured k-space into the estimate's k-space, then invert. Where
    // the mask is 0 the measured value is already 0 (unsampled), so this is the
    // classic "zero-filled" starting image every unrolled recon begins from.
    CUDA_CHECK(cudaMemcpy(d_kre, d_mre, fbytes, cudaMemcpyDeviceToDevice));
    CUDA_CHECK(cudaMemcpy(d_kim, d_mim, fbytes, cudaMemcpyDeviceToDevice));
    dc_idft_kernel<<<grid2d, block2d>>>(d_kre, d_kim, ny, nx, d_imgA);
    CUDA_CHECK_LAST("dc_idft_kernel(init)");

    // d_imgA now holds the current estimate. We alternate A<->B each stage.
    float* cur = d_imgA;   // buffer holding the current image estimate
    float* nxt = d_imgB;   // buffer the next kernel writes into

    // --- The unrolled cascade ----------------------------------------------
    for (int s = 0; s < p.stages; ++s) {
        // (A) image-domain denoiser step: cur -> nxt.
        regularize_kernel<<<grid2d, block2d>>>(cur, ny, nx, p.lambda, nxt);
        CUDA_CHECK_LAST("regularize_kernel");

        // (B) forward transform of the denoised image nxt -> estimate k-space.
        dft_forward_kernel<<<grid2d, block2d>>>(nxt, ny, nx, d_kre, d_kim);
        CUDA_CHECK_LAST("dft_forward_kernel");

        // (C1) data consistency: overwrite the sampled bins with the measurement.
        dc_apply_kernel<<<grid1d, block1d>>>(d_mask, d_mre, d_mim, n, d_kre, d_kim);
        CUDA_CHECK_LAST("dc_apply_kernel");

        // (C2) inverse transform of the data-consistent k-space -> cur (reused).
        //   Writing back into `cur` is safe: dc_idft reads only k-space, not cur.
        dc_idft_kernel<<<grid2d, block2d>>>(d_kre, d_kim, ny, nx, cur);
        CUDA_CHECK_LAST("dc_idft_kernel(stage)");
        // `cur` now holds the new estimate for the next stage (nxt was scratch).
    }
    *kernel_ms = timer.stop_ms();

    // The final estimate lives in `cur`. Bring it home.
    CUDA_CHECK(cudaMemcpy(recon.data(), cur, fbytes, cudaMemcpyDeviceToHost));

    // Free everything (no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_imgA));
    CUDA_CHECK(cudaFree(d_imgB));
    CUDA_CHECK(cudaFree(d_kre));
    CUDA_CHECK(cudaFree(d_kim));
    CUDA_CHECK(cudaFree(d_mre));
    CUDA_CHECK(cudaFree(d_mim));
    CUDA_CHECK(cudaFree(d_mask));
}
