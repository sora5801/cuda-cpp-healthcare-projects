// ===========================================================================
// src/kernels.cu  --  The GPU Non-Local-Means kernel and its host wrapper
// ---------------------------------------------------------------------------
// Project 4.9 : Image Denoising & Restoration  (Non-Local Means)
//
// WHAT THIS FILE DOES
//   Implements the device kernel (nlm_kernel) and the host-side glue
//   (denoise_gpu) that allocates GPU memory, uploads the noisy image, launches
//   the kernel over a 2-D thread grid, times it with CUDA events, and copies the
//   denoised image back. This is the GPU twin of denoise_cpu() in
//   reference_cpu.cpp; main.cu runs both and checks they agree.
//
//   The per-pixel arithmetic is NOT written here -- the kernel calls
//   nlm_pixel() from nlm_core.h, the SAME function the CPU reference uses.
//   nvcc compiles nlm_core.h with NLM_HD == "__host__ __device__", so the
//   identical code runs on-device. That shared core is what makes GPU==CPU exact.
//
// READ THIS AFTER: kernels.cuh (the thread-mapping idea) and nlm_core.h (math).
// ===========================================================================
#include "kernels.cuh"
#include "nlm_core.h"            // nlm_pixel  (per-pixel NLM math, __host__ __device__)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Tile side for the 2-D block. 16x16 = 256 threads/block:
//   * 256 is a multiple of the 32-lane warp and gives the scheduler 8 warps to
//     hide the long global-memory latency of the patch reads.
//   * A SQUARE tile matches the 2-D image so neighbouring threads in a block
//     touch neighbouring pixels -> their patch/search reads overlap heavily ->
//     the L1/L2 caches serve most of them (the reason this naive version is
//     already fast without explicit shared-memory tiling; see THEORY.md).
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// nlm_kernel: thread (col,row) owns output pixel (row,col).
//   Launch config (set in denoise_gpu):
//     block = TILE x TILE threads (256)
//     grid  = ceil(width/TILE) x ceil(height/TILE) blocks -> covers the image
//   Thread-to-data map:
//     col = blockIdx.x*blockDim.x + threadIdx.x
//     row = blockIdx.y*blockDim.y + threadIdx.y
//   Memory spaces touched: reads the noisy image from GLOBAL memory (cached);
//   accumulates the two running sums in REGISTERS inside nlm_pixel; writes one
//   output pixel to GLOBAL memory. No shared memory, no atomics -- every output
//   pixel is independent, so there is nothing to synchronise or reduce across
//   threads. This is the textbook "per-output-pixel gather" pattern (like the
//   CT backprojection flagship 4.01).
// ---------------------------------------------------------------------------
__global__ void nlm_kernel(const float* __restrict__ in, NlmParams params,
                           float* __restrict__ out) {
    // The pixel coordinate this thread is responsible for.
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;

    // GUARD THE RAGGED EDGE TILES: width/height are rarely exact multiples of
    // TILE, so blocks on the right/bottom edge spawn threads outside the image.
    // They must return before touching memory, or they would read/write out of
    // bounds (an illegal-address fault).
    if (col >= params.width || row >= params.height) return;

    // The ENTIRE per-pixel computation is delegated to the shared core so the
    // device result is bit-identical to the host reference. nlm_pixel() sweeps
    // the search window, computes each patch distance, exponentiates it into a
    // weight, and returns Σ(w*value)/Σ(w) for this pixel.
    out[(size_t)row * params.width + col] = nlm_pixel(in, params, row, col);
}

// ---------------------------------------------------------------------------
// denoise_gpu: host wrapper. The five canonical steps of a CUDA computation:
//   (1) allocate device memory  (2) copy the noisy image host->device
//   (3) launch the 2-D grid      (4) copy the denoised image device->host
//   (5) free device memory
// We time ONLY step (3) with CUDA events so the reported figure is the kernel
// cost, not the PCIe transfer cost (transfers are discussed in THEORY.md).
// ---------------------------------------------------------------------------
void denoise_gpu(const Image& in, const NlmParams& params, Image& out, float* kernel_ms) {
    out.width  = in.width;
    out.height = in.height;
    out.pix.assign(in.pix.size(), 0.0f);
    const std::size_t bytes = in.pix.size() * sizeof(float);

    // (1) Device buffers. The d_ prefix marks DEVICE pointers (CLAUDE.md §12):
    //     dereferencing one on the host would crash, so the naming matters.
    float *d_in = nullptr, *d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_in,  bytes));   // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_out, bytes));

    // (2) Upload the noisy image once; every thread will read from it.
    CUDA_CHECK(cudaMemcpy(d_in, in.pix.data(), bytes, cudaMemcpyHostToDevice));

    // (3) Launch a 2-D grid of TILE x TILE blocks covering the whole image.
    //     The ceiling division rounds the block count up so the ragged edge is
    //     still covered (the kernel's bounds check discards the extra threads).
    dim3 block(TILE, TILE);
    dim3 grid((in.width  + TILE - 1) / TILE,
              (in.height + TILE - 1) / TILE);
    GpuTimer timer;
    timer.start();
    nlm_kernel<<<grid, block>>>(d_in, params, d_out);
    *kernel_ms = timer.stop_ms();            // GPU-measured kernel time (ms)
    CUDA_CHECK_LAST("nlm_kernel");           // catch launch + execution errors

    // (4) Bring the denoised image back to the host.
    CUDA_CHECK(cudaMemcpy(out.pix.data(), d_out, bytes, cudaMemcpyDeviceToHost));

    // (5) Always free what we allocated (there is no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_in));
    CUDA_CHECK(cudaFree(d_out));
}
