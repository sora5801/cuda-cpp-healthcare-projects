// ===========================================================================
// src/kernels.cu  --  Per-pixel delay-and-sum kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 4.13 : Photoacoustic Image Reconstruction
//
// WHAT THIS FILE DOES
//   Implements the GPU twin of reconstruct_cpu(): the SAME delay-and-sum math,
//   but with one thread per output pixel instead of a serial double loop. The
//   per-pixel physics is not re-derived here -- das_kernel simply calls
//   pa_pixel_das() from pa_core.h, the very function the CPU reference calls, so
//   the GPU and CPU images agree to ~1e-5 (the GPU fuses multiply-adds, so it is
//   a tight tolerance, not bit-identical -- PATTERNS.md §4). main.cu runs both
//   and checks. See ../THEORY.md "GPU mapping".
//
// READ THIS AFTER: pa_core.h (the physics), kernels.cuh (the interface/idea).
// ===========================================================================
#include "kernels.cuh"
#include "pa_core.h"             // pa_pixel_das (shared __host__ __device__ core)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// 16x16 = 256 threads per block. A square 2-D tile is the natural mapping onto a
// 2-D image: it keeps neighbouring pixels (which read overlapping sensor samples)
// in the same block so they hit the same cache lines, and 256 threads is a good
// occupancy default across sm_75..sm_89 (8 warps to hide global-memory latency).
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// das_kernel: thread (px, py) owns image pixel (px, py).
//   It converts its pixel index into world coordinates (metres), then delegates
//   the entire delay-and-sum over all sensors to pa_pixel_das(). Each pixel is
//   independent -> no shared memory, no atomics, one global-memory write at the
//   end. The inner sensor loop lives inside pa_pixel_das (shared with the CPU).
// ---------------------------------------------------------------------------
__global__ void das_kernel(const float* __restrict__ d_sx,
                           const float* __restrict__ d_sy,
                           const float* __restrict__ d_sig,
                           int n_sensors, int n_samples, int img,
                           float world_half, float pix,
                           float inv_c, float inv_dt, float inv_ns,
                           float* __restrict__ d_image) {
    // This thread's pixel column (px) and row (py) within the img x img image.
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    // GUARD THE RAGGED EDGE TILES: img is rarely a multiple of 16, so the border
    // blocks contain threads with px or py >= img. They must not write out of
    // bounds, so they return early and do nothing.
    if (px >= img || py >= img) return;

    // Map the integer pixel to world coordinates in metres: pixel 0 -> -W,
    // pixel img-1 -> +W (identical to the CPU reference's mapping).
    const float wx = -world_half + px * pix;
    const float wy = -world_half + py * pix;

    // The whole reconstruction for this pixel, computed by the SHARED core so it
    // matches the CPU bit-for-bit. Row-major store: image[py*img + px].
    d_image[(size_t)py * img + px] =
        pa_pixel_das(wx, wy, d_sx, d_sy, d_sig, n_sensors, n_samples,
                     inv_c, inv_dt, inv_ns);
}

// ---------------------------------------------------------------------------
// reconstruct_gpu: host wrapper. The five canonical CUDA steps:
//   (1) allocate device buffers  (2) copy inputs host->device
//   (3) launch the 2-D grid       (4) copy the image device->host
//   (5) free device memory
// Only step (3) is timed with CUDA events, so *kernel_ms is the kernel cost, not
// the PCIe transfer cost (transfers are discussed separately in THEORY.md).
// ---------------------------------------------------------------------------
void reconstruct_gpu(const PAProblem& pa, std::vector<float>& image,
                     float* kernel_ms) {
    const int N = pa.img, n_sensors = pa.n_sensors, n_samples = pa.n_samples;
    const std::size_t img_cells  = static_cast<std::size_t>(N) * N;
    const std::size_t sig_count  = static_cast<std::size_t>(n_sensors) * n_samples;
    image.assign(img_cells, 0.0f);

    // (1) Device buffers. d_ prefix = DEVICE pointer (dereferencing on the host
    //     would crash). Three read-only inputs + one output image.
    float *d_sx = nullptr, *d_sy = nullptr, *d_sig = nullptr, *d_image = nullptr;
    CUDA_CHECK(cudaMalloc(&d_sx,    n_sensors * sizeof(float)));   // sensor x [m]
    CUDA_CHECK(cudaMalloc(&d_sy,    n_sensors * sizeof(float)));   // sensor y [m]
    CUDA_CHECK(cudaMalloc(&d_sig,   sig_count * sizeof(float)));   // all traces
    CUDA_CHECK(cudaMalloc(&d_image, img_cells * sizeof(float)));   // output image

    // (2) Upload the read-only geometry and traces (host -> device).
    CUDA_CHECK(cudaMemcpy(d_sx,  pa.sx.data(),  n_sensors * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sy,  pa.sy.data(),  n_sensors * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sig, pa.sig.data(), sig_count * sizeof(float), cudaMemcpyHostToDevice));

    // Precompute the same scalars the CPU used, so both feed pa_pixel_das
    // identical float operands (guarantees exact agreement).
    const float pix    = (N > 1) ? (2.0f * pa.world_half / (N - 1)) : 0.0f;  // m/pixel
    const float inv_c  = 1.0f / pa.c;
    const float inv_dt = 1.0f / pa.dt;
    const float inv_ns = 1.0f / static_cast<float>(n_sensors);

    // (3) Launch a 2-D grid of TILE x TILE blocks covering the N x N image.
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    GpuTimer timer;
    timer.start();
    das_kernel<<<grid, block>>>(d_sx, d_sy, d_sig, n_sensors, n_samples, N,
                                pa.world_half, pix, inv_c, inv_dt, inv_ns, d_image);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time (ms)
    CUDA_CHECK_LAST("das_kernel");         // catch launch + execution errors

    // (4) Bring the reconstructed image back to the host vector.
    CUDA_CHECK(cudaMemcpy(image.data(), d_image, img_cells * sizeof(float), cudaMemcpyDeviceToHost));

    // (5) Free everything (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_sx));
    CUDA_CHECK(cudaFree(d_sy));
    CUDA_CHECK(cudaFree(d_sig));
    CUDA_CHECK(cudaFree(d_image));
}
