// ===========================================================================
// src/kernels.cu  --  Per-pixel motion-compensated reconstruction kernel
// ---------------------------------------------------------------------------
// Project 4.19 : Motion-Compensated 4D-CT Reconstruction (2-D teaching version)
//
// WHAT THIS FILE DOES
//   Implements the device kernel (reconstruct_kernel) and the host glue
//   (reconstruct_gpu) that allocates GPU memory, moves data, launches the
//   kernel, times it, and brings the image back. The GPU twin of
//   reconstruct_cpu() in reference_cpu.cpp -- and it produces a BIT-IDENTICAL
//   image because both call the same mc_pixel() from mc4dct.h.
//
//   The kernel body is deliberately tiny: all the reconstruction physics (pixel
//   -> world, DVF warp, project onto detector, interpolate, accumulate) is in
//   the shared header. The kernel's only job is the thread-to-pixel MAPPING.
//
// READ THIS AFTER: kernels.cuh, mc4dct.h.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// 16x16 = 256 threads/block: a square tile that matches the 2-D image and gives
// good occupancy (8 warps) on sm_75..sm_89. Same choice as flagship 4.01.
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// reconstruct_kernel: thread (px,py) owns output pixel (px,py).
//   Launch config (set in reconstruct_gpu):
//     grid  = ceil(img/TILE) x ceil(img/TILE) blocks
//     block = TILE x TILE threads
//   Thread-to-data map:
//     px = blockIdx.x*blockDim.x + threadIdx.x
//     py = blockIdx.y*blockDim.y + threadIdx.y
//   Memory: each thread reads cosv/sinv/filtered from GLOBAL memory and writes
//   ONE output pixel. No shared memory or atomics: pixels are independent, and
//   the whole reduction (sum over phases and angles) is private to the thread.
//   This is the canonical CT reconstruction GPU pattern (a per-pixel GATHER).
// ---------------------------------------------------------------------------
__global__ void reconstruct_kernel(Geom g,
                                   const float* __restrict__ cosv,
                                   const float* __restrict__ sinv,
                                   const float* __restrict__ filtered,
                                   int motion_comp,
                                   float* __restrict__ image) {
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= g.img || py >= g.img) return;   // guard the ragged edge tiles

    // ALL the work is the shared __host__ __device__ routine -- identical to the
    // CPU reference's inner call. That identity is what makes verification exact.
    image[(long long)py * g.img + px] =
        mc_pixel(px, py, g, cosv, sinv, filtered, motion_comp);
}

// ---------------------------------------------------------------------------
// reconstruct_gpu: host wrapper. The five canonical steps of a CUDA computation:
//   (1) allocate device memory   (2) copy inputs host->device
//   (3) launch the kernel         (4) copy result device->host
//   (5) free device memory
//   We time ONLY step (3) with CUDA events so the reported figure is the kernel
//   cost, not the PCIe transfer cost (discussed separately in THEORY.md).
// ---------------------------------------------------------------------------
void reconstruct_gpu(const FourDCTProblem& prob, const std::vector<float>& filtered,
                     const std::vector<float>& cosv, const std::vector<float>& sinv,
                     int motion_comp, std::vector<float>& image, float* kernel_ms) {
    const Geom& g = prob.geom;
    const std::size_t img_cells = static_cast<std::size_t>(g.img) * g.img;
    image.assign(img_cells, 0.0f);

    // (1) Device buffers (d_ prefix marks DEVICE pointers -- CLAUDE.md section 12).
    float *d_filtered = nullptr, *d_cos = nullptr, *d_sin = nullptr, *d_image = nullptr;
    CUDA_CHECK(cudaMalloc(&d_filtered, filtered.size() * sizeof(float)));  // may fail: OOM
    CUDA_CHECK(cudaMalloc(&d_cos, cosv.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, sinv.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_image, img_cells * sizeof(float)));

    // (2) Copy inputs H2D. .data() is the contiguous backing array of the vector.
    CUDA_CHECK(cudaMemcpy(d_filtered, filtered.data(), filtered.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, cosv.data(), cosv.size() * sizeof(float),
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, sinv.data(), sinv.size() * sizeof(float),
                          cudaMemcpyHostToDevice));

    // (3) Launch a 2-D grid of TILE x TILE blocks covering the img x img image.
    dim3 block(TILE, TILE);
    dim3 grid((g.img + TILE - 1) / TILE, (g.img + TILE - 1) / TILE);
    GpuTimer timer;
    timer.start();
    reconstruct_kernel<<<grid, block>>>(g, d_cos, d_sin, d_filtered, motion_comp, d_image);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time
    CUDA_CHECK_LAST("reconstruct_kernel"); // catch launch + execution errors

    // (4) Bring the reconstructed image back to the host vector.
    CUDA_CHECK(cudaMemcpy(image.data(), d_image, img_cells * sizeof(float),
                          cudaMemcpyDeviceToHost));

    // (5) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_filtered));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_image));
}
