// ===========================================================================
// src/kernels.cu  --  Per-pixel Delay-and-Sum kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 4.6 : Ultrasound Beamforming (Delay-and-Sum)
//
// GPU twin of beamform_cpu(): same math (das_pixel from beamform.h), one thread
// per output pixel. The 2-D thread grid maps naturally onto the 2-D (x,z) image.
// main.cu runs both and checks they agree. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

// 16x16 = 256 threads/block: a square tile that matches the 2-D image and gives
// good occupancy on sm_75..sm_89. The image's lateral (x) dimension is the fast
// axis, so threadIdx.x striding over ix keeps neighbouring threads writing
// neighbouring image cells -> coalesced stores.
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// das_kernel: thread (ix, iz) owns image pixel (ix, iz).
//   It calls das_pixel() -- the shared __host__ __device__ core -- which loops
//   over every element, computes that element's round-trip focal delay to this
//   pixel, linearly interpolates the element's RF trace at the delay, and sums.
//   Each pixel is independent: no shared memory, no atomics, no inter-thread
//   communication. The loop is byte-for-byte the one the CPU reference runs.
//
//   grid  : ceil(nx/TILE) x ceil(nz/TILE) blocks
//   block : TILE x TILE threads
//   thread(blockIdx, threadIdx) -> pixel (ix = bx*TILE+tx, iz = by*TILE+ty)
// ---------------------------------------------------------------------------
__global__ void das_kernel(BeamformGeom g,
                           const float* __restrict__ rf,
                           float* __restrict__ img) {
    const int ix = blockIdx.x * blockDim.x + threadIdx.x;   // lateral index
    const int iz = blockIdx.y * blockDim.y + threadIdx.y;   // depth index
    if (ix >= g.nx || iz >= g.nz) return;       // guard the ragged edge tiles

    // The entire per-pixel computation is in beamform.h, identical to the CPU
    // path. We store the SIGNED coherent sum; main.cu takes |.| for the B-mode
    // envelope after verifying GPU==CPU on the raw sums.
    img[(std::size_t)iz * g.nx + ix] = das_pixel(g, rf, ix, iz);
}

// ---------------------------------------------------------------------------
// beamform_gpu: upload RF data, launch the 2-D grid, copy the image back.
//   We pass the small BeamformGeom struct BY VALUE in the launch -- it rides in
//   kernel-parameter (constant) space, so every thread reads the geometry with
//   no global-memory traffic. Only the bulky RF array goes through cudaMalloc.
// ---------------------------------------------------------------------------
void beamform_gpu(const BeamformProblem& p, std::vector<float>& image,
                  float* kernel_ms) {
    const BeamformGeom& g = p.geom;
    const std::size_t rf_n  = p.rf.size();                       // elements*samples
    const std::size_t img_n = static_cast<std::size_t>(g.nx) * g.nz;
    image.assign(img_n, 0.0f);

    // ---- Device buffers --------------------------------------------------
    float* d_rf  = nullptr;   // [n_elements*n_samples] RF data (read-only on GPU)
    float* d_img = nullptr;   // [nx*nz] output image
    CUDA_CHECK(cudaMalloc(&d_rf,  rf_n  * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_img, img_n * sizeof(float)));
    // Upload the RF echoes host->device. This is the only large H2D transfer.
    CUDA_CHECK(cudaMemcpy(d_rf, p.rf.data(), rf_n * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ---- Launch ----------------------------------------------------------
    // 2-D grid of TILE x TILE blocks covering the nx-by-nz image. dim3's unused
    // z-component defaults to 1, so this is a flat 2-D launch.
    dim3 block(TILE, TILE);
    dim3 grid((g.nx + TILE - 1) / TILE, (g.nz + TILE - 1) / TILE);

    GpuTimer timer;            // CUDA-event timer (util/timer.cuh)
    timer.start();
    das_kernel<<<grid, block>>>(g, d_rf, d_img);
    *kernel_ms = timer.stop_ms();          // blocks until the kernel finishes
    CUDA_CHECK_LAST("das_kernel");         // catch launch + execution errors

    // ---- Copy result back + free ----------------------------------------
    CUDA_CHECK(cudaMemcpy(image.data(), d_img, img_n * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_rf));
    CUDA_CHECK(cudaFree(d_img));
}
