// ===========================================================================
// src/kernels.cu  --  Per-pixel DRR ray-marching kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 4.28 : GPU-Accelerated DRR Generation for 2D/3D Registration
//
// GPU twin of render_drr_cpu(): same math (it calls the SAME integrate_ray() from
// drr_core.h), but one thread per detector pixel laid out as a 2-D grid over the
// detector panel. main.cu runs both and checks they agree. See ../THEORY.md
// "GPU mapping" for the thread/block reasoning and the texture-memory upgrade.
// ===========================================================================
#include "kernels.cuh"
#include "drr_core.h"             // integrate_ray, sample_trilinear (HD core)
#include "util/cuda_check.cuh"    // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"         // GpuTimer (CUDA-event stopwatch)

// 16x16 = 256 threads/block: a square tile that matches the 2-D detector panel
// and gives good occupancy on sm_75..sm_89. A square tile also keeps neighbouring
// threads' rays spatially close, so their tri-linear samples hit nearby voxels --
// good for the L1/L2 cache (and ideal for a 3-D texture in the production version).
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// drr_kernel: thread (u, vrow) owns detector pixel (u, vrow).
//   The thread-to-data mapping is the canonical 2-D one:
//       u    = blockIdx.x * blockDim.x + threadIdx.x   (detector column)
//       vrow = blockIdx.y * blockDim.y + threadIdx.y   (detector row)
//   Threads outside the panel (ragged edge tiles) return immediately. Each
//   surviving thread calls integrate_ray() -- the exact function the CPU
//   reference calls -- and writes its single result. No shared memory, no
//   atomics: a pure independent gather.
// ---------------------------------------------------------------------------
__global__ void drr_kernel(const float* __restrict__ d_mu,
                           VolumeDesc v, DrrGeometry g,
                           float* __restrict__ d_img) {
    const int u    = blockIdx.x * blockDim.x + threadIdx.x;   // detector column
    const int vrow = blockIdx.y * blockDim.y + threadIdx.y;   // detector row
    if (u >= g.width || vrow >= g.height) return;             // guard ragged tiles

    // integrate_ray() marches this pixel's ray through the volume and returns the
    // accumulated attenuation (the DRR pixel). Identical to the CPU path.
    float pixel = integrate_ray(d_mu, v, g, u, vrow);

    // Row-major [v][u] output: matches render_drr_cpu so verification is direct.
    d_img[(size_t)vrow * g.width + u] = pixel;
}

// ---------------------------------------------------------------------------
// render_drr_gpu: upload the volume, launch the 2-D grid, copy the image back.
//   Memory traffic:
//     * UP   : the whole attenuation volume (nx*ny*nz floats) -- the big transfer.
//     * DOWN : the rendered image (width*height floats) -- small.
//   In a real registration loop the volume is uploaded ONCE and hundreds of DRRs
//   are rendered from it at different poses, so this upload amortizes away -- the
//   per-iteration cost is just the kernel. We time only the kernel (CUDA events)
//   to reflect that steady-state cost; copies are excluded (and noted in main.cu).
// ---------------------------------------------------------------------------
void render_drr_gpu(const CtVolume& vol, const DrrGeometry& g,
                    std::vector<float>& image, float* kernel_ms) {
    const int W = g.width, H = g.height;
    const size_t n_vox = static_cast<size_t>(vol.mu.size());   // total voxels
    const size_t n_pix = static_cast<size_t>(W) * H;           // total DRR pixels
    image.assign(n_pix, 0.0f);

    // --- device buffers ---
    float* d_mu  = nullptr;   // attenuation volume on the device
    float* d_img = nullptr;   // rendered DRR on the device
    CUDA_CHECK(cudaMalloc(&d_mu,  n_vox * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_img, n_pix * sizeof(float)));

    // Upload the volume (H2D). This is the one large copy; see the note above.
    CUDA_CHECK(cudaMemcpy(d_mu, vol.mu.data(), n_vox * sizeof(float),
                          cudaMemcpyHostToDevice));

    // --- launch: 2-D grid of TILE x TILE blocks covering the W x H panel ---
    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);

    GpuTimer timer;          // CUDA-event timer: measures GPU time on the stream
    timer.start();
    // VolumeDesc and DrrGeometry are small PODs passed BY VALUE; CUDA copies them
    // into the kernel's parameter space (constant-bank backed), so every thread
    // reads the geometry cheaply without a global-memory fetch.
    drr_kernel<<<grid, block>>>(d_mu, vol.desc, g, d_img);
    *kernel_ms = timer.stop_ms();          // ms spent in the kernel itself
    CUDA_CHECK_LAST("drr_kernel");         // catch launch/exec errors (bad config, etc.)

    // Copy the rendered DRR back (D2H) and free device memory.
    CUDA_CHECK(cudaMemcpy(image.data(), d_img, n_pix * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_mu));
    CUDA_CHECK(cudaFree(d_img));
}
