// ===========================================================================
// src/kernels.cu  --  Per-pixel volume ray-casting kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 4.31 : Virtual Colonoscopy & CT Colonography
//
// GPU twin of render_cpu(): same math, one thread per output pixel. The 2-D
// thread grid maps naturally onto the 2-D image. main.cu runs both and checks
// they agree to FP32 rounding. The per-ray work (trilinear sampling, gradient,
// Phong, march) is the SHARED cast_ray() from volume_render.h -- so there is no
// "GPU version of the math", only a GPU LAUNCH of the one true math.
// See ../THEORY.md "The GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "volume_render.h"        // VolumeView, Vec3, cast_ray (HD: usable on device)
#include "util/cuda_check.cuh"    // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"         // GpuTimer

// 16x16 = 256 threads/block: a square tile that matches the 2-D image and gives
// good occupancy on sm_75..sm_89 (same default as the 4.01 backprojection
// flagship -- a square image deserves a square block).
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// render_kernel: thread (px,py) owns output pixel (px,py).
//   It rebuilds that pixel's ray with pixel_ray() and shades it with cast_ray()
//   -- the identical pair render_cpu() loops on the host. Each pixel is
//   independent: no shared memory, no atomics, no inter-thread communication.
//   This is the canonical volume-rendering kernel.
//
//   We pass the Camera and the volume scalars BY VALUE (they are tiny PODs), and
//   the volume DATA by device pointer. The VolumeView is rebuilt on-device from
//   those scalars so the kernel signature stays small and the math sees exactly
//   the same parameters as the CPU path.
//
//   grid  : ceil(W/TILE) x ceil(H/TILE) blocks covering the image
//   block : TILE x TILE threads
//   thread (blockIdx,threadIdx) -> pixel (px,py) -> image[py*W + px]
// ---------------------------------------------------------------------------
__global__ void render_kernel(const float* __restrict__ vol,
                              int nx, int ny, int nz,
                              float iso, float step, int max_steps,
                              Camera cam, int W, int H,
                              float* __restrict__ image) {
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= W || py >= H) return;     // guard the ragged edge tiles

    // Rebuild the VolumeView on-device (same fields as scene.view() on the host).
    VolumeView V;
    V.data = vol; V.nx = nx; V.ny = ny; V.nz = nz;
    V.iso = iso; V.step = step; V.max_steps = max_steps;

    Vec3 origin, dir;
    pixel_ray(cam, px, py, W, H, origin, dir);   // this pixel's ray
    float shade = cast_ray(V, origin, dir);      // march + Phong shade
    image[(size_t)py * W + px] = shade;          // one independent write
}

// ---------------------------------------------------------------------------
// render_gpu: upload the volume, launch the 2-D grid, copy the image back.
//   Mirrors backproject_gpu() in the 4.01 flagship: malloc -> H2D copy -> launch
//   (timed with CUDA events) -> D2H copy -> free. Every CUDA call is wrapped in
//   CUDA_CHECK so a failure is reported at its source line, not swallowed.
// ---------------------------------------------------------------------------
void render_gpu(const Scene& scene, std::vector<float>& image, float* kernel_ms) {
    const int W = scene.width, H = scene.height;
    const size_t n_vox = (size_t)scene.nx * scene.ny * scene.nz;
    const size_t n_pix = (size_t)W * H;
    image.assign(n_pix, 0.0f);

    // ---- Device allocations -------------------------------------------------
    float *d_vol = nullptr, *d_img = nullptr;
    CUDA_CHECK(cudaMalloc(&d_vol, n_vox * sizeof(float)));   // the CT volume
    CUDA_CHECK(cudaMalloc(&d_img, n_pix * sizeof(float)));   // the output frame

    // ---- Upload the volume (the only large H2D transfer) --------------------
    // In a real fly-through this upload happens ONCE and many frames are rendered
    // from the resident volume; here we render a single frame for clarity.
    CUDA_CHECK(cudaMemcpy(d_vol, scene.vol.data(), n_vox * sizeof(float),
                          cudaMemcpyHostToDevice));

    // ---- Launch the 2-D grid (one thread per pixel) -------------------------
    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);
    GpuTimer timer;
    timer.start();
    render_kernel<<<grid, block>>>(d_vol, scene.nx, scene.ny, scene.nz,
                                   scene.iso, scene.step, scene.max_steps,
                                   scene.cam, W, H, d_img);
    *kernel_ms = timer.stop_ms();           // blocks until the kernel finishes
    CUDA_CHECK_LAST("render_kernel");       // catch launch + execution errors

    // ---- Copy the rendered image back, then free ----------------------------
    CUDA_CHECK(cudaMemcpy(image.data(), d_img, n_pix * sizeof(float),
                          cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_vol));
    CUDA_CHECK(cudaFree(d_img));
}
