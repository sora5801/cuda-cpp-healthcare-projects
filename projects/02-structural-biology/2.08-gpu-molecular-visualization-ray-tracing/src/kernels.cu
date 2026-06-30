// ===========================================================================
// src/kernels.cu  --  Per-pixel ray-tracing kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 2.8 : GPU Molecular Visualization & Ray Tracing
//
// WHAT THIS FILE DOES
//   The GPU twin of render_cpu(). A 2-D thread grid covers the image; each
//   thread renders one pixel by calling the SHARED shade_pixel() from
//   render_core.h. The scene (atoms) is uploaded to CONSTANT memory because
//   every thread reads every atom and the data is read-only for the launch --
//   exactly what constant memory's broadcast cache is built for.
//
//   main.cu runs both render_cpu() and render_gpu() and verifies the byte
//   images are identical (see ../THEORY.md "How we verify correctness").
//
// READ THIS AFTER: kernels.cuh (the idea), render_core.h (the shared physics).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <stdexcept>             // std::runtime_error

// ---------------------------------------------------------------------------
// THE SCENE IN CONSTANT MEMORY.
//   c_atoms is a fixed-size device array in __constant__ space (64 KB total on
//   the GPU). The host fills it once with cudaMemcpyToSymbol before the launch;
//   thereafter every thread reads it through the constant cache. Because a warp
//   of 32 threads casting the same primary ray order reads the SAME atom at the
//   same step, the hardware broadcasts that read in one transaction -- cheap and
//   bandwidth-free. (For scenes larger than MAX_ATOMS you would instead pass a
//   global pointer + use a spatial acceleration structure; see THEORY §real-world.)
// ---------------------------------------------------------------------------
__constant__ Atom c_atoms[MAX_ATOMS];

// 16x16 = 256 threads per block: a square tile that matches the 2-D image and
// gives good occupancy on sm_75..sm_89. The same choice as flagship 4.01.
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// render_kernel: thread (px,py) owns image pixel (px,py).
//   Thread-to-data map (CLAUDE.md 6.1 rule 2):
//     px = blockIdx.x*blockDim.x + threadIdx.x
//     py = blockIdx.y*blockDim.y + threadIdx.y
//   Memory: reads the scene from __constant__ c_atoms (broadcast), writes one
//   byte to global `image`. No shared memory, no atomics -- pixels are
//   independent. All the actual rendering is shade_pixel() from render_core.h,
//   the identical function the CPU reference calls, which is why the results
//   match. quantize8() maps the float luminance to the stored 0..255 byte.
// ---------------------------------------------------------------------------
__global__ void render_kernel(Camera cam, int n_atoms, RenderParams rp,
                              unsigned char* __restrict__ image) {
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= cam.width || py >= cam.height) return;   // guard ragged edge tiles

    // Render this pixel with the shared physics, reading atoms from constant
    // memory. shade_pixel takes a const Atom* -- c_atoms decays to exactly that.
    const float luma = shade_pixel(cam, px, py, c_atoms, n_atoms, rp);

    // Store the quantized luminance. Row-major layout matches the CPU reference.
    image[(size_t)py * cam.width + px] = quantize8(luma);
}

// ---------------------------------------------------------------------------
// render_gpu: host wrapper. Canonical CUDA steps, with the scene in constant
//   memory instead of a malloc'd buffer:
//     (1) sanity-check the scene fits constant memory; upload it to c_atoms.
//     (2) allocate the output byte image on the device.
//     (3) launch the 2-D grid (timed with CUDA events).
//     (4) copy the rendered image back to the host.
//     (5) free the device image.
//   We time ONLY the kernel (step 3), not the copies -- the reported ms is the
//   render cost, a teaching artifact (CLAUDE.md §12), never a benchmark claim.
// ---------------------------------------------------------------------------
void render_gpu(const Scene& scene, std::vector<unsigned char>& image,
                float* kernel_ms) {
    const int W = scene.cam.width, H = scene.cam.height;
    const int n = static_cast<int>(scene.atoms.size());
    if (n > MAX_ATOMS)
        throw std::runtime_error("scene exceeds MAX_ATOMS for constant memory "
                                 "(raise MAX_ATOMS or use a global-memory path)");

    const std::size_t img_bytes = static_cast<std::size_t>(W) * H;  // 1 byte/pixel
    image.assign(img_bytes, 0);

    // (1) Upload the atom list into the __constant__ symbol. cudaMemcpyToSymbol
    //     writes a host buffer into a named device symbol; we copy exactly n
    //     atoms (the rest of c_atoms is never read because we pass n_atoms=n).
    CUDA_CHECK(cudaMemcpyToSymbol(c_atoms, scene.atoms.data(),
                                  static_cast<std::size_t>(n) * sizeof(Atom)));

    // (2) Output image on the device (one byte per pixel).
    unsigned char* d_image = nullptr;
    CUDA_CHECK(cudaMalloc(&d_image, img_bytes));

    // (3) Launch a 2-D grid of TILE x TILE blocks covering the W x H image.
    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);
    GpuTimer timer;
    timer.start();
    render_kernel<<<grid, block>>>(scene.cam, n, scene.rp, d_image);
    *kernel_ms = timer.stop_ms();           // GPU-measured kernel time
    CUDA_CHECK_LAST("render_kernel");       // catch launch + execution errors

    // (4) Copy the rendered image back to the host vector.
    CUDA_CHECK(cudaMemcpy(image.data(), d_image, img_bytes, cudaMemcpyDeviceToHost));

    // (5) Free the device image (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_image));
}
