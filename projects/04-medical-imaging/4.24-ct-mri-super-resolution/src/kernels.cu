// ===========================================================================
// src/kernels.cu  --  GPU super-resolution kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 4.24 : CT/MRI Super-Resolution   (reduced-scope teaching version)
//
// This is the GPU twin of super_resolve_cpu(): identical math (via sr_core.h),
// but one thread per HR output pixel instead of a serial double loop. main.cu
// runs both and checks they agree.  See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "sr_core.h"                 // sr_hr_pixel (the __host__ __device__ core)
#include "util/cuda_check.cuh"       // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"            // GpuTimer

#include <cstdio>
#include <cstdlib>

// ---------------------------------------------------------------------------
// c_weights: the network weights in CONSTANT memory.
//   WHY CONSTANT MEMORY: every thread reads the SAME weights (a few hundred
//   floats) and never writes them. Constant memory has a broadcast cache -- when
//   all threads in a warp read the same address, it costs one fetch, not 32.
//   That is the ideal store for read-only, uniformly-accessed parameters like
//   conv weights (same reasoning as the query in 1.12 / the filter in 7.10).
//   SrWeights is a POD struct, so it copies into constant memory as a blob.
// ---------------------------------------------------------------------------
__constant__ SrWeights c_weights;

// ---------------------------------------------------------------------------
// sr_kernel: compute ONE high-res output pixel per thread.
//   Launch configuration (set by super_resolve_gpu):
//     grid  : ceil(HW/16) x ceil(HH/16) blocks covering the HR image.
//     block : 16 x 16 = 256 threads (SR_BLOCK_X x SR_BLOCK_Y).
//   Thread-to-data map:
//     thread (blockIdx, threadIdx) -> HR pixel
//        hx = blockIdx.x*blockDim.x + threadIdx.x
//        hy = blockIdx.y*blockDim.y + threadIdx.y
//   Memory spaces touched:
//     * d_lr   : GLOBAL memory, read-only (the LR image). __restrict__ + const
//                let the compiler cache reads through the read-only data path.
//     * c_weights : CONSTANT memory (broadcast).
//     * d_hr   : GLOBAL memory, write-only (one store per thread -> coalesced
//                across a warp because consecutive threads write consecutive hx).
//   No shared memory, no atomics, no __syncthreads: pure independent gather.
//
//   All the arithmetic is delegated to sr_hr_pixel() in sr_core.h, so this
//   kernel and the CPU reference are guaranteed to compute the same value.
// ---------------------------------------------------------------------------
__global__ void sr_kernel(const float* __restrict__ d_lr, int lw, int lh,
                          int hw, int hh, float* __restrict__ d_hr) {
    const int hx = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's HR col
    const int hy = blockIdx.y * blockDim.y + threadIdx.y;   // this thread's HR row
    if (hx >= hw || hy >= hh) return;   // guard the ragged edge blocks

    // One call does the whole per-pixel forward pass (feature conv + ReLU, then
    // the pixel-shuffle-selected sub-pixel conv). Same code path as the CPU.
    const float v = sr_hr_pixel(d_lr, lw, lh, hx, hy, c_weights);
    d_hr[(size_t)hy * hw + hx] = v;     // coalesced write
}

// ---------------------------------------------------------------------------
// super_resolve_gpu: upload -> launch -> download. Declared in kernels.cuh.
// ---------------------------------------------------------------------------
void super_resolve_gpu(const Image& lr, const SrWeights& W, int scale,
                       Image& out, float* kernel_ms) {
    // This teaching kernel is compiled for a fixed R = SR_SCALE (the constant is
    // baked into sr_core.h's index math). Refuse mismatches loudly.
    if (scale != SR_SCALE) {
        std::fprintf(stderr, "[super_resolve_gpu] scale=%d but SR_SCALE=%d "
                             "(rebuild with matching SR_SCALE)\n", scale, SR_SCALE);
        std::exit(EXIT_FAILURE);
    }

    const int lw = lr.w, lh = lr.h;         // low-res dimensions
    const int hw = lw * scale, hh = lh * scale;  // high-res dimensions
    out.w = hw; out.h = hh;
    out.pix.assign(static_cast<size_t>(hw) * hh, 0.0f);

    const size_t lr_bytes = static_cast<size_t>(lw) * lh * sizeof(float);
    const size_t hr_bytes = static_cast<size_t>(hw) * hh * sizeof(float);

    // --- Upload the weights into constant memory (once per call). -----------
    // cudaMemcpyToSymbol copies host bytes into the __constant__ symbol; from
    // then on every thread reads them through the broadcast cache.
    CUDA_CHECK(cudaMemcpyToSymbol(c_weights, &W, sizeof(SrWeights)));

    // --- Device buffers for the LR input and HR output. ---------------------
    float* d_lr = nullptr;
    float* d_hr = nullptr;
    CUDA_CHECK(cudaMalloc(&d_lr, lr_bytes));   // read-only input
    CUDA_CHECK(cudaMalloc(&d_hr, hr_bytes));   // write-only output
    CUDA_CHECK(cudaMemcpy(d_lr, lr.pix.data(), lr_bytes, cudaMemcpyHostToDevice));

    // --- Launch: 2-D grid of 16x16 blocks tiling the HR image. --------------
    const dim3 block(SR_BLOCK_X, SR_BLOCK_Y);
    const dim3 grid((hw + block.x - 1) / block.x,
                    (hh + block.y - 1) / block.y);

    GpuTimer timer;
    timer.start();
    sr_kernel<<<grid, block>>>(d_lr, lw, lh, hw, hh, d_hr);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time
    CUDA_CHECK_LAST("sr_kernel");          // catch launch + execution errors

    // --- Copy the HR result back to the host. -------------------------------
    CUDA_CHECK(cudaMemcpy(out.pix.data(), d_hr, hr_bytes, cudaMemcpyDeviceToHost));

    // --- Free device memory (teaching code: explicit, paired with the mallocs).
    CUDA_CHECK(cudaFree(d_lr));
    CUDA_CHECK(cudaFree(d_hr));
}
