// ===========================================================================
// src/kernels.cu  --  GPU Demons: force + separable-Gaussian stencils + loop
// ---------------------------------------------------------------------------
// Project 4.8 : Deformable Image Registration (reduced-scope teaching version)
//
// WHAT THIS FILE DOES
//   The GPU twin of register_cpu(). It defines three kernels -- one per pass of
//   a Demons iteration -- and a host wrapper (register_gpu) that allocates
//   device memory, uploads the images once, runs the P.iters-long iteration
//   loop entirely on the device (ping-ponging the displacement buffers through
//   the two Gaussian passes), and copies the final field back.
//
//   Every per-pixel formula (warp, gradient, Thirion force, Gaussian weights)
//   comes from demons.h and is the SAME code the CPU reference runs, so the two
//   displacement fields agree to floating-point rounding (tolerance in THEORY).
//
// READ THIS AFTER: demons.h, kernels.cuh, reference_cpu.cpp (the serial mirror).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// A 16x16 tile = 256 threads per block over the 2-D image. 256 is a multiple of
// the 32-lane warp and gives the scheduler 8 warps per block to hide the global-
// memory latency of the neighbourhood reads. The 2-D block shape mirrors the 2-D
// image so the thread-to-pixel mapping is trivial (see below).
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// demons_force_kernel  (PASS 1 of a Demons iteration)
//   Launch config (set in register_gpu):
//     block = (TILE, TILE) = 16x16 threads
//     grid  = (ceil(nx/TILE), ceil(ny/TILE)) blocks -> covers every pixel
//   Thread-to-data map: thread (blockIdx,threadIdx) owns pixel
//     x = blockIdx.x*blockDim.x + threadIdx.x,  y = blockIdx.y*blockDim.y + ...
//   Memory: reads F,M and the current ux,uy from global memory; writes ux[i],
//     uy[i]. NO atomics and NO shared memory needed: dm_demons_force reads only
//     u[i] at THIS pixel (plus the images), so each thread updates a distinct
//     output element -> the write is race-free by construction.
// ---------------------------------------------------------------------------
__global__ void demons_force_kernel(const double* __restrict__ F,
                                    const double* __restrict__ M,
                                    double* __restrict__ ux,
                                    double* __restrict__ uy,
                                    DemonsParams P) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's column
    const int y = blockIdx.y * blockDim.y + threadIdx.y;   // this thread's row
    if (x >= P.nx || y >= P.ny) return;                    // guard ragged edges

    const int i = y * P.nx + x;
    double dux, duy;
    // The one true Demons update -- identical to the CPU force pass.
    dm_demons_force(F, M, ux, uy, x, y, P, &dux, &duy);
    ux[i] += dux;   // add the step to this pixel's displacement (in place, safe)
    uy[i] += duy;
}

// ---------------------------------------------------------------------------
// gauss_x_kernel  (PASS 2: horizontal half of the separable Gaussian)
//   Same launch geometry as PASS 1. Each thread reads a (2*radius+1)-wide row
//   window of `src` around its pixel and writes the blurred value to `dst`.
//   src and dst MUST be different buffers (ping-pong) so no thread reads a value
//   another thread is overwriting. This is the classic double-buffered stencil
//   (cf. project 6.04 / 14.02).
// ---------------------------------------------------------------------------
__global__ void gauss_x_kernel(const double* __restrict__ src,
                               double* __restrict__ dst,
                               DemonsParams P) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.nx || y >= P.ny) return;
    dst[y * P.nx + x] = dm_gauss_x(src, x, y, P.nx, P.ny, P.sigma, P.radius);
}

// ---------------------------------------------------------------------------
// gauss_y_kernel  (PASS 3: vertical half of the separable Gaussian)
//   Mirror of gauss_x_kernel along the other axis. After PASS 2 then PASS 3,
//   the displacement component has been convolved with a full 2-D Gaussian in
//   O(radius) work per pixel instead of O(radius^2) -- the reason we separate.
// ---------------------------------------------------------------------------
__global__ void gauss_y_kernel(const double* __restrict__ src,
                               double* __restrict__ dst,
                               DemonsParams P) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.nx || y >= P.ny) return;
    dst[y * P.nx + x] = dm_gauss_y(src, x, y, P.nx, P.ny, P.sigma, P.radius);
}

// ---------------------------------------------------------------------------
// register_gpu: host wrapper -- the five canonical CUDA steps, wrapped around
//   the Demons iteration loop.
//     (1) allocate device buffers (images + two displacement fields, ping-pong)
//     (2) copy F, M host->device (once)
//     (3) loop P.iters times: force -> smooth-x -> smooth-y (kernels)
//     (4) copy the final displacement field device->host
//     (5) free device memory
//   We CUDA-event-time only step (3) so the reported figure is the solver's
//   compute cost, not the one-time PCIe transfers (discussed in THEORY).
//
//   BUFFER BOOKKEEPING (why two buffers per component):
//     d_ux / d_uy       hold the "current" displacement field.
//     d_ux2 / d_uy2      are scratch for the separable Gaussian.
//   Per iteration:
//     force writes into d_ux/d_uy (in place);
//     smooth-x reads d_ux -> writes d_ux2  (and d_uy -> d_uy2);
//     smooth-y reads d_ux2 -> writes d_ux  (and d_uy2 -> d_uy).
//   So after the Y pass the smoothed field is back in d_ux/d_uy, ready for the
//   next force pass. No host<->device traffic inside the loop -- the whole
//   solver stays resident on the GPU, which is the point of GPU DIR.
// ---------------------------------------------------------------------------
void register_gpu(const DirImages& im, const DemonsParams& P,
                  std::vector<double>& ux, std::vector<double>& uy,
                  float* kernel_ms) {
    const std::size_t N     = static_cast<std::size_t>(im.nx) * im.ny;
    const std::size_t bytes = N * sizeof(double);

    // (1) Device buffers. d_ = DEVICE pointer (CLAUDE.md §12); dereferencing one
    //     on the host would crash, so the naming convention is load-bearing.
    double *d_F = nullptr, *d_M = nullptr;
    double *d_ux = nullptr, *d_uy = nullptr;     // current displacement field
    double *d_ux2 = nullptr, *d_uy2 = nullptr;   // Gaussian scratch (ping-pong)
    CUDA_CHECK(cudaMalloc(&d_F,   bytes));       // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_M,   bytes));
    CUDA_CHECK(cudaMalloc(&d_ux,  bytes));
    CUDA_CHECK(cudaMalloc(&d_uy,  bytes));
    CUDA_CHECK(cudaMalloc(&d_ux2, bytes));
    CUDA_CHECK(cudaMalloc(&d_uy2, bytes));

    // (2) Upload the images once. The displacement field starts at ZERO (the
    //     identity map); cudaMemset(...,0,...) sets every byte to 0, and an
    //     all-zero-bytes IEEE-754 double is exactly +0.0, so this is a correct
    //     way to zero a double array.
    CUDA_CHECK(cudaMemcpy(d_F, im.fixed.data(),  bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_M, im.moving.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_ux, 0, bytes));
    CUDA_CHECK(cudaMemset(d_uy, 0, bytes));

    // Launch geometry: a 2-D grid of 16x16 blocks covering the image. Ceiling
    // division rounds up so partial edge tiles still get a block (the kernels
    // guard the out-of-range threads inside those tiles).
    dim3 block(TILE, TILE);
    dim3 grid((im.nx + TILE - 1) / TILE, (im.ny + TILE - 1) / TILE);

    // (3) The Demons iteration loop, timed as one unit.
    GpuTimer timer;
    timer.start();
    for (int it = 0; it < P.iters; ++it) {
        // PASS 1 -- force: adds du to the current field in place.
        demons_force_kernel<<<grid, block>>>(d_F, d_M, d_ux, d_uy, P);
        // PASS 2 -- smooth along x: d_ux -> d_ux2, d_uy -> d_uy2.
        gauss_x_kernel<<<grid, block>>>(d_ux, d_ux2, P);
        gauss_x_kernel<<<grid, block>>>(d_uy, d_uy2, P);
        // PASS 3 -- smooth along y: d_ux2 -> d_ux, d_uy2 -> d_uy (back home).
        gauss_y_kernel<<<grid, block>>>(d_ux2, d_ux, P);
        gauss_y_kernel<<<grid, block>>>(d_uy2, d_uy, P);
    }
    *kernel_ms = timer.stop_ms();          // GPU-measured loop time
    CUDA_CHECK_LAST("demons iteration");   // catch any launch/execution error

    // (4) Bring the final displacement field back to the host.
    ux.resize(N);
    uy.resize(N);
    CUDA_CHECK(cudaMemcpy(ux.data(), d_ux, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(uy.data(), d_uy, bytes, cudaMemcpyDeviceToHost));

    // (5) Free everything (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_F));
    CUDA_CHECK(cudaFree(d_M));
    CUDA_CHECK(cudaFree(d_ux));
    CUDA_CHECK(cudaFree(d_uy));
    CUDA_CHECK(cudaFree(d_ux2));
    CUDA_CHECK(cudaFree(d_uy2));
}
