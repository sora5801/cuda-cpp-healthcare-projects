// ===========================================================================
// src/kernels.cu  --  GPU oART kernels + host-driven Demons loop
// ---------------------------------------------------------------------------
// Project 5.14 : GPU-Accelerated Adaptive MR-Linac Workflow (reduced-scope)
//
// WHAT THIS FILE DOES
//   The GPU twin of oart_cpu(). Four small kernels, each a per-voxel operation,
//   and a host wrapper (oart_gpu) that runs the Demons iteration entirely on the
//   device (no per-iteration copies back to the host). Each kernel body is just
//   "map this thread to a voxel, then call the shared physics from
//   mrl_registration.h" -- so the GPU and CPU compute identical arithmetic.
//
//   KERNELS
//     grad_kernel        : precompute the fixed-image gradient (gfx,gfy) once.
//     warp_kernel        : backward-warp an image by (u,v) into an output (GATHER).
//     demons_add_kernel  : add each voxel's demons force to (u,v)          (STENCIL).
//     smooth_axis_kernel : one axis of a separable Gaussian convolution.
//
//   All four are the "grid of 2-D threads over a 2-D image" mapping: thread
//   (bx*Tx+tx, by*Ty+ty) owns voxel (x,y). No atomics anywhere: every kernel
//   writes each output voxel exactly once from one thread, so there are no races
//   and the result is deterministic (PATTERNS.md section 3).
//
// READ THIS AFTER: kernels.cuh (the pattern), mrl_registration.h (the physics).
// ===========================================================================
#include "kernels.cuh"
#include "mrl_registration.h"     // shared per-voxel physics (host+device)
#include "util/cuda_check.cuh"    // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"         // GpuTimer (CUDA-event timing)

#include <vector>

// 16x16 = 256 threads/block: a multiple of the 32-lane warp, good occupancy on
// sm_75..sm_89, and a natural 2-D tile over a 2-D image (rows stay coalesced).
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// grad_kernel: thread (x,y) writes the fixed-image gradient at its voxel.
//   Reads a small 3x3-ish neighbourhood of `fixed` (central differences) -> a
//   STENCIL read. The gradient is constant across all Demons iterations, so we
//   compute it once up front and keep it resident.
//     grid  : covers the nx-by-ny image; block : 16x16.
//     thread (x,y) -> gfx[y*nx+x], gfy[y*nx+x].
// ---------------------------------------------------------------------------
__global__ void grad_kernel(const double* __restrict__ fixed, int nx, int ny,
                            double* __restrict__ gfx, double* __restrict__ gfy) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;                 // guard ragged edge tiles
    const std::size_t i = flat_idx(x, y, nx);
    gfx[i] = grad_x(fixed, nx, ny, x, y);           // shared central difference
    gfy[i] = grad_y(fixed, nx, ny, x, y);
}

// ---------------------------------------------------------------------------
// warp_kernel: thread (x,y) backward-warps `src` by (u,v) into `dst`.
//   dst(x,y) = src sampled (bilinear) at (x+u, y+v). A pure GATHER: each output
//   voxel reads independently, no writes collide -> no atomics, no races. This is
//   the same idiom as CT backprojection (4.01): compute a source coordinate, then
//   interpolate a read.
// ---------------------------------------------------------------------------
__global__ void warp_kernel(const double* __restrict__ src, int nx, int ny,
                            const double* __restrict__ u, const double* __restrict__ v,
                            double* __restrict__ dst) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const std::size_t i = flat_idx(x, y, nx);
    const double fx = x + u[i];                      // sub-voxel source location
    const double fy = y + v[i];
    dst[i] = sample_bilinear(src, nx, ny, fx, fy);   // shared bilinear gather
}

// ---------------------------------------------------------------------------
// demons_add_kernel: thread (x,y) adds its demons force to the field (u,v).
//   Reads the already-warped moving image, the fixed image, and the precomputed
//   fixed gradient at its own voxel, computes (du,dv) via the shared
//   demons_force(), and accumulates into u,v. Each thread touches only its own
//   (u[i],v[i]) -> no atomics.
// ---------------------------------------------------------------------------
__global__ void demons_add_kernel(const double* __restrict__ warped,
                                  const double* __restrict__ fixed,
                                  const double* __restrict__ gfx,
                                  const double* __restrict__ gfy,
                                  int nx, int ny, double k_norm,
                                  double* __restrict__ u, double* __restrict__ v) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    const std::size_t i = flat_idx(x, y, nx);
    double du, dv;
    demons_force(warped[i], fixed[i], gfx[i], gfy[i], k_norm, &du, &dv);  // shared
    u[i] += du;
    v[i] += dv;
}

// ---------------------------------------------------------------------------
// smooth_axis_kernel: one pass of a separable Gaussian convolution.
//   `axis` selects the pass: 0 = horizontal (offset x), 1 = vertical (offset y).
//   Weights `w` (length 2*radius+1) live in global memory (uploaded once). Each
//   thread reads a 1-D window along the axis (clamped at borders) and writes one
//   output voxel -> a bounded STENCIL, no atomics. Two launches (axis 0 then 1)
//   realise the full 2-D Gaussian, matching the CPU's two-pass smoother exactly.
// ---------------------------------------------------------------------------
__global__ void smooth_axis_kernel(const double* __restrict__ in, int nx, int ny,
                                   const double* __restrict__ w, int radius, int axis,
                                   double* __restrict__ out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    double acc = 0.0;
    for (int t = -radius; t <= radius; ++t) {
        int xs = x, ys = y;
        if (axis == 0) xs = clampi(x + t, nx);       // horizontal window
        else           ys = clampi(y + t, ny);       // vertical window
        acc += w[t + radius] * in[flat_idx(xs, ys, nx)];
    }
    out[flat_idx(x, y, nx)] = acc;
}

// ---------------------------------------------------------------------------
// smooth_separable_gpu (host helper): run the two axis passes for one field,
//   ping-ponging between `field` and a scratch buffer. After this returns, the
//   smoothed result is back in `field`. Both passes are timed by the caller.
// ---------------------------------------------------------------------------
static void smooth_separable_gpu(double* field, double* scratch, int nx, int ny,
                                 const double* d_w, int radius,
                                 dim3 grid, dim3 block) {
    // Horizontal pass: field -> scratch.
    smooth_axis_kernel<<<grid, block>>>(field, nx, ny, d_w, radius, 0, scratch);
    // Vertical pass: scratch -> field (so the result ends up back in `field`).
    smooth_axis_kernel<<<grid, block>>>(scratch, nx, ny, d_w, radius, 1, field);
}

// ---------------------------------------------------------------------------
// oart_gpu: the full workflow on the device (see kernels.cuh for the contract).
// ---------------------------------------------------------------------------
void oart_gpu(const OartCase& c, OartResult& r, float* kernel_ms) {
    const std::size_t n = static_cast<std::size_t>(c.nx) * c.ny;
    const std::size_t bytes = n * sizeof(double);

    // Build the (host) Gaussian weights, then upload them once. Reusing the CPU's
    // gaussian_kernel_1d guarantees identical coefficients on both paths.
    int radius; std::vector<double> w;
    gaussian_kernel_1d(c.sigma, radius, w);
    const std::size_t wbytes = w.size() * sizeof(double);

    // --- (1) Device buffers. d_ prefix = DEVICE pointer (never deref on host). ---
    double *d_fixed=nullptr, *d_moving=nullptr, *d_dose=nullptr;
    double *d_gfx=nullptr, *d_gfy=nullptr;
    double *d_u=nullptr, *d_v=nullptr, *d_scratch=nullptr;
    double *d_warped=nullptr, *d_w=nullptr;
    CUDA_CHECK(cudaMalloc(&d_fixed,  bytes));
    CUDA_CHECK(cudaMalloc(&d_moving, bytes));
    CUDA_CHECK(cudaMalloc(&d_dose,   bytes));
    CUDA_CHECK(cudaMalloc(&d_gfx,    bytes));
    CUDA_CHECK(cudaMalloc(&d_gfy,    bytes));
    CUDA_CHECK(cudaMalloc(&d_u,      bytes));
    CUDA_CHECK(cudaMalloc(&d_v,      bytes));
    CUDA_CHECK(cudaMalloc(&d_scratch,bytes));
    CUDA_CHECK(cudaMalloc(&d_warped, bytes));
    CUDA_CHECK(cudaMalloc(&d_w,      wbytes));

    // --- (2) Upload inputs; zero the displacement field (identity transform). ---
    CUDA_CHECK(cudaMemcpy(d_fixed,  c.fixed.data(),  bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_moving, c.moving.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_dose,   c.dose.data(),   bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w,      w.data(),        wbytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_u, 0, bytes));      // u = 0 everywhere
    CUDA_CHECK(cudaMemset(d_v, 0, bytes));      // v = 0 everywhere

    // Launch geometry: a 2-D grid of 16x16 tiles covering the image.
    dim3 block(TILE, TILE);
    dim3 grid((c.nx + TILE - 1) / TILE, (c.ny + TILE - 1) / TILE);

    // --- (3) Time the whole compute (registration + dose warp) with events. ---
    GpuTimer timer;
    timer.start();

    // Precompute the fixed-image gradient once.
    grad_kernel<<<grid, block>>>(d_fixed, c.nx, c.ny, d_gfx, d_gfy);

    // Demons iteration: warp -> add force -> smooth field, all on the device.
    for (int it = 0; it < c.iters; ++it) {
        warp_kernel<<<grid, block>>>(d_moving, c.nx, c.ny, d_u, d_v, d_warped);
        demons_add_kernel<<<grid, block>>>(d_warped, d_fixed, d_gfx, d_gfy,
                                           c.nx, c.ny, c.k_norm, d_u, d_v);
        smooth_separable_gpu(d_u, d_scratch, c.nx, c.ny, d_w, radius, grid, block);
        smooth_separable_gpu(d_v, d_scratch, c.nx, c.ny, d_w, radius, grid, block);
    }

    // Final warped moving image (for the after-MSE check) and warped dose.
    warp_kernel<<<grid, block>>>(d_moving, c.nx, c.ny, d_u, d_v, d_warped);
    // Reuse d_scratch as the warped-dose output buffer.
    warp_kernel<<<grid, block>>>(d_dose, c.nx, c.ny, d_u, d_v, d_scratch);

    *kernel_ms = timer.stop_ms();               // GPU-measured compute time
    CUDA_CHECK_LAST("oart kernels");            // catch launch + execution errors

    // --- (4) Copy results back to the host. ---
    r.u.assign(n, 0.0); r.v.assign(n, 0.0);
    r.warped_moving.assign(n, 0.0); r.warped_dose.assign(n, 0.0);
    CUDA_CHECK(cudaMemcpy(r.u.data(),            d_u,       bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.v.data(),            d_v,       bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.warped_moving.data(),d_warped,  bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(r.warped_dose.data(),  d_scratch, bytes, cudaMemcpyDeviceToHost));

    // Metrics use the SAME host routine as the CPU path (identical arithmetic).
    compute_metrics(c, r);

    // --- (5) Free every device allocation (no GPU garbage collector). ---
    CUDA_CHECK(cudaFree(d_fixed));  CUDA_CHECK(cudaFree(d_moving));
    CUDA_CHECK(cudaFree(d_dose));   CUDA_CHECK(cudaFree(d_gfx));
    CUDA_CHECK(cudaFree(d_gfy));    CUDA_CHECK(cudaFree(d_u));
    CUDA_CHECK(cudaFree(d_v));      CUDA_CHECK(cudaFree(d_scratch));
    CUDA_CHECK(cudaFree(d_warped)); CUDA_CHECK(cudaFree(d_w));
}
