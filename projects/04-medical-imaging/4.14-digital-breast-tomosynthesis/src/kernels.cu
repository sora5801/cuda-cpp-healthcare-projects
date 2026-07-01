// ===========================================================================
// src/kernels.cu  --  GPU SART: forward projection + backprojection kernels
// ---------------------------------------------------------------------------
// Project 4.14 : Digital Breast Tomosynthesis
//
// WHAT THIS FILE DOES
//   Implements the two device kernels that make up one SART iteration and the
//   host driver (reconstruct_sart_gpu) that keeps the image + scratch buffers
//   GPU-resident across all iterations and loops the kernels. Every kernel calls
//   the SAME per-ray helpers in dbt_geometry.h that the CPU reference uses, so
//   the GPU and CPU results match (verified in main.cu).
//
//   The residual subtraction (measured - simulated) is a tiny element-wise step;
//   we do it in its own trivial kernel so the whole SART iteration stays on the
//   GPU with zero host round-trips.
//
// GATHER, NOT SCATTER -> NO ATOMICS -> DETERMINISM (docs/PATTERNS.md §3)
//   Both projection kernels write each output element from exactly ONE thread
//   (a gather), so there are no atomic accumulations whose float summation order
//   would vary run-to-run. The GPU result is therefore bit-stable across runs
//   and matches the CPU to float rounding only. That is why main.cu can diff a
//   fixed expected_output.txt.
//
// READ THIS AFTER: kernels.cuh, dbt_geometry.h, util/cuda_check.cuh.
// ===========================================================================
#include "kernels.cuh"
#include "dbt_geometry.h"        // forward_ray_integral, bilinear_sample (shared HD core)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cmath>                 // floorf

// Threads per block for the 1-D forward kernel. 256 is a solid sm_75..sm_89
// default: a multiple of the 32-lane warp, 8 warps to hide the memory latency of
// the ray march, and many blocks resident for occupancy.
static constexpr int FWD_THREADS = 256;

// The backprojection kernel uses a 2-D thread block that matches the 2-D image.
// 16x16 = 256 threads: a square tile with good occupancy and coalesced row
// writes (adjacent threads in x write adjacent image[] elements).
static constexpr int TILE = 16;

// ===========================================================================
// KERNEL 1 -- forward_project_kernel: one thread per detector ray.
// ---------------------------------------------------------------------------
//   Launch config (set in the driver):
//     grid  = ceil((n_angles*n_det) / FWD_THREADS) blocks (1-D)
//     block = FWD_THREADS threads (1-D)
//   Thread-to-data map: global index r = blockIdx.x*blockDim.x + threadIdx.x,
//     then k = r / n_det (angle), j = r % n_det (detector bin).
//   Memory: reads the whole image estimate (via bilinear_sample) along the ray,
//     reads cos/sin tables, writes one sim[r]. No shared memory / atomics.
//   This is the exact GPU twin of forward_project_cpu(): identical math because
//   both call dbt_geometry.h::forward_ray_integral().
// ===========================================================================
__global__ void forward_project_kernel(const float* __restrict__ image,
                                       const float* __restrict__ cosv,
                                       const float* __restrict__ sinv,
                                       int n_angles, int n_det, int N,
                                       float ds, float center, float W, float pix,
                                       int steps, float dt,
                                       float* __restrict__ sim) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's ray index
    const int total = n_angles * n_det;
    if (r >= total) return;                                // guard the ragged last block

    const int k = r / n_det;   // projection angle of this ray
    const int j = r % n_det;   // detector bin of this ray

    const float ck = cosv[k], sk = sinv[k];
    const float s  = (j - center) * ds;                    // signed detector offset (world)
    // Raw sum of bilinear samples along the ray; scale by per-step world length
    // to get the physical line integral (same units as the measured projections).
    const float raw = forward_ray_integral(image, N, ck, sk, s, W, pix, steps);
    sim[r] = raw * dt;
}

// ===========================================================================
// KERNEL 2 -- residual_kernel: sim,meas -> residual = meas - sim (element-wise).
// ---------------------------------------------------------------------------
//   Trivial 1-D map (one thread per projection element) that keeps the residual
//   computation on the GPU so no data leaves the device between SART stages.
// ===========================================================================
__global__ void residual_kernel(const float* __restrict__ meas,
                                 const float* __restrict__ sim,
                                 int total,
                                 float* __restrict__ res) {
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= total) return;
    res[r] = meas[r] - sim[r];     // how far the current estimate's projection is off
}

// ===========================================================================
// KERNEL 3 -- backproject_update_kernel: one thread per output PIXEL.
// ---------------------------------------------------------------------------
//   Launch config (set in the driver):
//     grid  = 2-D, ceil(N/TILE) x ceil(N/TILE) blocks
//     block = TILE x TILE threads
//   Thread-to-data map: px = blockIdx.x*blockDim.x + threadIdx.x,
//                       py = blockIdx.y*blockDim.y + threadIdx.y; pixel (px,py).
//   For its pixel the thread gathers the residual from every angle (projecting
//   the pixel's world position onto each detector, linear-interpolating the
//   residual), averages over angles, applies the relaxed SART correction, and
//   clamps to >= 0. Writes image[py*N+px] in place. No atomics: each pixel is
//   owned by exactly one thread. Exact GPU twin of backproject_update_cpu().
// ===========================================================================
__global__ void backproject_update_kernel(const float* __restrict__ residual,
                                          const float* __restrict__ cosv,
                                          const float* __restrict__ sinv,
                                          int n_angles, int n_det, int N,
                                          float ds, float center, float W, float pix,
                                          float lambda, float inv_na,
                                          float* __restrict__ image) {
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= N || py >= N) return;                        // guard ragged edge tiles

    const float wx = -W + px * pix;                        // world coords of this pixel
    const float wy = -W + py * pix;

    float acc = 0.0f;
    // Gather this pixel's residual contribution from every projection angle.
    for (int k = 0; k < n_angles; ++k) {
        const float sproj = wx * cosv[k] + wy * sinv[k];   // detector offset for this pixel
        const float fidx  = sproj / ds + center;           // fractional detector bin
        const int   j0    = (int)floorf(fidx);
        if (j0 >= 0 && j0 + 1 < n_det) {
            const float w   = fidx - j0;                   // linear interpolation weight
            const float* rr = residual + (size_t)k * n_det;
            acc += rr[j0] * (1.0f - w) + rr[j0 + 1] * w;
        }
    }
    const size_t idx = (size_t)py * N + px;
    float v = image[idx] + lambda * acc * inv_na;          // relaxed, column-normalised update
    if (v < 0.0f) v = 0.0f;                                // physicality: attenuation >= 0
    image[idx] = v;
}

// ===========================================================================
// HOST DRIVER -- reconstruct_sart_gpu
// ---------------------------------------------------------------------------
//   Upload the measured projections and angle tables once, keep the image and
//   two projection-sized scratch buffers device-resident, then loop the three
//   kernels n_iters times. Only the final image is copied back. We time the
//   kernels (CUDA events) accumulated across every launch; H2D/D2H copies are
//   excluded so the figure is compute, not PCIe (discussed in THEORY).
// ===========================================================================
void reconstruct_sart_gpu(const DBTProblem& p,
                          const std::vector<float>& cosv,
                          const std::vector<float>& sinv,
                          std::vector<float>& image,
                          float* kernel_ms) {
    const int    N      = p.img;
    const int    n_det  = p.n_det;
    const int    na     = p.n_angles;
    const int    total  = na * n_det;                       // number of projection elements
    const size_t n_pix  = (size_t)N * N;
    image.assign(n_pix, 0.0f);

    // ---- Derived geometry constants (computed once, passed to kernels) ----
    const float W      = p.world_half;
    const float pix    = (N > 1) ? (2.0f * W / (N - 1)) : 0.0f;   // world units / pixel
    const float center = 0.5f * (n_det - 1);                      // detector index of s=0
    const int   steps  = n_ray_steps(p);                         // samples per ray
    const float L      = 1.41421356f * W;                        // ray half-length
    const float dt     = (steps > 1) ? (2.0f * L / (steps - 1)) : 0.0f;  // world length/step
    const float inv_na = 1.0f / (float)na;                       // column normalisation
    const float lambda = p.relax;                               // SART relaxation

    // ---- Device buffers ----------------------------------------------------
    // d_ prefix = DEVICE pointer (dereferencing on the host would crash).
    float *d_meas = nullptr, *d_sim = nullptr, *d_res = nullptr;
    float *d_cos = nullptr, *d_sin = nullptr, *d_image = nullptr;
    CUDA_CHECK(cudaMalloc(&d_meas,  (size_t)total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sim,   (size_t)total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_res,   (size_t)total * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cos,   (size_t)na    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin,   (size_t)na    * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_image, n_pix         * sizeof(float)));

    // ---- One-time uploads (measured data, angles, zeroed image) -----------
    CUDA_CHECK(cudaMemcpy(d_meas, p.proj.data(), (size_t)total * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos,  cosv.data(),   (size_t)na    * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin,  sinv.data(),   (size_t)na    * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_image, 0, n_pix * sizeof(float)));    // initial estimate: all air

    // ---- Launch geometry ---------------------------------------------------
    const int fwd_blocks = (total + FWD_THREADS - 1) / FWD_THREADS;   // 1-D over rays
    dim3 bp_block(TILE, TILE);
    dim3 bp_grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);       // 2-D over image

    // ---- SART loop (all on device) ----------------------------------------
    // One GpuTimer per phase, accumulated, so *kernel_ms is the total device
    // compute time across every launch of every iteration.
    float total_ms = 0.0f;
    GpuTimer timer;
    for (int it = 0; it < p.n_iters; ++it) {
        // (1) forward-project current estimate -> d_sim.
        timer.start();
        forward_project_kernel<<<fwd_blocks, FWD_THREADS>>>(
            d_image, d_cos, d_sin, na, n_det, N, p.ds, center, W, pix, steps, dt, d_sim);
        total_ms += timer.stop_ms();
        CUDA_CHECK_LAST("forward_project_kernel");

        // (2) residual = measured - simulated -> d_res.
        timer.start();
        residual_kernel<<<fwd_blocks, FWD_THREADS>>>(d_meas, d_sim, total, d_res);
        total_ms += timer.stop_ms();
        CUDA_CHECK_LAST("residual_kernel");

        // (3) backproject residual, relaxed + clamped update of d_image in place.
        timer.start();
        backproject_update_kernel<<<bp_grid, bp_block>>>(
            d_res, d_cos, d_sin, na, n_det, N, p.ds, center, W, pix, lambda, inv_na, d_image);
        total_ms += timer.stop_ms();
        CUDA_CHECK_LAST("backproject_update_kernel");
    }
    *kernel_ms = total_ms;

    // ---- Copy the final reconstruction back, then free device memory ------
    CUDA_CHECK(cudaMemcpy(image.data(), d_image, n_pix * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_meas));
    CUDA_CHECK(cudaFree(d_sim));
    CUDA_CHECK(cudaFree(d_res));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_image));
}
