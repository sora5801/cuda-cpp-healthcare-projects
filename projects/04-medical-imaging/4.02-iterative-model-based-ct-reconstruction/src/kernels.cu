// ===========================================================================
// src/kernels.cu  --  GPU SIRT: forward + backproject/update + TV, in a loop
// ---------------------------------------------------------------------------
// Project 4.2 : Iterative / Model-Based CT Reconstruction
//
// WHAT THIS FILE DOES
//   Implements the four device kernels and the host driver sirt_gpu() that runs
//   the entire iterative reconstruction on the GPU. Each kernel is the device
//   twin of a loop in reference_cpu.cpp, and every kernel calls the SAME shared
//   geometry (ct_geometry.h) as the CPU, so their arithmetic matches to within
//   the tiny FMA/rounding drift documented in main.cu (PATTERNS.md §2, §4).
//
//   Determinism (PATTERNS.md §3): every kernel writes each output from ONE
//   thread that accumulates in a fixed order. No floating-point atomics -> the
//   result (and thus stdout) is byte-identical on every run.
//
// READ THIS AFTER: kernels.cuh (the mapping idea), ct_geometry.h (the physics).
//                  reference_cpu.cpp is the serial twin to compare against.
// ===========================================================================
#include "kernels.cuh"
#include "ct_geometry.h"         // detector_coord, interp_stencil, pixel_world
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// 1-D block size for the element-wise / ray kernels. 256 threads = 8 warps: a
// multiple of the 32-lane warp with enough warps to hide global-memory latency.
static constexpr int THREADS_1D = 256;
// 2-D tile for the image kernels. 16x16 = 256 threads maps squarely onto the
// N x N image and keeps occupancy high on sm_75..sm_89.
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// forward_project_kernel  (A : image -> sinogram)
//   THREAD-TO-DATA MAP: one thread per DETECTOR BIN (ray). Flatten the ray index
//   ray = blockIdx.x*blockDim.x + threadIdx.x into (k, j) = (ray/n_det, ray%n_det).
//   Thread (k,j) computes sino[k,j] by looping over EVERY image pixel in the
//   SAME py-outer/px-inner order the CPU uses, and adding a pixel's contribution
//   iff that pixel's ray at angle k lands on bin j (lower bin -> weight 1-w,
//   upper bin -> weight w). Summing in that fixed order reproduces the CPU's
//   voxel-scatter sum EXACTLY (same terms, same order) -- deterministic, atomics-
//   free, and the exact transpose of the pixel-gather backprojection below.
//   Cost: O(N^2) per ray; fine for a teaching-sized image. (Production ray-driven
//   projectors march only the pixels ON the ray via a DDA; see THEORY §real-world.)
// ---------------------------------------------------------------------------
__global__ void forward_project_kernel(const float* __restrict__ image,
                                       const float* __restrict__ cosv,
                                       const float* __restrict__ sinv,
                                       int n_angles, int n_det, int N,
                                       float ds, float center, float W, float pix,
                                       float* __restrict__ sino_out) {
    const int ray = blockIdx.x * blockDim.x + threadIdx.x;
    const int n_rays = n_angles * n_det;
    if (ray >= n_rays) return;                    // guard the ragged last block

    const int k = ray / n_det;                    // projection angle index
    const int j = ray % n_det;                    // detector bin index
    const float ck = cosv[k], sk = sinv[k];       // this angle's trig (registers)

    float acc = 0.0f;                             // private accumulator (register)
    // Same iteration order as forward_project_cpu: py outer, px inner.
    for (int py = 0; py < N; ++py) {
        const float wy = pixel_world(py, W, pix);
        for (int px = 0; px < N; ++px) {
            const float val = image[(size_t)py * N + px];
            if (val == 0.0f) continue;            // skip empty pixels (matches CPU)
            const float wx   = pixel_world(px, W, pix);
            const float fidx = detector_coord(wx, wy, ck, sk, ds, center);
            int j0; float w;
            if (interp_stencil(fidx, n_det, &j0, &w)) {
                // Add this pixel's share to bin j only (this thread owns bin j).
                if (j0     == j) acc += val * (1.0f - w);
                if (j0 + 1 == j) acc += val * w;
            }
        }
    }
    sino_out[(size_t)k * n_det + j] = acc;        // one write, no race
}

// ---------------------------------------------------------------------------
// residual_kernel: element-wise resid[i] = (b[i] - sim[i]) * row_scale[i].
//   Trivial "one thread per element" map -- the row-normalized data residual
//   R(b - Ax) that SIRT backprojects. Independent per bin, so no sync needed.
// ---------------------------------------------------------------------------
__global__ void residual_kernel(const float* __restrict__ b,
                                const float* __restrict__ sim,
                                const float* __restrict__ row_scale,
                                int n_rays, float* __restrict__ resid) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n_rays) resid[i] = (b[i] - sim[i]) * row_scale[i];
}

// ---------------------------------------------------------------------------
// backproject_update_kernel  (A^T then the SIRT step, fused)
//   THREAD-TO-DATA MAP: 2-D grid over the image; thread (px,py) owns pixel p.
//   It gathers the residual sampled where its ray hits the detector at every
//   angle (the transpose of forward_project_kernel -- same interp_stencil), then
//   applies the column-normalized, relaxed SIRT update WITH non-negativity:
//       image[p] = max(0, image[p] + lambda * col_scale[p] * (A^T resid)[p]).
//   Reading the old image[p] and writing the new one is safe in place because
//   each thread touches only its own pixel (no neighbours) -- unlike the TV step.
// ---------------------------------------------------------------------------
__global__ void backproject_update_kernel(const float* __restrict__ resid,
                                          const float* __restrict__ cosv,
                                          const float* __restrict__ sinv,
                                          const float* __restrict__ col_scale,
                                          int n_angles, int n_det, int N,
                                          float ds, float center, float W, float pix,
                                          float lambda, float* __restrict__ image) {
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= N || py >= N) return;               // guard ragged edge tiles

    const float wx = pixel_world(px, W, pix);
    const float wy = pixel_world(py, W, pix);

    float acc = 0.0f;                             // (A^T resid)[p], summed over k
    for (int k = 0; k < n_angles; ++k) {
        const float fidx = detector_coord(wx, wy, cosv[k], sinv[k], ds, center);
        int j0; float w;
        if (interp_stencil(fidx, n_det, &j0, &w)) {
            const float* row = resid + (size_t)k * n_det;
            acc += row[j0] * (1.0f - w) + row[j0 + 1] * w;
        }
    }
    const size_t p = (size_t)py * N + px;
    float v = image[p] + lambda * col_scale[p] * acc;
    image[p] = (v > 0.0f) ? v : 0.0f;             // densities are non-negative
}

// ---------------------------------------------------------------------------
// tv_step_kernel: one edge-preserving TOTAL-VARIATION descent step.
//   THREAD-TO-DATA MAP: 2-D grid; thread (px,py) reads pixel + 4 neighbours from
//   img_in and writes img_out. Because a thread reads its NEIGHBOURS, we must NOT
//   update in place (a neighbour might already be overwritten). Writing to a
//   SEPARATE buffer (ping-pong / double buffer) removes that race -- the same
//   pattern the stencil flagships 6.04 / 14.02 use. The math is identical to
//   tv_step_cpu(): move each pixel toward its neighbours, damped by 1/|grad| so
//   strong edges barely move.
// ---------------------------------------------------------------------------
__global__ void tv_step_kernel(const float* __restrict__ img_in, int N,
                               float weight, float* __restrict__ img_out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= N || y >= N) return;

    // Clamped (Neumann) neighbour fetch: past the border, reuse the edge pixel.
    auto at = [&](int xx, int yy) -> float {
        if (xx < 0) xx = 0; if (xx >= N) xx = N - 1;
        if (yy < 0) yy = 0; if (yy >= N) yy = N - 1;
        return img_in[(size_t)yy * N + xx];
    };
    const float eps = 1e-3f;
    const float c  = at(x, y);
    const float dl = c - at(x - 1, y);
    const float dr = at(x + 1, y) - c;
    const float du = c - at(x, y - 1);
    const float dd = at(x, y + 1) - c;
    const float gmag = sqrtf(eps * eps + dl * dl + dr * dr + du * du + dd * dd);
    const float lap  = (dr - dl) + (dd - du);     // discrete Laplacian
    img_out[(size_t)y * N + x] = c + weight * lap / gmag;
}

// ---------------------------------------------------------------------------
// sirt_gpu: the host driver. Upload the constants ONCE, then re-launch the
//   kernels ct.iters times. The image (and all scratch buffers) stay RESIDENT on
//   the device across iterations -- only two tiny copies cross PCIe (sinogram in
//   at the start, image out at the end). This "state lives on the GPU, host just
//   orchestrates launches" shape is exactly how production iterative solvers run.
// ---------------------------------------------------------------------------
void sirt_gpu(const CTProblem& ct,
              const std::vector<float>& cosv, const std::vector<float>& sinv,
              const std::vector<float>& row_scale, const std::vector<float>& col_scale,
              std::vector<float>& image, float* kernel_ms) {
    const int   N = ct.img, n_det = ct.n_det, n_angles = ct.n_angles;
    const size_t n_rays = (size_t)n_angles * n_det;
    const size_t n_pix  = (size_t)N * N;

    // Derived geometry scalars (identical to the CPU's make_geom()).
    const float W      = ct.world_half;
    const float pix    = (N > 1) ? (2.0f * W / (N - 1)) : 0.0f;
    const float center = 0.5f * (n_det - 1);

    // --- Allocate all device buffers up front (reused every iteration) -----
    float *d_sino=nullptr, *d_cos=nullptr, *d_sin=nullptr, *d_rowscale=nullptr,
          *d_colscale=nullptr, *d_img=nullptr, *d_img2=nullptr,
          *d_sim=nullptr, *d_resid=nullptr;
    CUDA_CHECK(cudaMalloc(&d_sino,     n_rays * sizeof(float)));   // measured b
    CUDA_CHECK(cudaMalloc(&d_cos,      n_angles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin,      n_angles * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_rowscale, n_rays * sizeof(float)));   // R diagonal
    CUDA_CHECK(cudaMalloc(&d_colscale, n_pix  * sizeof(float)));   // C diagonal
    CUDA_CHECK(cudaMalloc(&d_img,      n_pix  * sizeof(float)));   // x (estimate)
    CUDA_CHECK(cudaMalloc(&d_img2,     n_pix  * sizeof(float)));   // TV ping-pong
    CUDA_CHECK(cudaMalloc(&d_sim,      n_rays * sizeof(float)));   // A x
    CUDA_CHECK(cudaMalloc(&d_resid,    n_rays * sizeof(float)));   // R(b - A x)

    // --- Upload the constants (done once) ----------------------------------
    CUDA_CHECK(cudaMemcpy(d_sino, ct.sino.data(), n_rays * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos,  cosv.data(),  n_angles * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin,  sinv.data(),  n_angles * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_rowscale, row_scale.data(), n_rays * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_colscale, col_scale.data(), n_pix  * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_img, 0, n_pix * sizeof(float)));       // x^0 = 0

    // --- Launch configurations ---------------------------------------------
    const int rays_blocks = (int)((n_rays + THREADS_1D - 1) / THREADS_1D);
    dim3 img_block(TILE, TILE);
    dim3 img_grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);

    // Time the SUM of all kernel launches across all iterations (one figure).
    GpuTimer timer;
    timer.start();
    for (int it = 0; it < ct.iters; ++it) {
        // 1. sim = A x
        forward_project_kernel<<<rays_blocks, THREADS_1D>>>(
            d_img, d_cos, d_sin, n_angles, n_det, N, ct.ds, center, W, pix, d_sim);
        // 2. resid = R .* (b - sim)
        residual_kernel<<<rays_blocks, THREADS_1D>>>(
            d_sino, d_sim, d_rowscale, (int)n_rays, d_resid);
        // 3+4. x = max(0, x + lambda * C .* (A^T resid))
        backproject_update_kernel<<<img_grid, img_block>>>(
            d_resid, d_cos, d_sin, d_colscale, n_angles, n_det, N,
            ct.ds, center, W, pix, ct.lambda, d_img);
        // 5. optional TV step: read d_img, write d_img2, then swap the pointers
        //    so d_img always names the current estimate (ping-pong).
        if (ct.tv_weight > 0.0f) {
            tv_step_kernel<<<img_grid, img_block>>>(d_img, N, ct.tv_weight, d_img2);
            float* tmp = d_img; d_img = d_img2; d_img2 = tmp;
        }
    }
    *kernel_ms = timer.stop_ms();                 // total device time, all iters
    CUDA_CHECK_LAST("sirt iteration kernels");    // catch any launch/exec error

    // --- Copy the final reconstruction back and free everything ------------
    image.assign(n_pix, 0.0f);
    CUDA_CHECK(cudaMemcpy(image.data(), d_img, n_pix * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_sino));  CUDA_CHECK(cudaFree(d_cos));  CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_rowscale)); CUDA_CHECK(cudaFree(d_colscale));
    CUDA_CHECK(cudaFree(d_img));   CUDA_CHECK(cudaFree(d_img2));
    CUDA_CHECK(cudaFree(d_sim));   CUDA_CHECK(cudaFree(d_resid));
}
