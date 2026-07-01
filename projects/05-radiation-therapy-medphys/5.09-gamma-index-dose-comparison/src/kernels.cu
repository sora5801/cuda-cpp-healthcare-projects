// ===========================================================================
// src/kernels.cu  --  Per-reference-voxel gamma kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 5.9 -- Gamma-Index Dose Comparison
//
// WHAT THIS FILE DOES
//   Implements the device kernel (gamma_kernel) and the host-side glue
//   (gamma_map_gpu) that allocates GPU memory, moves the two dose maps over,
//   launches the kernel, times it, and brings the gamma map back. This is the
//   GPU twin of gamma_map_cpu() in reference_cpu.cpp; main.cu runs both and
//   compares them.
//
//   The per-pair arithmetic (gamma_sq_term) is shared with the CPU via
//   gamma_core.h, so the GPU reproduces the CPU's minima EXACTLY -- the kernel
//   only adds the thread decomposition around that identical math.
//
// READ THIS AFTER: kernels.cuh (interface + thread-mapping idea), gamma_core.h
// (the shared physics). See ../THEORY.md §4-§6.
// ===========================================================================
#include "kernels.cuh"
#include "gamma_core.h"          // gamma_sq_term(), GammaParams (shared w/ CPU)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <algorithm>             // std::max on the host side
#include <cmath>                 // std::ceil on the host side

// 16x16 = 256 threads/block: a square tile that matches the 2-D reference grid
// and gives good occupancy on sm_75..sm_89 (8 warps/block to hide the memory
// latency of the gather). Mirrors the tiling choice in flagship 4.01.
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// gamma_kernel: thread (gx, gy) owns reference voxel (rx=gx, ry=gy).
//
//   Launch config (set in gamma_map_gpu):
//     block = (TILE, TILE)                        = 256 threads
//     grid  = (ceil(W/TILE), ceil(H/TILE))        covers the W x H voxel grid
//   Thread-to-data map: rx = blockIdx.x*blockDim.x + threadIdx.x, similarly ry.
//
//   Memory spaces (THEORY §4):
//     * d_eval / d_ref : GLOBAL memory, read-only here (marked __restrict__ +
//       const so the compiler may route them through the read-only data cache).
//     * best_sq        : a REGISTER -- the running minimum lives per-thread; no
//       shared memory or atomics are needed because each thread owns exactly one
//       output voxel and never writes another thread's result. (Contrast the
//       Monte-Carlo flagship 5.01, which DOES need atomics because many threads
//       tally into shared bins.)
//
//   Determinism: this is a MIN reduction, and min is associative AND exact in
//   floating point (unlike a sum), so scanning the window in the same fixed
//   row-major order as the CPU yields a bit-identical result -- no reordering
//   caveat (THEORY §5).
// ---------------------------------------------------------------------------
__global__ void gamma_kernel(const float* __restrict__ d_ref,   // [W*H] reference dose
                             const float* __restrict__ d_eval,  // [W*H] evaluated dose
                             int W, int H,
                             double spacing_mm,                 // voxel edge [mm]
                             int radius_vox,                    // search half-width [voxels]
                             double dose_thresh,                // low-dose cutoff [dose]
                             GammaParams gp,                    // inverse-sq criteria
                             float* __restrict__ d_gamma) {     // [W*H] output gamma
    // Which reference voxel this thread is responsible for.
    const int rx = blockIdx.x * blockDim.x + threadIdx.x;
    const int ry = blockIdx.y * blockDim.y + threadIdx.y;
    if (rx >= W || ry >= H) return;          // guard the ragged edge tiles

    const int    ridx     = ry * W + rx;
    const double dose_ref = d_ref[ridx];

    // Below-threshold background: not analyzed, gamma = 0 (matches the CPU).
    if (dose_ref < dose_thresh) { d_gamma[ridx] = 0.0f; return; }

    // Running minimum of gamma^2 over the search window, held in a register.
    double best_sq = 1.0e30;                 // "infinity" for our purposes

    // Clamp the search window to the grid (identical bounds to the CPU).
    const int ex0 = max(0,     rx - radius_vox);
    const int ex1 = min(W - 1, rx + radius_vox);
    const int ey0 = max(0,     ry - radius_vox);
    const int ey1 = min(H - 1, ry + radius_vox);

    // Gather + min-reduce over evaluated voxels in fixed row-major order. Each
    // thread re-reads the (small) window from global memory; because adjacent
    // threads in a warp scan overlapping windows, those loads hit the L2/read-
    // only cache, so a naive gather is already fast for small windows. Shared-
    // memory tiling of the evaluated tile (the catalog's suggested optimization)
    // is left as an exercise -- see THEORY §4 and README "Exercises".
    for (int ey = ey0; ey <= ey1; ++ey) {
        for (int ex = ex0; ex <= ex1; ++ex) {
            const double dx = (ex - rx) * spacing_mm;   // [mm]
            const double dy = (ey - ry) * spacing_mm;   // [mm]
            const double dist_sq = dx * dx + dy * dy;   // [mm^2]

            const double dose_eval = d_eval[ey * W + ex];
            const double term = gamma_sq_term(dose_eval, dose_ref, dist_sq, gp);
            if (term < best_sq) best_sq = term;         // exact float min
        }
    }

    // One sqrt at the end -> the gamma index at this reference voxel.
    d_gamma[ridx] = (float)sqrt(best_sq);
}

// ---------------------------------------------------------------------------
// map_max_host -- largest dose in a map (host side), to normalize the criteria.
//   Kept here (not in gamma_core.h) because it touches std::vector, which we
//   deliberately keep out of the __host__ __device__ core.
// ---------------------------------------------------------------------------
static double map_max_host(const std::vector<float>& m) {
    double mx = 0.0;
    for (float v : m) mx = std::max(mx, static_cast<double>(v));
    return mx;
}

// ---------------------------------------------------------------------------
// gamma_map_gpu: host wrapper. The five canonical steps of a CUDA computation:
//   (1) allocate device memory  (2) copy inputs host->device
//   (3) launch the kernel        (4) copy result device->host
//   (5) free device memory
// We time ONLY step (3) with CUDA events so the reported figure is the kernel
// cost, not the PCIe transfer cost (discussed separately in THEORY §7).
//
// CRITICAL: every derived quantity (dd_crit, dta_crit, radius_vox, dose_thresh)
// is computed with the SAME formulas the CPU uses in reference_cpu.cpp. If these
// diverged, the two candidate sets or normalizers would differ and the exact
// CPU==GPU check would (correctly) fail -- so keep them in lockstep.
// ---------------------------------------------------------------------------
void gamma_map_gpu(const DoseProblem& prob, std::vector<float>& gamma_out,
                   float* kernel_ms) {
    const int W = prob.width;
    const int H = prob.height;
    const std::size_t n = static_cast<std::size_t>(prob.size());
    gamma_out.assign(n, 0.0f);

    // --- Derive the same criteria the CPU reference derives -----------------
    const double ref_max  = map_max_host(prob.ref);
    const double dd_crit  = prob.dd_percent * 0.01 * ref_max;   // [dose]
    const double dta_crit = prob.dta_mm;                        // [mm]
    GammaParams gp;
    gp.inv_dd_crit_sq  = 1.0 / (dd_crit  * dd_crit);
    gp.inv_dta_crit_sq = 1.0 / (dta_crit * dta_crit);
    const double search_mm  = 3.0 * dta_crit;
    const int    radius_vox = static_cast<int>(std::ceil(search_mm / prob.spacing_mm));
    const double dose_thresh = prob.dose_threshold_frac * ref_max;

    // (1) Device buffers. The d_ prefix marks DEVICE pointers (CLAUDE.md §12):
    //     dereferencing one on the host would crash, so the naming matters.
    const std::size_t bytes = n * sizeof(float);
    float *d_ref = nullptr, *d_eval = nullptr, *d_gamma = nullptr;
    CUDA_CHECK(cudaMalloc(&d_ref,   bytes));   // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_eval,  bytes));
    CUDA_CHECK(cudaMalloc(&d_gamma, bytes));

    // (2) Copy the two dose maps H2D. .data() is the contiguous backing array.
    CUDA_CHECK(cudaMemcpy(d_ref,  prob.ref.data(),  bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_eval, prob.eval.data(), bytes, cudaMemcpyHostToDevice));

    // (3) Launch a 2-D grid of TILE x TILE blocks covering the W x H voxel grid.
    dim3 block(TILE, TILE);
    dim3 grid((W + TILE - 1) / TILE, (H + TILE - 1) / TILE);
    GpuTimer timer;
    timer.start();
    gamma_kernel<<<grid, block>>>(d_ref, d_eval, W, H, prob.spacing_mm,
                                  radius_vox, dose_thresh, gp, d_gamma);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time
    CUDA_CHECK_LAST("gamma_kernel");       // catch launch + execution errors

    // (4) Bring the gamma map back to the host vector.
    CUDA_CHECK(cudaMemcpy(gamma_out.data(), d_gamma, bytes, cudaMemcpyDeviceToHost));

    // (5) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_ref));
    CUDA_CHECK(cudaFree(d_eval));
    CUDA_CHECK(cudaFree(d_gamma));
}
