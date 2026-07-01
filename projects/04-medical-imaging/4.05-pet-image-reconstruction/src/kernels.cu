// ===========================================================================
// src/kernels.cu  --  PET projection kernels + the on-device MLEM loop
// ---------------------------------------------------------------------------
// Project 4.5 : PET Image Reconstruction (MLEM / OS-EM)
//
// WHAT THIS FILE DOES
//   Implements the two device kernels (forward_project_kernel, update_kernel), a
//   tiny helper kernel (ratio_kernel), and the host driver mlem_gpu() that keeps
//   all the reconstruction state on the GPU and loops the MLEM iteration. This is
//   the GPU twin of mlem_cpu() in reference_cpu.cpp; main.cu runs both and
//   compares the final images.
//
//   The math lives in pet_geometry.h and is shared with the CPU reference, so the
//   two implementations differ only in HOW they parallelize, not in WHAT they
//   compute (docs/PATTERNS.md §2). Both are GATHERS -> no atomics -> the demo's
//   stdout is deterministic (docs/PATTERNS.md §3).
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea) and
// pet_geometry.h (the shared projection math).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <cstddef>

// Threads per block for the 1-D LOR grid. 256 is a solid sm_75..sm_89 default:
// a multiple of the 32-lane warp, 8 warps to hide latency, many resident blocks.
static constexpr int LOR_THREADS = 256;
// 16x16 = 256-thread square tiles for the 2-D image grid (matches the N x N
// image the way 4.01's backprojection does).
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// forward_project_kernel: y_hat = A x, ONE THREAD PER LOR (k,j).
//   Launch config (set in mlem_gpu):
//     grid  = ceil(K*D / LOR_THREADS) blocks, block = LOR_THREADS threads
//     thread linear index i in [0, K*D) -> LOR (k = i/D, j = i%D)
//   The thread sweeps EVERY pixel and gathers the ones whose linear split lands
//   in its bin j at angle k. Recall (pet_geometry.h) a pixel at fractional bin
//   fidx contributes weight (1-w) to bin j0=floor(fidx) and w to bin j0+1. So LOR
//   bin j collects:
//       from pixels with j0   == j    -> weight (1 - w)
//       from pixels with j0+1 == j    -> weight  w        (i.e. j0 == j-1)
//   Summing both cases over all pixels reproduces exactly what the CPU's pixel-
//   driven scatter wrote into bin j -- same triples, so same value (to rounding).
//   Cost: O(N^2) per LOR thread. Fine for the teaching sample; a production
//   projector would restrict the pixel sweep to the ray's neighborhood (Siddon).
//   Memory: reads image + trig from global; writes one y_hat element. No atomics.
// ---------------------------------------------------------------------------
__global__ void forward_project_kernel(PetGeom g,
                                       const float* __restrict__ image,
                                       const float* __restrict__ cosv,
                                       const float* __restrict__ sinv,
                                       float* __restrict__ yhat) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;  // linear LOR index
    if (i >= g.K * g.D) return;                            // guard ragged block
    const int k = i / g.D;                                 // angle index
    const int j = i - k * g.D;                             // detector bin index
    const float cos_k = cosv[k];
    const float sin_k = sinv[k];

    double acc = 0.0;   // double accumulator: a faithful sum over many pixels
    for (int py = 0; py < g.N; ++py) {
        const float wy = pixel_world_y(g, py);
        for (int px = 0; px < g.N; ++px) {
            const float xv = image[static_cast<std::size_t>(py) * g.N + px];
            if (xv == 0.0f) continue;                      // empty pixel: skip
            const float wx = pixel_world_x(g, px);
            const float fidx = detector_fidx(g, wx, wy, cos_k, sin_k);
            int j0; float w;
            if (!split_bin(g, fidx, j0, w)) continue;      // ray leaves detector
            // This pixel deposits into bins j0 and j0+1. Add whichever equals j.
            if (j0     == j) acc += static_cast<double>(xv) * (1.0 - w);
            if (j0 + 1 == j) acc += static_cast<double>(xv) * w;
        }
    }
    yhat[i] = static_cast<float>(acc);
}

// ---------------------------------------------------------------------------
// ratio_kernel: element-wise ratio_i = y_i / y_hat_i, ONE THREAD PER LOR.
//   Trivially parallel (like SAXPY). Guarded so a zero forward projection gives
//   ratio 0 (a bin with no modeled counts contributes nothing to the update).
//   This mirrors step (2) of mlem_cpu exactly.
// ---------------------------------------------------------------------------
__global__ void ratio_kernel(int n_lor,
                             const float* __restrict__ counts,
                             const float* __restrict__ yhat,
                             float* __restrict__ ratio) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_lor) return;
    ratio[i] = (yhat[i] > 0.0f) ? (counts[i] / yhat[i]) : 0.0f;
}

// ---------------------------------------------------------------------------
// update_kernel: back-project the ratio and apply the MLEM update, ONE THREAD
//   PER PIXEL (px,py) on a 2-D tile grid (same mapping as 4.01's backprojection).
//   The thread gathers, over all K angles, the interpolated ratio on the LORs
//   through this pixel -- using the SAME split weights forward projection used,
//   so this is the exact transpose A^T. Then it applies, in place:
//       image[j] <- image[j] * corr[j] / sens[j]      (guarded when sens==0)
//   which is steps (3)+(4) of mlem_cpu fused into one pass. Independent pixel
//   outputs -> no atomics, deterministic.
// ---------------------------------------------------------------------------
__global__ void update_kernel(PetGeom g,
                              const float* __restrict__ ratio,
                              const float* __restrict__ cosv,
                              const float* __restrict__ sinv,
                              const float* __restrict__ sens,
                              float* __restrict__ image) {
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= g.N || py >= g.N) return;                    // guard edge tiles

    const float wx = pixel_world_x(g, px);
    const float wy = pixel_world_y(g, py);

    double acc = 0.0;   // back-projected ratio for this pixel (corr_j)
    for (int k = 0; k < g.K; ++k) {
        const float fidx = detector_fidx(g, wx, wy, cosv[k], sinv[k]);
        int j0; float w;
        if (!split_bin(g, fidx, j0, w)) continue;
        const std::size_t base = static_cast<std::size_t>(k) * g.D + j0;
        acc += static_cast<double>(ratio[base])     * (1.0 - w)
             + static_cast<double>(ratio[base + 1]) * w;
    }

    const std::size_t idx = static_cast<std::size_t>(py) * g.N + px;
    const float s = sens[idx];
    // Multiplicative update; freeze pixels no LOR sees (s == 0) to avoid 0/0.
    if (s > 0.0f) image[idx] = image[idx] * static_cast<float>(acc) / s;
}

// ---------------------------------------------------------------------------
// mlem_gpu: the host driver. Keeps ALL reconstruction state on the device across
//   iterations (only the final image is copied back), so the per-iteration cost
//   is pure compute -- the whole point of GPU MLEM. Steps:
//     upload geometry-independent buffers once (trig, counts, sensitivity),
//     init image to uniform 1, then loop `iters` times:
//        forward_project_kernel -> ratio_kernel -> update_kernel.
//   We time the SUM of all kernel launches with CUDA events (transfers excluded;
//   they happen once, outside the loop).
// ---------------------------------------------------------------------------
void mlem_gpu(const PetProblem& p, const std::vector<float>& sens, int iters,
              std::vector<float>& image, float* kernel_ms) {
    const PetGeom& g = p.geom;
    const std::size_t n_pix = static_cast<std::size_t>(g.N) * g.N;
    const std::size_t n_lor = static_cast<std::size_t>(g.K) * g.D;

    // ---- Device buffers ----------------------------------------------------
    float *d_image = nullptr, *d_yhat = nullptr, *d_ratio = nullptr;
    float *d_counts = nullptr, *d_sens = nullptr, *d_cos = nullptr, *d_sin = nullptr;
    CUDA_CHECK(cudaMalloc(&d_image,  n_pix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_yhat,   n_lor * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ratio,  n_lor * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_counts, n_lor * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sens,   n_pix * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cos,    static_cast<std::size_t>(g.K) * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin,    static_cast<std::size_t>(g.K) * sizeof(float)));

    // ---- One-time uploads --------------------------------------------------
    CUDA_CHECK(cudaMemcpy(d_counts, p.counts.data(), n_lor * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sens,   sens.data(),      n_pix * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos,    p.cosv.data(),    static_cast<std::size_t>(g.K) * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin,    p.sinv.data(),    static_cast<std::size_t>(g.K) * sizeof(float), cudaMemcpyHostToDevice));

    // Initial estimate x^0 = 1 everywhere (MLEM must start strictly positive).
    // cudaMemset writes BYTES, not floats, so we fill on the host and copy.
    std::vector<float> ones_img(n_pix, 1.0f);
    CUDA_CHECK(cudaMemcpy(d_image, ones_img.data(), n_pix * sizeof(float), cudaMemcpyHostToDevice));

    // ---- Launch geometry ---------------------------------------------------
    const int lor_blocks = (static_cast<int>(n_lor) + LOR_THREADS - 1) / LOR_THREADS;
    dim3 img_block(TILE, TILE);
    dim3 img_grid((g.N + TILE - 1) / TILE, (g.N + TILE - 1) / TILE);

    // ---- Iterate -----------------------------------------------------------
    GpuTimer timer;
    timer.start();
    for (int it = 0; it < iters; ++it) {
        forward_project_kernel<<<lor_blocks, LOR_THREADS>>>(g, d_image, d_cos, d_sin, d_yhat);
        ratio_kernel<<<lor_blocks, LOR_THREADS>>>(static_cast<int>(n_lor), d_counts, d_yhat, d_ratio);
        update_kernel<<<img_grid, img_block>>>(g, d_ratio, d_cos, d_sin, d_sens, d_image);
    }
    *kernel_ms = timer.stop_ms();
    // One check after the loop is enough to surface any launch/exec error; the
    // event-sync inside stop_ms() already forced the GPU to finish.
    CUDA_CHECK_LAST("mlem iteration kernels");

    // ---- Result + cleanup --------------------------------------------------
    image.assign(n_pix, 0.0f);
    CUDA_CHECK(cudaMemcpy(image.data(), d_image, n_pix * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_image));
    CUDA_CHECK(cudaFree(d_yhat));
    CUDA_CHECK(cudaFree(d_ratio));
    CUDA_CHECK(cudaFree(d_counts));
    CUDA_CHECK(cudaFree(d_sens));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));
}
