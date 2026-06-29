// ===========================================================================
// src/kernels.cu  --  Per-pixel backprojection kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 4.01 : CT Reconstruction (Filtered Backprojection)
//
// GPU twin of backproject_cpu(): same math, one thread per output pixel. The
// 2-D thread grid maps naturally onto the 2-D image. main.cu runs both and
// checks they agree. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

// 16x16 = 256 threads/block: a square tile that matches the 2-D image and gives
// good occupancy on sm_75..sm_89.
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// backproject_kernel: thread (px, py) owns image pixel (px, py).
//   It walks every projection angle, finds the detector position its ray
//   crosses (s = wx*cos + wy*sin), linearly interpolates the filtered
//   projection there, and accumulates. Each pixel is independent -> no shared
//   memory or atomics; the loop is the same one the CPU reference runs.
// ---------------------------------------------------------------------------
__global__ void backproject_kernel(const float* __restrict__ filtered,
                                   const float* __restrict__ cosv,
                                   const float* __restrict__ sinv,
                                   int n_angles, int n_det, int N,
                                   float ds, float center, float W, float pix, float scale,
                                   float* __restrict__ image) {
    const int px = blockIdx.x * blockDim.x + threadIdx.x;
    const int py = blockIdx.y * blockDim.y + threadIdx.y;
    if (px >= N || py >= N) return;               // guard the ragged edge tiles

    const float wx = -W + px * pix;               // world coords of this pixel
    const float wy = -W + py * pix;

    float acc = 0.0f;
    for (int k = 0; k < n_angles; ++k) {
        const float s = wx * cosv[k] + wy * sinv[k];
        const float fidx = s / ds + center;       // fractional detector index
        const int j0 = (int)floorf(fidx);
        if (j0 >= 0 && j0 + 1 < n_det) {
            const float w = fidx - j0;             // linear interpolation weight
            const float* row = filtered + (size_t)k * n_det;
            acc += row[j0] * (1.0f - w) + row[j0 + 1] * w;
        }
    }
    image[(size_t)py * N + px] = acc * scale;      // scale = d(theta) = pi/n_angles
}

// ---------------------------------------------------------------------------
// backproject_gpu: upload inputs, launch the 2-D grid, copy the image back.
// ---------------------------------------------------------------------------
void backproject_gpu(const CTProblem& ct, const std::vector<float>& filtered,
                     const std::vector<float>& cosv, const std::vector<float>& sinv,
                     std::vector<float>& image, float* kernel_ms) {
    const int N = ct.img, n_det = ct.n_det, n_angles = ct.n_angles;
    const std::size_t img_cells = static_cast<std::size_t>(N) * N;
    image.assign(img_cells, 0.0f);

    float *d_filtered = nullptr, *d_cos = nullptr, *d_sin = nullptr, *d_image = nullptr;
    CUDA_CHECK(cudaMalloc(&d_filtered, filtered.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_cos, cosv.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_sin, sinv.size() * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_image, img_cells * sizeof(float)));
    CUDA_CHECK(cudaMemcpy(d_filtered, filtered.data(), filtered.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_cos, cosv.data(), cosv.size() * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_sin, sinv.data(), sinv.size() * sizeof(float), cudaMemcpyHostToDevice));

    const float center = 0.5f * (n_det - 1);
    const float W = ct.world_half;
    const float pix = (N > 1) ? (2.0f * W / (N - 1)) : 0.0f;
    const float scale = 3.14159265358979323846f / n_angles;

    // 2-D grid of TILE x TILE blocks covering the N x N image.
    dim3 block(TILE, TILE);
    dim3 grid((N + TILE - 1) / TILE, (N + TILE - 1) / TILE);
    GpuTimer timer;
    timer.start();
    backproject_kernel<<<grid, block>>>(d_filtered, d_cos, d_sin, n_angles, n_det, N,
                                        ct.ds, center, W, pix, scale, d_image);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("backproject_kernel");

    CUDA_CHECK(cudaMemcpy(image.data(), d_image, img_cells * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_filtered));
    CUDA_CHECK(cudaFree(d_cos));
    CUDA_CHECK(cudaFree(d_sin));
    CUDA_CHECK(cudaFree(d_image));
}
