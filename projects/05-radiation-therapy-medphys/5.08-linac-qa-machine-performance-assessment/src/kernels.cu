// ===========================================================================
// src/kernels.cu  --  Per-measured-pixel gamma-index kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 5.8 : Linac QA & Machine Performance Assessment  (catalog ID 5.8)
//
// GPU twin of gamma_map_cpu(): the identical math (gamma_value_at from gamma.h),
// one thread per measured pixel, laid out as a 2-D grid over the 2-D dose plane.
// main.cu runs both the CPU and GPU paths and asserts the maps are bit-identical
// (they call the SAME __host__ __device__ function, so the tolerance is 0).
// See ../THEORY.md "GPU mapping" for the occupancy / memory-layout reasoning.
// ===========================================================================
#include "kernels.cuh"
#include "gamma.h"               // gamma_value_at (shared host/device core)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

// 16x16 = 256 threads/block: a square tile that maps naturally onto the 2-D
// dose plane and gives good occupancy on sm_75..sm_89. A power-of-two square
// also keeps the ragged-edge guard simple.
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// gamma_kernel: thread (mx, my) owns measured pixel (mx, my).
//   Each thread evaluates gamma_value_at() for its own pixel -- scanning the
//   local (2R+1)^2 window of the REFERENCE plane and taking the min combined
//   dose/distance disagreement. Pixels are fully independent: no shared memory,
//   no atomics, no inter-thread communication. The window reads of `ref` are the
//   only "gather"; because neighbouring threads read heavily-overlapping windows,
//   the L1/L2 cache (and, on real hardware, texture memory -- see THEORY §"real
//   world") serves most of those reads. We keep plain global loads here because
//   they are the clearest teaching form and the cache already does the work.
//
//   grid  : ceil(nx/TILE) x ceil(ny/TILE) blocks
//   block : TILE x TILE threads
//   map   : output gamma map, row-major [ny*nx]
// ---------------------------------------------------------------------------
__global__ void gamma_kernel(const float* __restrict__ meas,
                             const float* __restrict__ ref,
                             GammaParams p,
                             float* __restrict__ map) {
    const int mx = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's column
    const int my = blockIdx.y * blockDim.y + threadIdx.y;  // this thread's row
    if (mx >= p.nx || my >= p.ny) return;                  // guard ragged edges

    // One call into the shared core -- IDENTICAL arithmetic to the CPU reference.
    map[(size_t)my * p.nx + mx] = gamma_value_at(meas, ref, mx, my, p);
}

// ---------------------------------------------------------------------------
// gamma_map_gpu: upload the two planes, launch the 2-D grid, copy the map back.
//   Mirrors backproject_gpu() in the CT flagship (4.01): a thin, fully error-
//   checked host wrapper whose only job is device plumbing. Kernel time is
//   measured with CUDA events (util/timer.cuh) and returned via *kernel_ms.
// ---------------------------------------------------------------------------
void gamma_map_gpu(const QAProblem& q, const GammaParams& p,
                   std::vector<float>& gamma_out, float* kernel_ms) {
    const int nx = q.nx, ny = q.ny;
    const std::size_t n = static_cast<std::size_t>(nx) * ny;   // pixels per plane
    gamma_out.assign(n, 0.0f);

    // --- Device buffers: two input planes + one output map. ----------------
    float *d_meas = nullptr, *d_ref = nullptr, *d_map = nullptr;
    CUDA_CHECK(cudaMalloc(&d_meas, n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_ref,  n * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_map,  n * sizeof(float)));

    // H2D copies. cudaMemcpy can fail on a bad size/pointer -> CUDA_CHECK guards.
    CUDA_CHECK(cudaMemcpy(d_meas, q.meas.data(), n * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ref,  q.ref.data(),  n * sizeof(float), cudaMemcpyHostToDevice));

    // --- Launch config: a 2-D grid of TILE x TILE blocks over the plane. ---
    dim3 block(TILE, TILE);
    dim3 grid((nx + TILE - 1) / TILE, (ny + TILE - 1) / TILE);

    GpuTimer timer;
    timer.start();
    gamma_kernel<<<grid, block>>>(d_meas, d_ref, p, d_map);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("gamma_kernel");   // catch launch-config + in-kernel errors

    // D2H copy of the finished map, then free every device allocation.
    CUDA_CHECK(cudaMemcpy(gamma_out.data(), d_map, n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_meas));
    CUDA_CHECK(cudaFree(d_ref));
    CUDA_CHECK(cudaFree(d_map));
}
