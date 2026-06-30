// ===========================================================================
// src/kernels.cu  --  The GPU DCC kernel and its host wrapper
// ---------------------------------------------------------------------------
// Project 2.17 : Allosteric Network Analysis
//
// WHAT THIS FILE DOES
//   Implements dcc_kernel (one thread = one matrix entry) and dcc_matrix_gpu
//   (the host glue: allocate, copy, launch, time, copy back). This is the GPU
//   twin of dcc_matrix_cpu() in reference_cpu.cpp; main.cu runs both and asserts
//   they agree exactly. Because BOTH call dcc_pair() from dcc_core.h, the device
//   thread and the host loop execute the identical sequence of double-precision
//   operations -> the GPU matrix is bit-for-bit equal to the CPU matrix.
//
// READ THIS AFTER: kernels.cuh (the launch idea), dcc_core.h (the per-pair math).
// ===========================================================================
#include "kernels.cuh"
#include "dcc_core.h"            // dcc_pair, coord_index (shared __host__ __device__ math)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// 2-D block geometry. 16x16 = 256 threads/block: a multiple of the 32-lane warp,
// gives the scheduler 8 warps to hide the latency of the O(T) inner sum, and is
// the canonical tile for "one thread per matrix element" launches on sm_75..89.
static constexpr int BLOCK_X = 16;   // threads spanning the COLUMN (j) direction
static constexpr int BLOCK_Y = 16;   // threads spanning the ROW    (i) direction

// ---------------------------------------------------------------------------
// dcc_kernel: each thread computes exactly one C[row][col].
//
//   LAUNCH CONFIG (set in dcc_matrix_gpu):
//     block = (BLOCK_X, BLOCK_Y)              -> 256 threads, a 16x16 tile
//     grid  = (ceil(N/BLOCK_X), ceil(N/BLOCK_Y)) -> enough tiles to cover N x N
//   THREAD -> DATA MAP:
//     col = blockIdx.x*blockDim.x + threadIdx.x   (the residue j / matrix column)
//     row = blockIdx.y*blockDim.y + threadIdx.y   (the residue i / matrix row)
//   MEMORY: reads the whole trajectory and the means from GLOBAL memory; writes
//     ONE float to C[row*N+col] in global memory. No shared memory and no atomics
//     are needed because every thread owns a distinct output element -- there is
//     no overlap to coordinate. (THEORY.md discusses a shared-memory tiling
//     optimization that would cache each residue's track; we keep the naive,
//     readable version because it teaches the mapping cleanly.)
// ---------------------------------------------------------------------------
__global__ void dcc_kernel(const float* __restrict__ coords,
                           const double* __restrict__ mean,
                           int T, int N,
                           float* __restrict__ C) {
    // This thread's matrix coordinates.
    const int col = blockIdx.x * blockDim.x + threadIdx.x;   // residue j
    const int row = blockIdx.y * blockDim.y + threadIdx.y;   // residue i

    // GUARD THE RAGGED EDGE: N is rarely a multiple of 16, so the right/bottom
    // tiles contain threads with row >= N or col >= N. They must do nothing, or
    // they would read residues that do not exist and write outside C.
    if (row >= N || col >= N) return;

    // The entire per-entry computation is delegated to the SHARED physics in
    // dcc_core.h. This is the crux of the project: the device thread runs the
    // exact same dcc_pair() the CPU reference runs, so verification is exact.
    const double c = dcc_pair(coords, mean, row, col, T, N);

    // Store as float (the matrix is consumed downstream as float on both sides).
    C[static_cast<std::size_t>(row) * N + col] = static_cast<float>(c);
}

// ---------------------------------------------------------------------------
// dcc_matrix_gpu: host wrapper -- the five canonical steps of a CUDA computation:
//   (1) allocate device memory  (2) copy inputs host->device
//   (3) launch the kernel        (4) copy result device->host
//   (5) free device memory
// We time ONLY step (3) with CUDA events so the reported figure is the kernel
// cost, not the PCIe transfer cost (transfers are discussed separately in THEORY).
// ---------------------------------------------------------------------------
void dcc_matrix_gpu(const Trajectory& traj, const std::vector<double>& mean,
                    std::vector<float>& C, float* kernel_ms) {
    const int N = traj.N, T = traj.T;
    C.assign(static_cast<std::size_t>(N) * N, 0.0f);

    // Byte sizes of the three device buffers.
    const std::size_t coords_bytes = static_cast<std::size_t>(T) * N * 3 * sizeof(float);
    const std::size_t mean_bytes   = static_cast<std::size_t>(N) * 3 * sizeof(double);
    const std::size_t mat_bytes    = static_cast<std::size_t>(N) * N * sizeof(float);

    // (1) Device buffers. The d_ prefix marks DEVICE pointers (CLAUDE.md 12):
    //     dereferencing one on the host would crash, so the naming matters.
    float*  d_coords = nullptr;   // [T*N*3] trajectory
    double* d_mean   = nullptr;   // [N*3]   per-residue means
    float*  d_C      = nullptr;   // [N*N]   output matrix
    CUDA_CHECK(cudaMalloc(&d_coords, coords_bytes));   // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_mean,   mean_bytes));
    CUDA_CHECK(cudaMalloc(&d_C,      mat_bytes));

    // (2) Copy inputs H2D. .data() is the contiguous backing array of vector.
    CUDA_CHECK(cudaMemcpy(d_coords, traj.coords.data(), coords_bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_mean,   mean.data(),        mean_bytes,   cudaMemcpyHostToDevice));

    // (3) Launch over a 2-D grid that tiles the N x N matrix. The ceiling
    //     division (N + B - 1) / B is integer "round up" so the grid fully
    //     covers the matrix even when N is not a multiple of the block size.
    const dim3 block(BLOCK_X, BLOCK_Y);
    const dim3 grid((N + BLOCK_X - 1) / BLOCK_X, (N + BLOCK_Y - 1) / BLOCK_Y);
    GpuTimer timer;
    timer.start();
    dcc_kernel<<<grid, block>>>(d_coords, d_mean, T, N, d_C);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time
    CUDA_CHECK_LAST("dcc_kernel");         // catch launch + execution errors

    // (4) Bring the matrix back to the host vector.
    CUDA_CHECK(cudaMemcpy(C.data(), d_C, mat_bytes, cudaMemcpyDeviceToHost));

    // (5) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_coords));
    CUDA_CHECK(cudaFree(d_mean));
    CUDA_CHECK(cudaFree(d_C));
}
