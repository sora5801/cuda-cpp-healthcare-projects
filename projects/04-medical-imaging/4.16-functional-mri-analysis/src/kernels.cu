// ===========================================================================
// src/kernels.cu  --  Per-voxel fMRI GLM kernel + host wrapper
// ---------------------------------------------------------------------------
// Project 4.16 : Functional MRI Analysis
//
// This is the GPU twin of glm_cpu() in reference_cpu.cpp. Both loop over voxels
// calling the SAME fit_voxel() from glm.h -- only the "loop" differs: a serial
// for-loop on the CPU, one-thread-per-voxel here. main.cu runs both and asserts
// they agree. See ../THEORY.md §"GPU mapping" for the reasoning.
// ===========================================================================
#include "kernels.cuh"
#include "glm.h"                 // GlmDesign, VoxelStat, fit_voxel  (HD core)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer

// ---------------------------------------------------------------------------
// Voxel-independent data in CONSTANT memory.
//   * c_design   : the GlmDesign (T, TR, block_scans) -- identical for every
//                  voxel, read by every thread, never written during the launch.
//   * c_XtX_inv  : the precomputed 3x3 (X^T X)^-1 (9 doubles = 72 bytes).
//   Constant memory's broadcast cache serves one address to a whole warp in a
//   single transaction, so these tiny shared operands cost ~nothing to read.
//   (Contrast: passing them as kernel args also works and lives in constant
//   memory too, but a named __constant__ symbol makes the "shared, read-only"
//   intent explicit and mirrors flagship 1.12's constant-memory query.)
// ---------------------------------------------------------------------------
__constant__ GlmDesign c_design;
__constant__ double    c_XtX_inv[9];

// 256 threads/block: a multiple of the 32-lane warp with good occupancy on
// sm_75..sm_89. Each thread does an FP64-heavy per-voxel fit; 256 keeps enough
// warps resident to hide the global-memory latency of streaming its y-row.
static constexpr int THREADS_PER_BLOCK = 256;

// ---------------------------------------------------------------------------
// glm_kernel: one logical thread per voxel, via a grid-stride loop so a fixed
//   grid covers any V. Thread (blockIdx.x, threadIdx.x) starts at
//   v = block*blockDim + thread and strides by the total thread count.
//   Memory: c_design/c_XtX_inv from the constant cache; voxel v's y-row from
//   global memory (T contiguous doubles). No shared memory or atomics -- the
//   outputs are fully independent, so this is pure map-style parallelism.
// ---------------------------------------------------------------------------
__global__ void glm_kernel(const double* __restrict__ d_bold, int V, int T,
                           double* __restrict__ d_t, double* __restrict__ d_beta) {
    const int stride = blockDim.x * gridDim.x;                 // total threads
    for (int v = blockIdx.x * blockDim.x + threadIdx.x; v < V; v += stride) {
        // Pointer to this voxel's contiguous time-series (voxel-major layout).
        const double* y = d_bold + static_cast<std::size_t>(v) * T;
        // The ENTIRE fit is the shared HD core -> identical math to the CPU.
        const VoxelStat s = fit_voxel(y, c_design, c_XtX_inv);
        d_t[v]    = s.tstat;
        d_beta[v] = s.beta_task;
    }
}

// ---------------------------------------------------------------------------
// glm_gpu: the canonical CUDA steps. Design + inverse go to constant memory;
//   BOLD goes to global memory. We time ONLY the kernel (CUDA events), not the
//   H2D/D2H copies (discussed separately in THEORY §"honest timing").
// ---------------------------------------------------------------------------
void glm_gpu(const FmriDataset& ds, const double XtX_inv[9],
             std::vector<double>& tstat, std::vector<double>& beta, float* kernel_ms) {
    const int V = ds.V;
    const int T = ds.design.T;
    tstat.assign(static_cast<std::size_t>(V), 0.0);
    beta.assign(static_cast<std::size_t>(V), 0.0);
    const std::size_t bold_bytes = static_cast<std::size_t>(V) * T * sizeof(double);
    const std::size_t out_bytes  = static_cast<std::size_t>(V) * sizeof(double);

    // (a) Upload the voxel-independent operands to the __constant__ symbols.
    CUDA_CHECK(cudaMemcpyToSymbol(c_design,  &ds.design, sizeof(GlmDesign)));
    CUDA_CHECK(cudaMemcpyToSymbol(c_XtX_inv, XtX_inv,    9 * sizeof(double)));

    // (b) Allocate + upload the BOLD data, and allocate the two output buffers.
    double* d_bold = nullptr;   // [V*T] device, voxel-major
    double* d_t    = nullptr;   // [V]   device, t-statistics
    double* d_beta = nullptr;   // [V]   device, task betas
    CUDA_CHECK(cudaMalloc(&d_bold, bold_bytes));
    CUDA_CHECK(cudaMalloc(&d_t,    out_bytes));
    CUDA_CHECK(cudaMalloc(&d_beta, out_bytes));
    CUDA_CHECK(cudaMemcpy(d_bold, ds.bold.data(), bold_bytes, cudaMemcpyHostToDevice));

    // (c) Launch. Enough blocks to cover V one-thread-per-voxel, capped so the
    //     grid stays modest; the grid-stride loop covers any larger V.
    int blocks = (V + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    if (blocks < 1)    blocks = 1;
    if (blocks > 4096) blocks = 4096;
    GpuTimer timer;
    timer.start();
    glm_kernel<<<blocks, THREADS_PER_BLOCK>>>(d_bold, V, T, d_t, d_beta);
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("glm_kernel");

    // (d) Copy results back, then (e) free device memory.
    CUDA_CHECK(cudaMemcpy(tstat.data(), d_t,    out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(beta.data(),  d_beta, out_bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_bold));
    CUDA_CHECK(cudaFree(d_t));
    CUDA_CHECK(cudaFree(d_beta));
}
