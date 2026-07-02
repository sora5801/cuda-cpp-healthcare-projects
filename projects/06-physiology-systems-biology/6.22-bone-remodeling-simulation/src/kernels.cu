// ===========================================================================
// src/kernels.cu  --  Bone-remodeling stencil kernels + host drive loop
// ---------------------------------------------------------------------------
// Project 6.22 : Bone Remodeling Simulation   (REDUCED-SCOPE teaching version)
//
// GPU twin of bone_cpu(): identical per-voxel physics (shared bone_remodel.h),
// one thread per voxel, host-driven nested loops with ping-pong buffers. main.cu
// runs both CPU and GPU and compares the final density fields. The kernels are
// pure thread-to-voxel mapping wrappers around the shared physics functions --
// all the biology lives in bone_remodel.h. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "bone_remodel.h"        // BR_HD physics: bone_relax_point / bone_apply_stimulus
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <algorithm>             // std::swap
#include <vector>

// 16x16 = 256 threads/block over the 2-D voxel grid: a multiple of the 32-lane
// warp, good occupancy on sm_75..sm_89, and a natural 2-D tiling of the domain.
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// relax_kernel: one Jacobi sweep of the stimulus field. Thread (x,y) owns one
//   voxel. Nodes are independent within a sweep (each writes only its own
//   S_new, reading neighbours from the separate read-only S_old buffer), so
//   there are no data races and no atomics -- the reason ping-pong works.
// ---------------------------------------------------------------------------
__global__ void relax_kernel(int nx, int ny, double load, int load_x0, int load_x1,
                             const double* __restrict__ S_old,
                             const double* __restrict__ rho,
                             double* __restrict__ S_new) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's column
    const int y = blockIdx.y * blockDim.y + threadIdx.y;   // this thread's row
    if (x >= nx || y >= ny) return;                        // guard ragged edge tiles
    // The entire stencil is the shared function -> the kernel body is just the map.
    S_new[bone_idx(x, y, nx)] =
        bone_relax_point(x, y, nx, ny, load, load_x0, load_x1, S_old, rho);
}

// ---------------------------------------------------------------------------
// remodel_kernel: one mechanostat density update. Thread (x,y) reads the
//   settled stimulus S and the current density rho_old, writes rho_new. Again
//   fully independent per voxel.
// ---------------------------------------------------------------------------
__global__ void remodel_kernel(int nx, int ny, double setpoint, double lazy,
                               double rate, double rho_min,
                               const double* __restrict__ S,
                               const double* __restrict__ rho_old,
                               double* __restrict__ rho_new) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    rho_new[bone_idx(x, y, nx)] =
        bone_apply_stimulus(x, y, nx, setpoint, lazy, rate, rho_min, S, rho_old);
}

// ---------------------------------------------------------------------------
// bone_gpu: host wrapper running the whole simulation on the device.
//   IMPORTANT: the buffer-swap bookkeeping MUST mirror bone_cpu() exactly so the
//   two paths perform the identical sequence of arithmetic. In particular, after
//   the inner Jacobi loop we make the FRESHEST stimulus field live in the "Sa"
//   buffer (d_Sa), just as the CPU copies Sb into Sa when the sweep count is odd.
// ---------------------------------------------------------------------------
void bone_gpu(const BoneParams& p,
              std::vector<double>& rho_final,
              std::vector<double>& S_final,
              float* kernel_ms) {
    const std::size_t N     = static_cast<std::size_t>(p.nx) * p.ny;  // voxel count
    const std::size_t bytes = N * sizeof(double);

    // --- Device buffers -----------------------------------------------------
    //   d_rho / d_rho_next : density ping-pong (updated once per remodeling step)
    //   d_Sa   / d_Sb      : stimulus ping-pong (updated once per Jacobi sweep)
    // The d_ prefix marks DEVICE pointers (CLAUDE.md section 12): dereferencing
    // one on the host would crash, so the naming is load-bearing.
    double *d_rho = nullptr, *d_rho_next = nullptr, *d_Sa = nullptr, *d_Sb = nullptr;
    CUDA_CHECK(cudaMalloc(&d_rho,      bytes));   // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_rho_next, bytes));
    CUDA_CHECK(cudaMalloc(&d_Sa,       bytes));
    CUDA_CHECK(cudaMalloc(&d_Sb,       bytes));

    // Initialize on the host, then upload: rho = uniform rho_init, S = 0.
    std::vector<double> rho_init(N, p.rho_init);
    std::vector<double> S_zero(N, 0.0);
    CUDA_CHECK(cudaMemcpy(d_rho, rho_init.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Sa,  S_zero.data(),   bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Sb,  S_zero.data(),   bytes, cudaMemcpyHostToDevice));

    // 2-D launch config covering the whole grid.
    dim3 block(TILE, TILE);
    dim3 grid((p.nx + TILE - 1) / TILE, (p.ny + TILE - 1) / TILE);

    // Time ALL kernel launches together with CUDA events (the honest way to time
    // GPU work; a host clock would mostly measure launch overhead -- see timer.cuh).
    GpuTimer timer;
    timer.start();

    for (int step = 0; step < p.remodel_steps; ++step) {
        // (1) Relax the stimulus field with `relax_iters` Jacobi sweeps,
        //     ping-ponging d_Sin <-> d_Sout. Warm-started from d_Sa each step.
        double* d_Sin  = d_Sa;
        double* d_Sout = d_Sb;
        for (int it = 0; it < p.relax_iters; ++it) {
            relax_kernel<<<grid, block>>>(p.nx, p.ny, p.load, p.load_x0, p.load_x1,
                                          d_Sin, d_rho, d_Sout);
            std::swap(d_Sin, d_Sout);      // last-written becomes next input
        }
        // Make the freshest field live in d_Sa (mirror of the CPU's `Sa = Sb`
        // when the sweep count is odd). If d_Sin already points at d_Sa, nothing
        // to do; otherwise copy the fresh d_Sb into d_Sa on the device.
        if (d_Sin != d_Sa) {
            CUDA_CHECK(cudaMemcpy(d_Sa, d_Sin, bytes, cudaMemcpyDeviceToDevice));
        }

        // (2) Apply the mechanostat: rho_next(x,y) from the settled field d_Sa.
        remodel_kernel<<<grid, block>>>(p.nx, p.ny, p.setpoint, p.lazy,
                                        p.rate, p.rho_min, d_Sa, d_rho, d_rho_next);
        std::swap(d_rho, d_rho_next);      // adopt the remodeled density
    }

    *kernel_ms = timer.stop_ms();          // GPU-measured total kernel time
    CUDA_CHECK_LAST("bone remodeling kernels");   // catch launch + execution errors

    // --- Download results ---------------------------------------------------
    // `d_rho` holds the final density after the last swap; `d_Sa` holds the last
    // settled stimulus field (matching what bone_cpu() returns in S_final).
    rho_final.assign(N, 0.0);
    S_final.assign(N, 0.0);
    CUDA_CHECK(cudaMemcpy(rho_final.data(), d_rho, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(S_final.data(),   d_Sa,  bytes, cudaMemcpyDeviceToHost));

    // --- Free (no GPU garbage collector exists) -----------------------------
    CUDA_CHECK(cudaFree(d_rho));
    CUDA_CHECK(cudaFree(d_rho_next));
    CUDA_CHECK(cudaFree(d_Sa));
    CUDA_CHECK(cudaFree(d_Sb));
}
