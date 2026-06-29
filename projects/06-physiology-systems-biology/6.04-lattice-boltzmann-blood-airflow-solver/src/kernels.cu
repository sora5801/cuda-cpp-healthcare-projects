// ===========================================================================
// src/kernels.cu  --  LBM stencil kernel + host ping-pong time loop
// ---------------------------------------------------------------------------
// Project 6.04 : Lattice-Boltzmann Blood/Airflow Solver
//
// GPU twin of lbm_cpu(): identical per-node physics (shared lbm_d2q9.h), one
// thread per node, host-driven time loop with two buffers. main.cu runs both and
// compares the velocity fields. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "lbm_d2q9.h"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

static constexpr int TILE = 16;   // 16x16 = 256 threads/block over the 2-D lattice

// ---------------------------------------------------------------------------
// lbm_step_kernel: thread (x,y) updates its node for one timestep. The whole
// collide+stream is the shared function -> the kernel body is just the mapping.
// Nodes are independent within a step (each writes only its own f_new), reading
// neighbours from the read-only f_old buffer -> no races, no atomics.
// ---------------------------------------------------------------------------
__global__ void lbm_step_kernel(int nx, int ny, double tau, double gx,
                                const double* __restrict__ f_old,
                                double* __restrict__ f_new) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;                  // guard ragged edge tiles
    lbm_collide_stream(x, y, nx, ny, tau, gx, f_old, f_new);
}

void lbm_gpu(const LbmParams& p, std::vector<double>& f_final, float* kernel_ms) {
    const std::size_t cells = static_cast<std::size_t>(9) * p.nx * p.ny;
    const std::size_t bytes = cells * sizeof(double);

    // Initialize at rest equilibrium on the host, then upload.
    std::vector<double> init(cells);
    for (int y = 0; y < p.ny; ++y)
        for (int x = 0; x < p.nx; ++x)
            for (int i = 0; i < 9; ++i)
                init[lbm_idx(i, x, y, p.nx, p.ny)] = w_i(i);

    double* d_a = nullptr;
    double* d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMemcpy(d_a, init.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((p.nx + TILE - 1) / TILE, (p.ny + TILE - 1) / TILE);

    // Time loop: launch one kernel per step, ping-ponging the two buffers. We
    // time the whole loop (all step kernels) with CUDA events.
    double* src = d_a;
    double* dst = d_b;
    GpuTimer timer;
    timer.start();
    for (int s = 0; s < p.steps; ++s) {
        lbm_step_kernel<<<grid, block>>>(p.nx, p.ny, p.tau, p.gx, src, dst);
        double* tmp = src; src = dst; dst = tmp;
    }
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("lbm_step_kernel");

    // `src` holds the latest state after the final swap.
    f_final.assign(cells, 0.0);
    CUDA_CHECK(cudaMemcpy(f_final.data(), src, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
}
