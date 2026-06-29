// ===========================================================================
// src/kernels.cu  --  Reaction-diffusion stencil kernel + ping-pong time loop
// ---------------------------------------------------------------------------
// Project 14.02 : Spatial / Whole-Cell Reaction-Diffusion (teaching stencil)
//
// GPU twin of simulate_cpu(): identical per-cell physics (rd.h), one thread per
// cell, double-buffered. main.cu compares the final fields. See ../THEORY.md.
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

static constexpr int TILE = 16;   // 16x16 = 256 threads/block over the 2-D grid

__global__ void rd_step_kernel(RdParams P, const double* __restrict__ U, const double* __restrict__ V,
                               double* __restrict__ Un, double* __restrict__ Vn) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= P.nx || y >= P.ny) return;
    rd_update(x, y, P, U, V, Un, Vn);
}

void simulate_gpu(const RdParams& P, std::vector<double>& U, std::vector<double>& V, float* kernel_ms) {
    const int N = P.nx * P.ny;
    const std::size_t bytes = static_cast<std::size_t>(N) * sizeof(double);

    double *d_Ua = nullptr, *d_Ub = nullptr, *d_Va = nullptr, *d_Vb = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Ua, bytes));
    CUDA_CHECK(cudaMalloc(&d_Ub, bytes));
    CUDA_CHECK(cudaMalloc(&d_Va, bytes));
    CUDA_CHECK(cudaMalloc(&d_Vb, bytes));
    CUDA_CHECK(cudaMemcpy(d_Ua, U.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_Va, V.data(), bytes, cudaMemcpyHostToDevice));

    dim3 block(TILE, TILE);
    dim3 grid((P.nx + TILE - 1) / TILE, (P.ny + TILE - 1) / TILE);

    double* Us = d_Ua; double* Vs = d_Va;   // source
    double* Ud = d_Ub; double* Vd = d_Vb;   // destination
    GpuTimer timer;
    timer.start();
    for (int s = 0; s < P.steps; ++s) {
        rd_step_kernel<<<grid, block>>>(P, Us, Vs, Ud, Vd);
        double* tu = Us; Us = Ud; Ud = tu;   // ping-pong U
        double* tv = Vs; Vs = Vd; Vd = tv;   // ping-pong V
    }
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("rd_step_kernel");

    // Us/Vs hold the latest state after the final swap.
    CUDA_CHECK(cudaMemcpy(U.data(), Us, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(V.data(), Vs, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_Ua));
    CUDA_CHECK(cudaFree(d_Ub));
    CUDA_CHECK(cudaFree(d_Va));
    CUDA_CHECK(cudaFree(d_Vb));
}
