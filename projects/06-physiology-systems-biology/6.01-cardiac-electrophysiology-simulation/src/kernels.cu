// ===========================================================================
// src/kernels.cu  --  Monodomain GPU kernels + host ping-pong time loop
// ---------------------------------------------------------------------------
// Project 6.1 : Cardiac Electrophysiology Simulation
//
// GPU twin of monodomain_cpu(): identical per-cell physics (shared cardiac_cell.h),
// one thread per grid cell, host-driven operator-split time loop. Two kernels run
// per step -- react_kernel (pointwise ODE) then diffuse_kernel (5-point stencil,
// ping-pong buffers). main.cu runs both CPU and GPU and compares the voltage
// fields. See ../THEORY.md "GPU mapping".
// ===========================================================================
#include "kernels.cuh"
#include "cardiac_cell.h"        // react_step, diffuse_cell (host+device physics)
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// 16x16 = 256 threads/block over the 2-D tissue grid. 256 is a multiple of the
// 32-lane warp and gives the scheduler enough warps to hide global-memory
// latency; a 2-D block matches the 2-D grid so neighbour reads stay local.
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// react_kernel: thread (x,y) owns cell i=(x,y). It advances ONLY that cell's
//   (V,w) by one reaction sub-step. Cells are independent (each touches only its
//   own V[i], w[i]) -> no shared memory, no atomics, no races. The math is the
//   shared react_step() so it matches the CPU exactly.
//     grid  : ceil(nx/16) x ceil(ny/16) blocks
//     block : 16 x 16 threads
//     thread (blockIdx,threadIdx) -> cell (x,y) = block*TILE + thread
// ---------------------------------------------------------------------------
__global__ void react_kernel(int nx, int ny, MonodomainParams p,
                             double* __restrict__ V, double* __restrict__ w) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;                 // guard ragged edge tiles
    const std::size_t i = cell_idx(x, y, nx);
    react_step(&V[i], &w[i], p);                    // in-place local ODE update
}

// ---------------------------------------------------------------------------
// diffuse_kernel: thread (x,y) computes the diffusion update for its cell,
//   reading the 4 neighbours from the READ-ONLY V_in buffer and writing the
//   result to V_out. Because reads and writes use SEPARATE buffers, a thread
//   never observes a neighbour half-updated -> the stencil is deterministic and
//   race-free (this is why we ping-pong). The math is the shared diffuse_cell().
// ---------------------------------------------------------------------------
__global__ void diffuse_kernel(int nx, int ny, MonodomainParams p,
                               const double* __restrict__ V_in,
                               double* __restrict__ V_out) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;                 // guard ragged edge tiles
    V_out[cell_idx(x, y, nx)] = diffuse_cell(x, y, V_in, p);
}

// ---------------------------------------------------------------------------
// monodomain_gpu: host wrapper. Allocates device buffers, uploads the shared
//   initial state, runs the operator-split loop (react then diffuse each step,
//   ping-ponging the two V buffers), and brings the final fields back. We time
//   the WHOLE loop (all kernel launches) with CUDA events -- a teaching artifact,
//   not a benchmark claim (CLAUDE.md section 12).
// ---------------------------------------------------------------------------
void monodomain_gpu(const MonodomainParams& p,
                    std::vector<double>& V_final, std::vector<double>& w_final,
                    float* kernel_ms) {
    const std::size_t cells = static_cast<std::size_t>(p.nx) * p.ny;
    const std::size_t bytes = cells * sizeof(double);

    // Build the SAME initial condition the CPU uses, on the host, then upload.
    std::vector<double> V0, w0;
    init_state(p, V0, w0);

    // Device buffers. d_ prefix = DEVICE pointer (dereferencing on host crashes).
    //   d_Va / d_Vb : the two voltage buffers we ping-pong for diffusion.
    //   d_w         : recovery variable (only the reaction kernel touches it).
    double *d_Va = nullptr, *d_Vb = nullptr, *d_w = nullptr;
    CUDA_CHECK(cudaMalloc(&d_Va, bytes));           // can fail: out of device mem
    CUDA_CHECK(cudaMalloc(&d_Vb, bytes));
    CUDA_CHECK(cudaMalloc(&d_w,  bytes));
    CUDA_CHECK(cudaMemcpy(d_Va, V0.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w,  w0.data(), bytes, cudaMemcpyHostToDevice));

    // 2-D launch grid: one thread per cell, blocks tile the grid; ceil-division
    // covers a grid whose size is not a multiple of TILE (edge tiles are ragged).
    dim3 block(TILE, TILE);
    dim3 grid((p.nx + TILE - 1) / TILE, (p.ny + TILE - 1) / TILE);

    // Operator-split time loop. `src` holds the current voltage field; the
    // reaction kernel updates it in place, then the diffusion kernel reads src
    // and writes dst, and we swap. We time the entire loop with CUDA events.
    double* src = d_Va;
    double* dst = d_Vb;
    GpuTimer timer;
    timer.start();
    for (int s = 0; s < p.steps; ++s) {
        // (A) REACTION half-step: pointwise ODE, in place on `src`.
        react_kernel<<<grid, block>>>(p.nx, p.ny, p, src, d_w);
        // (B) DIFFUSION half-step: stencil, src -> dst, then swap (ping-pong).
        diffuse_kernel<<<grid, block>>>(p.nx, p.ny, p, src, dst);
        double* tmp = src; src = dst; dst = tmp;
    }
    *kernel_ms = timer.stop_ms();                   // GPU-measured loop time
    CUDA_CHECK_LAST("monodomain kernels");          // catch launch/exec errors

    // `src` holds the latest voltage field (after the final swap). Copy both
    // fields back so main.cu can verify V (and, for interest, w) against the CPU.
    V_final.assign(cells, 0.0);
    w_final.assign(cells, 0.0);
    CUDA_CHECK(cudaMemcpy(V_final.data(), src, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(w_final.data(), d_w, bytes, cudaMemcpyDeviceToHost));

    // Always free device memory (there is no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_Va));
    CUDA_CHECK(cudaFree(d_Vb));
    CUDA_CHECK(cudaFree(d_w));
}
