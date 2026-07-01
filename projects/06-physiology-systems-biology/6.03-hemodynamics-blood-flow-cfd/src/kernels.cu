// ===========================================================================
// src/kernels.cu  --  Fractional-step NSE kernels + host ping-pong time loop
// ---------------------------------------------------------------------------
// Project 6.3 : Hemodynamics / Blood-Flow CFD   (reduced-scope teaching version)
//
// WHAT THIS FILE DOES
//   The GPU twin of the CPU reference (reference_cpu.cpp). It implements the four
//   per-step kernels of Chorin's projection method and the host wrapper nse_gpu()
//   that drives the time loop, allocating device buffers, launching the kernels,
//   ping-ponging the velocity and pressure buffers, and copying the final field
//   back. main.cu runs this and the CPU reference and compares them.
//
//   Every kernel is a thin wrapper: it maps thread (x,y) to grid cell (x,y),
//   guards the ragged edge tiles, and calls the SAME per-cell function from
//   nse_channel.h that the CPU loops -> the two paths compute identical math.
//   The whole update touches only nearest neighbours (a stencil), so within a
//   step every cell is independent: each kernel writes only its own cell and
//   reads read-only "old" buffers -> no races, no atomics (PATTERNS.md §1).
//
// READ THIS AFTER: kernels.cuh (declarations + thread mapping), nse_channel.h.
// ===========================================================================
#include "kernels.cuh"
#include "nse_channel.h"         // predictor/divergence/pressure/corrector cells
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

#include <algorithm>             // std::swap
#include <vector>

// 16x16 = 256 threads per block over the 2-D grid. 256 is a multiple of the
// 32-lane warp and gives the scheduler enough warps to hide global-memory
// latency on sm_75..sm_89. The tile is square so the halo of neighbour reads is
// balanced in x and y. Tune per GPU (BUILD_GUIDE covers occupancy).
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// predictor_kernel: thread (x,y) computes the provisional velocity u* for its
//   cell (advection + diffusion + body force, no pressure yet). Reads the
//   read-only u,v (state at step n), writes us,vs. See predictor_cell().
//   grid  : ceil(nx/TILE) x ceil(ny/TILE) blocks
//   block : TILE x TILE threads
//   thread (x,y) = (blockIdx*blockDim + threadIdx) -> cell (x,y)
// ---------------------------------------------------------------------------
__global__ void predictor_kernel(int nx, int ny, double h, double dt, double gx,
                                 double nu0, double nu_inf, double lambda,
                                 double n_cy, double a_cy,
                                 const double* __restrict__ u,
                                 const double* __restrict__ v,
                                 double* __restrict__ us,
                                 double* __restrict__ vs) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;      // guard ragged edge tiles
    predictor_cell(x, y, nx, ny, h, dt, gx, nu0, nu_inf, lambda, n_cy, a_cy,
                   u, v, us, vs);
}

// ---------------------------------------------------------------------------
// divergence_kernel: thread (x,y) writes rhs = scale * div(u*) for its cell,
//   where scale = rho/dt. This is the right-hand side of the pressure Poisson
//   equation. Reads us,vs; writes rhs. See divergence_cell().
// ---------------------------------------------------------------------------
__global__ void divergence_kernel(int nx, int ny, double h, double scale,
                                  const double* __restrict__ us,
                                  const double* __restrict__ vs,
                                  double* __restrict__ rhs) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    rhs[idx(x, y, nx)] = scale * divergence_cell(x, y, nx, ny, h, us, vs);
}

// ---------------------------------------------------------------------------
// pressure_kernel: thread (x,y) performs ONE Jacobi sweep for its cell, reading
//   the OLD pressure p_old everywhere and writing the NEW pressure p_new. Jacobi
//   (as opposed to Gauss-Seidel) uses a frozen p_old, so every cell is
//   independent within a sweep -> perfect for the GPU. The host ping-pongs the
//   two pressure buffers between sweeps. See pressure_jacobi_cell().
// ---------------------------------------------------------------------------
__global__ void pressure_kernel(int nx, int ny, double h,
                                const double* __restrict__ p_old,
                                const double* __restrict__ rhs,
                                double* __restrict__ p_new) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    p_new[idx(x, y, nx)] = pressure_jacobi_cell(x, y, nx, ny, h, p_old,
                                                rhs[idx(x, y, nx)]);
}

// ---------------------------------------------------------------------------
// corrector_kernel: thread (x,y) applies the projection u = u* - (dt/rho)grad(p)
//   for its cell, producing the divergence-free velocity. Reads us,vs,p; writes
//   u_new,v_new (which become the state at step n+1). See corrector_cell().
// ---------------------------------------------------------------------------
__global__ void corrector_kernel(int nx, int ny, double h, double dt, double rho,
                                 const double* __restrict__ us,
                                 const double* __restrict__ vs,
                                 const double* __restrict__ p,
                                 double* __restrict__ u_new,
                                 double* __restrict__ v_new) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= nx || y >= ny) return;
    corrector_cell(x, y, nx, ny, h, dt, rho, us, vs, p, u_new, v_new);
}

// ---------------------------------------------------------------------------
// nse_gpu: allocate device buffers, run the fractional-step time loop, and copy
//   the final velocity fields back. The five canonical CUDA steps, but the "run
//   the kernel" step is a whole loop of four kernels per time step with two sets
//   of ping-pong buffers (velocity u<->u_new; pressure pa<->pb).
//
//   We time ONLY the kernel loop with CUDA events (H2D/D2H copies excluded) so
//   the figure reflects compute, not PCIe transfer.
// ---------------------------------------------------------------------------
void nse_gpu(const ChannelParams& p,
             std::vector<double>& u_final,
             std::vector<double>& v_final,
             float* kernel_ms) {
    const std::size_t N     = static_cast<std::size_t>(p.nx) * p.ny;
    const std::size_t bytes = N * sizeof(double);

    // (1) Device buffers. d_ prefix = DEVICE pointer (dereferencing on the host
    //     would crash). Two velocity fields for ping-pong (u->u_new then swap),
    //     the predictor fields us,vs, two pressure fields, and the Poisson RHS.
    double *d_u=nullptr, *d_v=nullptr, *d_un=nullptr, *d_vn=nullptr;
    double *d_us=nullptr, *d_vs=nullptr;
    double *d_pa=nullptr, *d_pb=nullptr, *d_rhs=nullptr;
    CUDA_CHECK(cudaMalloc(&d_u,  bytes));
    CUDA_CHECK(cudaMalloc(&d_v,  bytes));
    CUDA_CHECK(cudaMalloc(&d_un, bytes));
    CUDA_CHECK(cudaMalloc(&d_vn, bytes));
    CUDA_CHECK(cudaMalloc(&d_us, bytes));
    CUDA_CHECK(cudaMalloc(&d_vs, bytes));
    CUDA_CHECK(cudaMalloc(&d_pa, bytes));
    CUDA_CHECK(cudaMalloc(&d_pb, bytes));
    CUDA_CHECK(cudaMalloc(&d_rhs, bytes));

    // (2) Initialize from REST: u=v=0 everywhere. cudaMemset writes zero bytes,
    //     which is exactly 0.0 for IEEE-754 doubles.
    CUDA_CHECK(cudaMemset(d_u, 0, bytes));
    CUDA_CHECK(cudaMemset(d_v, 0, bytes));

    // 2-D launch grid covering all nx*ny cells with TILE x TILE blocks.
    dim3 block(TILE, TILE);
    dim3 grid((p.nx + TILE - 1) / TILE, (p.ny + TILE - 1) / TILE);
    const double scale = p.rho / p.dt;   // Poisson RHS scaling (rho/dt)

    // (3) Time loop. Current velocity lives in (d_u,d_v); each step writes the
    //     next velocity into (d_un,d_vn) and we swap. Time the whole loop.
    GpuTimer timer;
    timer.start();
    for (int s = 0; s < p.steps; ++s) {
        // 3a. predictor u* (advection + diffusion + body force)
        predictor_kernel<<<grid, block>>>(p.nx, p.ny, p.h, p.dt, p.gx,
                                          p.nu0, p.nu_inf, p.lambda, p.n_cy, p.a_cy,
                                          d_u, d_v, d_us, d_vs);

        // 3b. Poisson RHS = (rho/dt) div(u*)
        divergence_kernel<<<grid, block>>>(p.nx, p.ny, p.h, scale,
                                           d_us, d_vs, d_rhs);

        // 3c. Jacobi pressure solve. Start each step from p=0, then sweep,
        //     ping-ponging d_pa <-> d_pb. After the loop, p_src holds the result.
        CUDA_CHECK(cudaMemset(d_pa, 0, bytes));   // fresh initial guess
        double* p_src = d_pa;
        double* p_dst = d_pb;
        for (int it = 0; it < p.p_iters; ++it) {
            pressure_kernel<<<grid, block>>>(p.nx, p.ny, p.h, p_src, d_rhs, p_dst);
            std::swap(p_src, p_dst);
        }

        // 3d. corrector: u_{n+1} = u* - (dt/rho) grad(p)
        corrector_kernel<<<grid, block>>>(p.nx, p.ny, p.h, p.dt, p.rho,
                                          d_us, d_vs, p_src, d_un, d_vn);

        // Swap so the freshly computed velocity becomes the current state.
        std::swap(d_u, d_un);
        std::swap(d_v, d_vn);
    }
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("nse fractional-step kernels");   // launch + execution check

    // (4) Copy the final velocity back to host vectors (current state = d_u,d_v).
    u_final.assign(N, 0.0);
    v_final.assign(N, 0.0);
    CUDA_CHECK(cudaMemcpy(u_final.data(), d_u, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(v_final.data(), d_v, bytes, cudaMemcpyDeviceToHost));

    // (5) Free every device allocation (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_u));  CUDA_CHECK(cudaFree(d_v));
    CUDA_CHECK(cudaFree(d_un)); CUDA_CHECK(cudaFree(d_vn));
    CUDA_CHECK(cudaFree(d_us)); CUDA_CHECK(cudaFree(d_vs));
    CUDA_CHECK(cudaFree(d_pa)); CUDA_CHECK(cudaFree(d_pb));
    CUDA_CHECK(cudaFree(d_rhs));
}
