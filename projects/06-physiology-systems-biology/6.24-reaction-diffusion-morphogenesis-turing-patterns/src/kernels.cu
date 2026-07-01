// ===========================================================================
// src/kernels.cu  --  Turing stencil kernel + ping-pong time loop (GPU)
// ---------------------------------------------------------------------------
// Project 6.24 : Reaction-Diffusion Morphogenesis (Turing Patterns)
//
// WHAT THIS FILE DOES
//   Implements the device stencil kernel (rd_step_kernel) and the host-side glue
//   (simulate_gpu) that allocates GPU memory, uploads the seed, runs the time
//   loop launching one kernel per step while ping-ponging two buffer pairs,
//   times the loop, and brings the final fields back. This is the GPU twin of
//   simulate_cpu(); the per-cell physics is the SHARED tu_update() in turing.h,
//   so the two agree. main.cu runs both and compares them.
//
// READ THIS AFTER: kernels.cuh (the thread-mapping idea) and turing.h (physics).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// 16x16 = 256 threads per block over the 2-D grid. 256 is a multiple of the
// 32-lane warp and a solid occupancy default on sm_75..sm_89; a 16x16 tile keeps
// neighbouring cells (and thus the stencil's neighbour reads) close together in
// the L1/L2 cache, which matters because each cell re-reads its 4 neighbours.
static constexpr int TILE = 16;

// ---------------------------------------------------------------------------
// rd_step_kernel: one thread advances one grid cell by one timestep.
//
//   Launch config (set in simulate_gpu):
//     block = (TILE, TILE)                          -- 256 threads
//     grid  = (ceil(nx/TILE), ceil(ny/TILE))        -- covers every cell
//   Thread-to-data map:
//     x = blockIdx.x*blockDim.x + threadIdx.x  (column)
//     y = blockIdx.y*blockDim.y + threadIdx.y  (row)
//   Memory: reads a[],h[] (this cell + 4 neighbours) from GLOBAL memory, writes
//   an[],hn[] for its own cell. No shared memory and no atomics: neighbours are
//   READ-only within a step, and each cell is written by exactly one thread, so
//   there are no races. (A shared-memory tiled variant that caches the halo is a
//   classic optimization -- see THEORY §GPU mapping and the Exercises.)
// ---------------------------------------------------------------------------
__global__ void rd_step_kernel(TuringParams P,
                               const double* __restrict__ a,
                               const double* __restrict__ h,
                               double* __restrict__ an,
                               double* __restrict__ hn) {
    // This thread's cell coordinates within the 2-D domain.
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;

    // GUARD THE RAGGED EDGE: nx/ny need not be exact multiples of TILE, so the
    // border blocks contain threads with x>=nx or y>=ny. They must do nothing,
    // or they would index out of bounds (an illegal-address fault).
    if (x >= P.nx || y >= P.ny) return;

    // The whole per-cell update -- the SAME function the CPU reference runs, so
    // the fields stay in lockstep. This is where the "short-range activation /
    // long-range inhibition" physics actually happens.
    tu_update(x, y, P, a, h, an, hn);
}

// ---------------------------------------------------------------------------
// simulate_gpu: host wrapper running the whole GPU simulation.
//
// The canonical CUDA steps, specialized for an iterative stencil:
//   (1) allocate FOUR device buffers: two ping-pong pairs (a/h source + dest)
//   (2) copy the initial fields host->device
//   (3) TIME LOOP: launch rd_step_kernel, then swap source<->dest pointers
//   (4) copy the final fields device->host
//   (5) free device memory
// We time only the loop (step 3) with CUDA events, excluding the one-time
// H2D/D2H transfers, so the reported figure reflects the compute itself.
// ---------------------------------------------------------------------------
void simulate_gpu(const TuringParams& P, std::vector<double>& a,
                  std::vector<double>& h, float* kernel_ms) {
    const int N = P.nx * P.ny;
    const std::size_t bytes = static_cast<std::size_t>(N) * sizeof(double);

    // (1) Two ping-pong PAIRS. The d_ prefix marks DEVICE pointers (CLAUDE.md
    //     §12): dereferencing one on the host would crash, so naming matters.
    //     *_a = activator buffers, *_h = inhibitor buffers; "a"/"b" = the two
    //     alternating copies we swap between.
    double *d_aa = nullptr, *d_ab = nullptr;   // activator: buffer A / buffer B
    double *d_ha = nullptr, *d_hb = nullptr;   // inhibitor: buffer A / buffer B
    CUDA_CHECK(cudaMalloc(&d_aa, bytes));      // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_ab, bytes));
    CUDA_CHECK(cudaMalloc(&d_ha, bytes));
    CUDA_CHECK(cudaMalloc(&d_hb, bytes));

    // (2) Upload the seed into the "source" buffers (A).
    CUDA_CHECK(cudaMemcpy(d_aa, a.data(), bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_ha, h.data(), bytes, cudaMemcpyHostToDevice));

    // Launch geometry: a 2-D block grid tiling the domain (round up each axis).
    dim3 block(TILE, TILE);
    dim3 grid((P.nx + TILE - 1) / TILE, (P.ny + TILE - 1) / TILE);

    // Working pointers: start with A as source, B as destination.
    double* a_src = d_aa; double* a_dst = d_ab;
    double* h_src = d_ha; double* h_dst = d_hb;

    // (3) The time loop. One kernel launch per step; each launch reads the frozen
    //     "src" state and writes the fresh "dst" state, then we swap the roles.
    //     Ping-pong via pointer swap costs nothing (no data is copied).
    GpuTimer timer;
    timer.start();
    for (int s = 0; s < P.steps; ++s) {
        rd_step_kernel<<<grid, block>>>(P, a_src, h_src, a_dst, h_dst);
        double* ta = a_src; a_src = a_dst; a_dst = ta;   // ping-pong activator
        double* th = h_src; h_src = h_dst; h_dst = th;   // ping-pong inhibitor
    }
    *kernel_ms = timer.stop_ms();          // GPU-measured loop time
    CUDA_CHECK_LAST("rd_step_kernel");     // catch launch + execution errors

    // (4) After the final swap, *_src hold the newest state. Copy them back so
    //     the caller sees the result in the vectors it passed in.
    CUDA_CHECK(cudaMemcpy(a.data(), a_src, bytes, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h.data(), h_src, bytes, cudaMemcpyDeviceToHost));

    // (5) Always free what we allocated (there is no GPU garbage collector).
    CUDA_CHECK(cudaFree(d_aa));
    CUDA_CHECK(cudaFree(d_ab));
    CUDA_CHECK(cudaFree(d_ha));
    CUDA_CHECK(cudaFree(d_hb));
}
