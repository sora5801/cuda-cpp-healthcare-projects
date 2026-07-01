// ===========================================================================
// src/kernels.cu  --  Tumor-growth stencil + treatment kernels + time loop
// ---------------------------------------------------------------------------
// Project 6.8 : Tumor Growth & Treatment-Response Modeling
//
// WHAT THIS FILE DOES
//   Implements the two device kernels and the host time-loop that drives them:
//     * tumor_grow_kernel  -- one Fisher-KPP explicit-Euler step (a 5-point
//                             stencil), ONE THREAD PER CELL, double-buffered.
//     * tumor_treat_kernel -- one radiotherapy fraction: multiply every cell by
//                             the LQ surviving fraction (embarrassingly parallel).
//     * simulate_gpu       -- allocate device buffers, copy the seed up, run the
//                             time loop (treatment when scheduled, then growth,
//                             then ping-pong swap), copy the final field back.
//
//   The per-cell math is the shared tumor.h (tumor_grow_update /
//   tumor_treat_update), the SAME code the CPU reference runs -- so main.cu can
//   compare the two fields and trust an agreement. This is the GPU twin of
//   simulate_cpu() in reference_cpu.cpp.
//
// READ THIS AFTER: kernels.cuh, tumor.h, reference_cpu.cpp (the serial twin).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// 16x16 = 256 threads/block tiles the 2-D grid. 256 is a solid occupancy default
// on sm_75..sm_89 (8 warps to hide latency); a square tile keeps each thread's
// four neighbour loads close together in the row-major field for good coalescing.
static constexpr int TILE = 16;

// Threads per block for the flat (1-D) treatment kernel over nx*ny cells.
static constexpr int THREADS_1D = 256;

// ---------------------------------------------------------------------------
// tumor_grow_kernel: one Fisher-KPP growth step, one thread per grid cell.
//   Launch config (set in simulate_gpu):
//     block = (TILE, TILE)                          -> 256 threads
//     grid  = (ceil(nx/TILE), ceil(ny/TILE))        -> covers the whole field
//   Thread-to-data map: thread (blockIdx,threadIdx) owns cell
//     x = blockIdx.x*blockDim.x + threadIdx.x,  y = blockIdx.y*blockDim.y + ...
//   Memory: reads u[i] and its 4 neighbours from global memory (via the shared
//   tumor_laplacian), writes un[i]. No shared memory / atomics: within a step
//   every cell reads the FROZEN input buffer `u` and writes a DISTINCT output
//   cell in `un`, so there is no data race. Correctness comes from the host
//   ping-ponging `u`/`un` between steps, exactly as the CPU reference swaps.
//
//   (A shared-memory tiled variant would stage each block's cells + halo into
//   __shared__ to cut the 5x redundant global reads; we keep the naive version
//   because it TEACHES the stencil clearly. See THEORY "GPU mapping".)
// ---------------------------------------------------------------------------
__global__ void tumor_grow_kernel(TumorParams P, const double* __restrict__ u,
                                  double* __restrict__ un) {
    const int x = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's column
    const int y = blockIdx.y * blockDim.y + threadIdx.y;   // this thread's row
    if (x >= P.nx || y >= P.ny) return;                    // guard the ragged edge
    // Delegate to the shared per-cell physics so CPU and GPU compute identically.
    tumor_grow_update(x, y, P, u, un);
}

// ---------------------------------------------------------------------------
// tumor_treat_kernel: apply ONE radiotherapy fraction, one thread per cell.
//   A flat 1-D launch over n = nx*ny cells: thread i multiplies u[i] by the
//   precomputed LQ surviving fraction `survival`. Fully independent per cell, in
//   place, so no buffers to swap. Guards the ragged last block with i < n.
// ---------------------------------------------------------------------------
__global__ void tumor_treat_kernel(int n, double survival, double* __restrict__ u) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // this thread's cell
    if (i >= n) return;
    tumor_treat_update(i, survival, u);                    // u[i] *= survival
}

// ---------------------------------------------------------------------------
// simulate_gpu: the host wrapper that owns the whole GPU computation.
//   Steps: (1) allocate two device density buffers; (2) copy the seed field up;
//   (3) run the time loop -- on a scheduled fraction launch the treatment kernel
//   in place on the current buffer, then launch the growth kernel current->next
//   and swap the pointers (ping-pong); (4) copy the final field back; (5) free.
//   We time the whole loop with CUDA events -- a teaching artifact, not a
//   benchmark claim (CLAUDE.md section 12).
// ---------------------------------------------------------------------------
void simulate_gpu(const TumorParams& P, std::vector<double>& u, float* kernel_ms) {
    const int N = P.nx * P.ny;
    const std::size_t bytes = static_cast<std::size_t>(N) * sizeof(double);

    // (1) Two device buffers for the ping-pong: d_a (current) and d_b (next).
    //     The d_ prefix marks DEVICE pointers -- dereferencing on the host crashes.
    double *d_a = nullptr, *d_b = nullptr;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));   // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_b, bytes));

    // (2) Upload the seeded initial field into the "current" buffer.
    CUDA_CHECK(cudaMemcpy(d_a, u.data(), bytes, cudaMemcpyHostToDevice));

    // Launch geometry: a 2-D tile grid for growth, a 1-D grid for treatment.
    dim3 block(TILE, TILE);
    dim3 grid((P.nx + TILE - 1) / TILE, (P.ny + TILE - 1) / TILE);
    const int blocks_1d = (N + THREADS_1D - 1) / THREADS_1D;

    double* us = d_a;   // source      (current state; read by this step)
    double* ud = d_b;   // destination (next state;    written by this step)

    GpuTimer timer;
    timer.start();
    for (int s = 0; s < P.steps; ++s) {
        // (3a) Treatment on scheduled fractions: in-place per-cell multiply on
        //      the CURRENT buffer, so the following growth step sees the kill.
        //      is_fraction_step + lq_survival are the SAME host helpers the CPU
        //      reference uses, so both paths dose identically.
        if (is_fraction_step(P, s)) {
            const double S = lq_survival(P.alpha, P.beta, P.dose);
            tumor_treat_kernel<<<blocks_1d, THREADS_1D>>>(N, S, us);
        }
        // (3b) Growth: current -> next, then swap so "next" becomes "current".
        tumor_grow_kernel<<<grid, block>>>(P, us, ud);
        double* tmp = us; us = ud; ud = tmp;   // ping-pong the two buffers
    }
    *kernel_ms = timer.stop_ms();              // GPU-measured loop time
    CUDA_CHECK_LAST("tumor kernels");          // catch launch + execution errors

    // (4) After the final swap, `us` holds the latest field. Copy it back into
    //     the caller's vector so main.cu can compare it with the CPU result.
    CUDA_CHECK(cudaMemcpy(u.data(), us, bytes, cudaMemcpyDeviceToHost));

    // (5) Free both device buffers (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
}
