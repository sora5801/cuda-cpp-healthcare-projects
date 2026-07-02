// ===========================================================================
// src/kernels.cu  --  GPU oxygen-field kernel (one thread per grid point)
// ---------------------------------------------------------------------------
// Project 6.21 : Microcirculation & Oxygen Transport
//
// WHAT THIS FILE DOES
//   Implements the device kernel (solve_field_kernel) and the host-side glue
//   (solve_gpu) that allocates GPU memory, uploads the capillary sources,
//   launches the kernel, times it, and brings the PO2 field back. This is the
//   GPU twin of solve_cpu() in reference_cpu.cpp; main.cu runs both and compares.
//
//   The kernel demonstrates the SHARED-MEMORY TILING optimisation: because every
//   grid point sums over EVERY source, the source array is read N_grid times. If
//   each thread read it straight from global memory that would be N_grid*N_src
//   global loads. Instead, the threads of a block COOPERATIVELY stage the sources
//   into on-chip shared memory one TILE at a time, then all threads in the block
//   read that tile from shared memory (about 100x faster than global). This is
//   the same technique as tiled matrix multiply / N-body force summation.
//
//   CRITICAL for verification: the thread still accumulates sources in index
//   order 0..n_src-1 (tiles are processed in order, and within a tile in order),
//   so the double-precision partial sums are added in the SAME order as the CPU's
//   solve_point() -> the two results match to round-off. The tiling changes WHERE
//   the sources are read from, not the ORDER they are summed.
//
// READ THIS AFTER: kernels.cuh (declarations + the thread-mapping idea), oxygen.h
//   (the per-pair physics), reference_cpu.h (solve_point, the CPU twin).
// ===========================================================================
#include "kernels.cuh"
#include "util/cuda_check.cuh"   // CUDA_CHECK, CUDA_CHECK_LAST
#include "util/timer.cuh"        // GpuTimer (CUDA-event timing)

// Threads per block. 128 is a solid default on sm_75..sm_89: a multiple of the
// 32-lane warp, gives the scheduler 4 warps to hide latency, and (with the
// shared-memory tile below sized to blockDim) keeps shared-memory use modest so
// many blocks stay resident. It also sets the source TILE size.
static constexpr int THREADS_PER_BLOCK = 128;

// ---------------------------------------------------------------------------
// solve_field_kernel: one thread computes the PO2 at one grid point.
//   Launch config (set in solve_gpu):
//     grid  = ceil(N_grid / THREADS_PER_BLOCK) blocks
//     block = THREADS_PER_BLOCK threads
//   Thread-to-data map: idx = blockIdx.x * blockDim.x + threadIdx.x  owns grid
//     point idx (the same linear index solve_point()/grid_point_coords() use).
//   Memory: reads sources through a shared-memory tile (declared below); writes
//     one double to po2_out[idx]. No atomics -- each output is owned by exactly
//     one thread, so there is no contention.
//
//   We deliberately RE-DERIVE the arithmetic of solve_point() here rather than
//   calling it directly, because solve_point() reads sources from a linear array;
//   to tile through shared memory we must interleave the loop with __syncthreads.
//   The math (green_function, dist3, mm_consumption, clamp_po2 from oxygen.h) is
//   identical, and summed in the identical order, so CPU==GPU still holds exactly.
// ---------------------------------------------------------------------------
__global__ void solve_field_kernel(TissueGrid grid,
                                    const OxySource* __restrict__ sources,
                                    int n_src,
                                    double* __restrict__ po2_out) {
    // SHARED-MEMORY TILE: one OxySource per thread in the block. All threads in
    // the block will jointly fill this from global memory, then read it many
    // times. Static size = THREADS_PER_BLOCK so the compiler knows it at compile
    // time (no dynamic-shared bookkeeping needed).
    __shared__ OxySource tile[THREADS_PER_BLOCK];

    const int idx = blockIdx.x * blockDim.x + threadIdx.x;  // this thread's grid point
    const int n   = grid_size(grid);

    // Each thread first works out ITS grid point's coordinates. Threads with
    // idx >= n are "dead" (ragged last block) but they must still participate in
    // the cooperative loads and the __syncthreads below, so we do NOT early-return
    // here -- we just skip the final write for them.
    double x = 0.0, y = 0.0, z = 0.0;
    if (idx < n) grid_point_coords(grid, idx, x, y, z);

    // Running superposition, seeded with the inflow PO2 (matches solve_point()).
    double po2 = grid.po2_inflow;

    // Walk the sources TILE BY TILE. Each iteration stages up to blockDim.x
    // sources into shared memory, syncs, then every thread adds that tile's
    // contributions -- in strict index order, preserving the CPU's sum order.
    for (int base = 0; base < n_src; base += blockDim.x) {
        // Cooperative load: thread t loads source (base + t), if it exists.
        const int load_j = base + threadIdx.x;
        if (load_j < n_src) {
            tile[threadIdx.x] = sources[load_j];
        }
        // Make the whole tile visible to every thread before anyone reads it.
        __syncthreads();

        // How many sources landed in this tile (the last tile may be partial).
        const int tile_count = min(blockDim.x, n_src - base);

        // Only live threads accumulate; dead threads still had to help load.
        if (idx < n) {
            for (int t = 0; t < tile_count; ++t) {
                const double r = dist3(x, y, z, tile[t].x, tile[t].y, tile[t].z);
                po2 += tile[t].q * green_function(r);   // same op as solve_point()
            }
        }
        // Re-sync before the next iteration overwrites the tile.
        __syncthreads();
    }

    // Only live threads finish and write. Subtract the background consumption and
    // clamp -- identical to solve_point()'s final two lines.
    if (idx < n) {
        po2 -= mm_consumption(grid.po2_inflow, grid.m0, grid.km);
        po2_out[idx] = clamp_po2(po2);
    }
}

// ---------------------------------------------------------------------------
// solve_gpu: host wrapper. The five canonical steps of a CUDA computation:
//   (1) allocate device memory  (2) copy inputs host->device
//   (3) launch the kernel        (4) copy result device->host
//   (5) free device memory
// We time ONLY step (3) with CUDA events so the reported figure is the kernel
// cost, not the PCIe transfer cost (those are discussed separately in THEORY).
// ---------------------------------------------------------------------------
void solve_gpu(const OxyProblem& problem, std::vector<double>& po2, float* kernel_ms) {
    const int n     = grid_size(problem.grid);
    const int n_src = static_cast<int>(problem.sources.size());
    po2.assign(static_cast<std::size_t>(n), 0.0);

    const std::size_t src_bytes = static_cast<std::size_t>(n_src) * sizeof(OxySource);
    const std::size_t out_bytes = static_cast<std::size_t>(n)     * sizeof(double);

    // (1) Device buffers. The d_ prefix marks DEVICE pointers (CLAUDE.md 12):
    //     dereferencing one on the host would crash, so the naming matters.
    OxySource* d_src = nullptr;
    double*    d_out = nullptr;
    CUDA_CHECK(cudaMalloc(&d_src, src_bytes));   // can fail: out of device memory
    CUDA_CHECK(cudaMalloc(&d_out, out_bytes));

    // (2) Copy the source list H2D. The grid/physiology travels by value in the
    //     kernel argument (TissueGrid is a tiny POD), so nothing to copy for it.
    CUDA_CHECK(cudaMemcpy(d_src, problem.sources.data(), src_bytes, cudaMemcpyHostToDevice));

    // (3) Launch. Blocks must cover all N_grid points, hence the ceiling division
    //     (n + B - 1) / B -- integer-arithmetic "round up".
    const int blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
    GpuTimer timer;
    timer.start();
    solve_field_kernel<<<blocks, THREADS_PER_BLOCK>>>(problem.grid, d_src, n_src, d_out);
    *kernel_ms = timer.stop_ms();          // GPU-measured kernel time
    CUDA_CHECK_LAST("solve_field_kernel"); // catch launch + execution errors

    // (4) Bring the PO2 field back to the host vector.
    CUDA_CHECK(cudaMemcpy(po2.data(), d_out, out_bytes, cudaMemcpyDeviceToHost));

    // (5) Always free what we allocated (no GPU garbage collector exists).
    CUDA_CHECK(cudaFree(d_src));
    CUDA_CHECK(cudaFree(d_out));
}
