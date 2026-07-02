// ===========================================================================
// src/kernels.cu  --  The three ABM kernels + the host time loop
// ---------------------------------------------------------------------------
// Project 6.9 : Agent-Based Tissue / Immune Simulation
//
// GPU twin of abm_cpu(). Each timestep launches kernels that mirror the CPU
// phases, in the same order, calling the SAME shared abm_core.h math:
//     secrete_kernel  : scatter fixed-point quanta into the grid (atomicAdd)
//     fold_kernel     : add the secreted quanta into the concentration field
//     diffuse_kernel  : one explicit reaction-diffusion stencil step (ping-pong)
//     move_kernel     : integrate cell positions (repulsion + chemotaxis)
// The spatial bins are rebuilt on the host each step (build_bins, shared with the
// CPU) and re-uploaded, so the neighbour scan order matches the CPU exactly.
// main.cu runs both paths and asserts the results agree. See ../THEORY.md.
// ===========================================================================
#include "kernels.cuh"
#include "abm_core.h"
#include "util/cuda_check.cuh"
#include "util/timer.cuh"

#include <utility>   // std::swap
#include <vector>

// 256 threads/block is a solid occupancy default across sm_75..sm_89 for these
// light, memory-bound kernels (both the per-cell and per-grid-cell launches).
static constexpr int THREADS = 256;

// ---------------------------------------------------------------------------
// (1) secrete_kernel: one thread per CELL. A tumor cell adds `q` fixed-point
//   quanta to the grid cell it sits in. Many tumor cells can share a grid cell,
//   so the adds collide -> atomicAdd. Because they are INTEGER adds they commute,
//   so the accumulated total is order-independent (deterministic) AND exactly
//   equals the CPU's serial sum. Immune cells do not secrete. (Pattern: 11.09.)
//   thread i -> cell i.
// ---------------------------------------------------------------------------
__global__ void secrete_kernel(int n, AbmParams p,
                               const double* __restrict__ x,
                               const double* __restrict__ y,
                               const int* __restrict__ type,
                               unsigned long long q,
                               unsigned long long* __restrict__ quanta) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    if (type[i] != CELL_TUMOR) return;
    const int col = abm_col_of(x[i], p.dx, p.gx);
    const int row = abm_row_of(y[i], p.dx, p.gy);
    atomicAdd(&quanta[abm_grid_idx(col, row, p.gx)], q);
}

// ---------------------------------------------------------------------------
// fold_kernel: one thread per GRID CELL. Convert this cell's freshly-secreted
//   quanta to a concentration and add it into the field. Separated from secrete
//   so the atomic scatter (over cells) and the fold (over grid cells) are each a
//   clean, race-free map. thread g -> grid cell g.
// ---------------------------------------------------------------------------
__global__ void fold_kernel(int gc, const unsigned long long* __restrict__ quanta,
                            double* __restrict__ field) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= gc) return;
    field[g] += abm_from_quanta(quanta[g]);
}

// ---------------------------------------------------------------------------
// (2) diffuse_kernel: one thread per GRID CELL. One explicit reaction-diffusion
//   stencil update (shared abm_diffuse_cell): reads c_old, writes c_new. Cells
//   are independent within a step (each writes only its own cell, reads
//   neighbours from the read-only c_old) -> no races, no atomics. (Pattern: 6.04.)
//   We flatten the 2-D grid into a 1-D launch and recover (col,row) by div/mod.
// ---------------------------------------------------------------------------
__global__ void diffuse_kernel(AbmParams p,
                               const double* __restrict__ c_old,
                               double* __restrict__ c_new) {
    const int g = blockIdx.x * blockDim.x + threadIdx.x;
    if (g >= p.gx * p.gy) return;
    const int col = g % p.gx;
    const int row = g / p.gx;
    abm_diffuse_cell(col, row, p, c_old, c_new);
}

// ---------------------------------------------------------------------------
// (3) move_kernel: one thread per CELL. Computes cell i's next position from
//   soft-sphere repulsion (scanning the 3x3 neighbouring bins) plus chemotaxis
//   (immune cells follow grad(chemokine)). All the physics is the shared
//   abm_move_cell; the kernel is just the thread->cell mapping. Reads the
//   read-only current positions and writes new_x/new_y (double-buffered) -> no
//   races. (Pattern: spatial binning, the ABM-specific O(N) neighbour search.)
// ---------------------------------------------------------------------------
__global__ void move_kernel(int n, AbmParams p,
                            const double* __restrict__ x,
                            const double* __restrict__ y,
                            const int* __restrict__ type,
                            const double* __restrict__ field,
                            int bins_x, int bins_y, double bin_size,
                            const int* __restrict__ bin_start,
                            const int* __restrict__ bin_count,
                            const int* __restrict__ sorted,
                            double* __restrict__ new_x,
                            double* __restrict__ new_y) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    abm_move_cell(i, p, x, y, type, field,
                  bins_x, bins_y, bin_size,
                  bin_start, bin_count, sorted, new_x, new_y);
}

// ---------------------------------------------------------------------------
// abm_gpu: allocate device buffers, run the time loop, return the summary.
//   The host rebuilds the spatial bins each step (build_bins, shared with the
//   CPU) and uploads them, so the GPU's neighbour order matches the CPU exactly.
//   This is a deliberate teaching simplification -- see kernels.cuh / THEORY.md
//   for the fully on-GPU binning (Thrust sort-by-key) production codes use.
// ---------------------------------------------------------------------------
AbmResult abm_gpu(const AbmParams& p, const Cells& cells0,
                  std::vector<double>& field_out, float* kernel_ms) {
    const int n  = cells0.n;
    const int gc = p.grid_cells();

    // ---- Device buffers -----------------------------------------------------
    double *d_x = nullptr, *d_y = nullptr, *d_nx = nullptr, *d_ny = nullptr;
    int    *d_type = nullptr;
    double *d_field = nullptr, *d_field_new = nullptr;
    unsigned long long *d_quanta = nullptr;
    // Bin arrays: the bin GRID size is fixed across steps (bin_size + domain are
    // fixed); only the CONTENTS change, so we re-upload the arrays each step.
    int *d_bin_start = nullptr, *d_bin_count = nullptr, *d_sorted = nullptr;

    CUDA_CHECK(cudaMalloc(&d_x,  n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_y,  n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_nx, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_ny, n * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_type, n * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_field,     gc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_field_new, gc * sizeof(double)));
    CUDA_CHECK(cudaMalloc(&d_quanta,    gc * sizeof(unsigned long long)));

    // Upload the initial state. Positions + type are not re-uploaded (they evolve
    // on the device); the field starts at zero.
    CUDA_CHECK(cudaMemcpy(d_x, cells0.x.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_y, cells0.y.data(), n * sizeof(double), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_type, cells0.type.data(), n * sizeof(int), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_field, 0, gc * sizeof(double)));

    // Determine the (fixed) bin-grid size once by binning the initial layout.
    SpatialBins bins;
    build_bins(p, cells0, bins);
    const int nb = bins.bins_x * bins.bins_y;
    CUDA_CHECK(cudaMalloc(&d_bin_start, nb * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_bin_count, nb * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&d_sorted,    n  * sizeof(int)));

    const int grid_blocks = (gc + THREADS - 1) / THREADS;
    const int cell_blocks = (n  + THREADS - 1) / THREADS;

    // Host mirror of the evolving positions (needed to rebuild bins each step).
    Cells snap; snap.n = n; snap.x = cells0.x; snap.y = cells0.y; snap.type = cells0.type;

    const double amount = p.secretion * p.dt;         // chemokine per tumor per step
    const unsigned long long q = abm_to_quanta(amount);

    GpuTimer timer;
    timer.start();
    for (int s = 0; s < p.steps; ++s) {
        // --- (1) SECRETE: zero the quanta, then atomic-scatter over cells.
        CUDA_CHECK(cudaMemset(d_quanta, 0, gc * sizeof(unsigned long long)));
        secrete_kernel<<<cell_blocks, THREADS>>>(n, p, d_x, d_y, d_type, q, d_quanta);
        // Fold the secreted quanta into the field (per grid cell).
        fold_kernel<<<grid_blocks, THREADS>>>(gc, d_quanta, d_field);

        // --- (2) DIFFUSE: one stencil step, then swap the ping-pong buffers.
        diffuse_kernel<<<grid_blocks, THREADS>>>(p, d_field, d_field_new);
        std::swap(d_field, d_field_new);

        // --- (3) MOVE: rebuild + upload the spatial bins (host), then integrate.
        // We copy the CURRENT device positions to the host mirror to bin them; a
        // tiny transfer (2*n doubles) -- teaching clarity over speed.
        CUDA_CHECK(cudaMemcpy(snap.x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(snap.y.data(), d_y, n * sizeof(double), cudaMemcpyDeviceToHost));
        build_bins(p, snap, bins);
        CUDA_CHECK(cudaMemcpy(d_bin_start, bins.bin_start.data(), nb * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_bin_count, bins.bin_count.data(), nb * sizeof(int), cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_sorted,    bins.sorted.data(),    n  * sizeof(int), cudaMemcpyHostToDevice));

        move_kernel<<<cell_blocks, THREADS>>>(n, p, d_x, d_y, d_type, d_field,
                                              bins.bins_x, bins.bins_y, bins.bin_size,
                                              d_bin_start, d_bin_count, d_sorted,
                                              d_nx, d_ny);
        std::swap(d_x, d_nx);
        std::swap(d_y, d_ny);
    }
    *kernel_ms = timer.stop_ms();
    CUDA_CHECK_LAST("abm kernels");

    // ---- Retrieve the final state ------------------------------------------
    std::vector<double> x(n), y(n);
    std::vector<int> type = cells0.type;
    field_out.assign(gc, 0.0);
    CUDA_CHECK(cudaMemcpy(x.data(), d_x, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(y.data(), d_y, n * sizeof(double), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(field_out.data(), d_field, gc * sizeof(double), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_x));   CUDA_CHECK(cudaFree(d_y));
    CUDA_CHECK(cudaFree(d_nx));  CUDA_CHECK(cudaFree(d_ny));
    CUDA_CHECK(cudaFree(d_type));
    CUDA_CHECK(cudaFree(d_field)); CUDA_CHECK(cudaFree(d_field_new));
    CUDA_CHECK(cudaFree(d_quanta));
    CUDA_CHECK(cudaFree(d_bin_start)); CUDA_CHECK(cudaFree(d_bin_count));
    CUDA_CHECK(cudaFree(d_sorted));

    return summarize(p, x, y, type, field_out);
}
